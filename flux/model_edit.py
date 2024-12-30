from dataclasses import dataclass
import os
import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux_edit(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # Define the normal blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )


        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    '''
    对每次的循环都统一输入：
    每一层img_x img_x_hat各自都有自己的输入输出，在特定层替换相应的img
    对每一层的输入输出进行替换，这个所谓的平行替换？
    
    先将img_x_hat传入到block中获取注意力权重，
    根据注意力权重，将对于txt的注意力权重进行排序，然后进行掩膜，选择注意力最大的token进行保留，而将其他的替换为img_x
    '''
    def forward(
        self,
        img: Tensor,
        img2: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_2: Tensor,
        txt_ids_2: Tensor,
        timesteps: Tensor,
        y: Tensor,
        y_2: Tensor,
        guidance: Tensor | None = None,
        replace_double_blocks: list[int] = [],
        replace_single_blocks: list[int] = [],
    ) -> Tensor:
        # 对 x 和 xˆ 同时生成
        # print("img shape: ", img.shape)         # [1, 1024, 64]
        img_x_hat = self.img_in(img)  # 图像嵌入 xˆ (初始相同)
        img_x = self.img_in(img2)  # 图像嵌入 x      # [1, 1024, 3072]

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec_2 = vec + self.vector_in(y_2)
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)
        txt_2 = self.txt_in(txt_2)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        ids_2 = torch.cat((txt_ids_2, img_ids), dim=1)
        pe_2 = self.pe_embedder(ids_2)


        # blocks loop
        # 处理 double_blocks 和 single_blocks
        for i, block in enumerate(self.double_blocks):
            if i in replace_double_blocks:  # 如果当前层在 replace_layers 中，则替换 xˆ 的图像嵌入
                # 让我们先计算一下这一层带来的注意力分布情况
                img2txt_attn, img2img_attn = block(img=img_x_hat, txt=txt, vec=vec, pe=pe, weight_output=True)
                # img2txt is (1, 1024, 512), img2img is (1, 1024, 1024)
                img_avg_atten = img2txt_attn.mean(dim=-1)  # (1, 1024) # 计算每一个img_token对txt_token的平均注意力

                """
                # 进行atten的可视化处理
                save_dir = '/exports/d3/hhr346/flux/attn/ptfiles'  # 替换为你的路径
                os.makedirs(save_dir, exist_ok=True)
                i = 1
                while True:
                    filename = os.path.join(save_dir, f"{i}.pt")  # 使用 .pt 格式保存
                    if not os.path.exists(filename):
                        torch.save(img_avg_atten, filename)
                        print(f"Tensor saved to {filename}")
                        break
                    i += 1
                """

                # 按照比例或者阈值来对替换进行筛选

                RATIO = 0.02            # 比例越大和原图像越接近
                top_percent = int(img_avg_atten.size(1) * RATIO)
                _, indices = torch.sort(img_avg_atten, descending=False)  # 升序排序，返回索引
                mask = torch.zeros_like(img_avg_atten, dtype=torch.bool)  # 初始化布尔掩码
                mask[0, indices[0, :top_percent]] = True  # 前50%的位置为True，其余为False，可能不是按照绝对数目筛选而是强度筛选


                THRESHOLD = 8e-5        # 阈值越大和原图像越接近
                mask = (img_avg_atten < THRESHOLD)
                img_replace = torch.where(mask.unsqueeze(-1), img_x, img_x_hat) # True 保留 img_x, False 保留 img_x_hat
                img_x_hat, txt = block(img=img_replace, txt=txt, vec=vec, pe=pe)


                # 这里测试了一下只替换一半带来的效果
                # img_replace_x = torch.split(img_x, [512, 512], dim=1)[0]
                # img_replace_x_hat = torch.split(img_x_hat, [512, 512], dim=1)[1]
                # img_replace = torch.cat((img_replace_x, img_replace_x_hat), 1)

            else:
                img_x_hat, txt = block(img=img_x_hat, txt=txt, vec=vec, pe=pe)
            img_x, txt_2 = block(img=img_x, txt=txt_2, vec=vec_2, pe=pe_2)

        img_x_hat_cat = torch.cat((txt, img_x_hat), 1)
        img_x_cat = torch.cat((txt_2, img_x), 1)

        for i, block in enumerate(self.single_blocks):
            if i in replace_single_blocks:
                txt, img_x_hat = torch.split(img_x_hat_cat, [512, 1024], dim=1)
                txt_2, img_x = torch.split(img_x_cat, [512, 1024], dim=1)

                img2txt_attn = block(img_x_hat_cat, vec=vec, pe=pe, weight_output=True)
                img_avg_atten = img2txt_attn.mean(dim=-1)  # (1, 1024) # 计算每一个img_token对txt_token的平均注意力

                # Use the ratio
                # top_percent = int(img_avg_atten.size(1) * RATIO)  # 前50%的数量，比例越大和原图像越接近
                # _, indices = torch.sort(img_avg_atten, descending=False)  # 升序排序，返回索引
                # mask = torch.zeros_like(img_avg_atten, dtype=torch.bool)  # 初始化布尔掩码
                # mask[0, indices[0, :top_percent]] = True  # 前50%的位置为True，其余为False，可能不是按照绝对数目筛选而是强度筛选

                # Use the threshold
                mask = (img_avg_atten < THRESHOLD)  # 超过阈值的为True，其他为False

                # Do the replacement
                img_replace = torch.where(mask.unsqueeze(-1), img_x, img_x_hat) # True 保留 img_x, False 保留 img_x_hat

                replace_cat = torch.cat((txt, img_replace), 1)
                img_x_hat_cat = block(replace_cat, vec=vec, pe=pe)  # 用 x 替换 xˆ
            else:
                img_x_hat_cat = block(img_x_hat_cat, vec=vec, pe=pe)
            img_x_cat = block(img_x_cat, vec=vec_2, pe=pe_2)

        img_x_hat = img_x_hat_cat[:, txt.shape[1]:, ...]
        img_x_hat = self.final_layer(img_x_hat, vec)

        img_x = img_x_cat[:, txt_2.shape[1]:, ...]
        img_x = self.final_layer(img_x, vec_2)
        return img_x_hat, img_x


class FluxLoraWrapper(Flux_edit):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
