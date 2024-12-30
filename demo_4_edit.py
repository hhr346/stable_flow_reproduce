'''
这个是初步尝试替换掉所有的img_token但是生成的结果和参考图像基本完全一样
探索了一下替换一部分的img_token，发现在图像一致性上有很大的提升，并且还有几乎理想的生成结果
所以之后需要在新的给定的layer上根据attention的权重来选择性的替换掉部分的img_token
最后要是再探索一下别的真实图像还原重建就更好了

寄希望与返回每个token的位置编码和注意力分布
每一个token在每一层的每一个循环都会有一个全局的注意力分布，你要输出哪一个来看呢？
关键问题是设定恰当的阈值来选取替换掉哪些img_token
目标就是根据每个img_token的注意力分布来决定其是否需要被替换，重点看是否在txt_token上有足够多的注意力，如果有就保留，否则替换
"
'''
import os
import json
import time
import numpy as np
import torch

from einops import rearrange
from PIL import Image
from transformers import pipeline

from flux.cli import SamplingOptions_edit
from flux.sampling import denoise_edit, get_noise, get_schedule, prepare_edit, unpack
from flux.util import embed_watermark, load_ae, load_clip, load_flow_model_edit, load_t5

# 参数设定
MODEL_NAME = "flux-dev"
RESOLUTION = (500, 500)
OUTPUT_FOLDER = "output/edit"
NSFW_THRESHOLD = 0.999


def get_models(name: str, device: torch.device, offload: bool):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model_edit(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, t5, clip, nsfw_classifier


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool):
        self.device = torch.device(device)
        self.offload = offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip, self.nsfw_classifier = get_models(
            model_name, device=self.device, offload=self.offload
        )

    @torch.inference_mode()
    def generate_image(self, prompt, prompt2, seed, init_image=None, img2img_strength=0.0):
        # 定义生成的分辨率
        width, height = RESOLUTION
        replace_double_blocks = []
        replace_single_blocks = []

        for replace_num in [0, 1, 2, 17, 18, 25, 28, 31, 53, 54, 56]:
        # for replace_num in [0, 1, 17, 18, 25, 28, 53, 54, 56]:
        # for replace_num in [0, 1, 17, 18, 25, 28]:
        # for replace_num in [0, 1, 2, 17, 18]:
        # for replace_num in [18]:
            if replace_num < 19:
                replace_double_blocks.append(replace_num)
            else:
                replace_single_blocks.append(replace_num-19)

        # 处理随机种子
        seed = int(seed) if seed != -1 else None
        opts = SamplingOptions_edit(
            prompt=prompt,
            prompt2=prompt2,
            width=width,
            height=height,
            num_steps=50,
            guidance=3.5,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' and '{opts.prompt2}' with seed {opts.seed}")

        # 初始化图像处理
        if init_image is not None:
            init_image = Image.open(init_image).convert("RGB")
            init_image = np.array(init_image).astype(np.float32) / 255.0
            init_image = torch.from_numpy(init_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (opts.height, opts.width))
            init_image = self.ae.encode(init_image)
            init_image = init_image.to(dtype=torch.bfloat16)

        t0 = time.perf_counter()
        # 准备初始噪声
        x = get_noise(1, opts.height, opts.width, device=self.device, dtype=torch.bfloat16, seed=opts.seed)
        timesteps = get_schedule(opts.num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

        if init_image is not None:
            t_idx = int((1 - img2img_strength) * len(timesteps))
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image

        inp = prepare_edit(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt, prompt_2=opts.prompt2)
        x = denoise_edit(self.model, **inp, timesteps=timesteps, guidance=opts.guidance, replace_double_blocks=replace_double_blocks, replace_single_blocks=replace_single_blocks)
        x = unpack(x.float(), opts.height, opts.width)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")

        # 转换为图像格式
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).clamp(0, 255).byte().cpu().numpy())

        return img


def main(json_path: str, img2img: bool = False, init_image_path: str = None):
    # 加载 JSON 参数文件
    with open(json_path, "r") as f:
        params = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = FluxGenerator(MODEL_NAME, device, offload=False)

    # 创建输出目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for key, param in params.items():
        t0 = time.perf_counter()
        prompt = param.get("tuned_prompt", "An empty prompt.")
        prompt2 = param.get("prompt", "An empty prompt.")
        seed = param.get("seed", -1)
        print(f"Generating image for key {key}")

        # 如果启用 img2img 模式但未提供初始图像路径，跳过
        if img2img and not init_image_path:
            print(f"Skipping key {key}: img2img enabled but no init image provided.")
            continue

        # 生成图像
        img = generator.generate_image(prompt, prompt2, seed, init_image=init_image_path if img2img else None)
        # 保存图像
        if img is not None:
            filename = os.path.join(OUTPUT_FOLDER, f"stable_{key}.jpg")
            img.save(filename)
        else:
            print(f"Failed to generate image for key {key}")
        t1 = time.perf_counter()
        print(f"Saved image {filename}, took {t1 - t0:.1f}s.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flux JSON-based Image Generator")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with input parameters.")
    parser.add_argument("--img2img", action="store_true", help="Enable Image-to-Image mode.")
    parser.add_argument("--init_image", type=str, default=None, help="Path to the initial image for img2img.")
    args = parser.parse_args()

    main(args.json_path, args.img2img, args.init_image)
