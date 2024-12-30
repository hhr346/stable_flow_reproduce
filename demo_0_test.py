import os
import json
import time
import numpy as np
import torch

from einops import rearrange
from PIL import Image
from transformers import pipeline

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import embed_watermark, load_ae, load_clip, load_flow_model, load_t5

# 参数设定
MODEL_NAME = "flux-dev"
RESOLUTION = (500, 500)
OUTPUT_FOLDER = "output/edit"
NSFW_THRESHOLD = 0.999


def get_models(name: str, device: torch.device, offload: bool):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
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
    def generate_image(self, prompt, seed, init_image=None, img2img_strength=0.0):
        # 定义生成的分辨率
        width, height = RESOLUTION

        # 处理随机种子
        seed = int(seed) if seed != -1 else None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=50,
            guidance=3.5,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        # 初始化图像处理
        if init_image is not None:
            init_image = Image.open(init_image).convert("RGB")
            init_image = np.array(init_image).astype(np.float32) / 255.0
            init_image = torch.from_numpy(init_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (opts.height, opts.width))
            init_image = self.ae.encode(init_image)
            init_image = init_image.to(dtype=torch.bfloat16)

        # 准备初始噪声
        x = get_noise(1, opts.height, opts.width, device=self.device, dtype=torch.bfloat16, seed=opts.seed)
        timesteps = get_schedule(opts.num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

        if init_image is not None:
            t_idx = int((1 - img2img_strength) * len(timesteps))
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        print(f"the timestep is {timesteps}, the length of timestep is {len(timesteps)}")
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=opts.guidance)

        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")

        # 转换为图像格式
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).clamp(0, 255).byte().cpu().numpy())

        # 检测 NSFW
        nsfw_score = [x["score"] for x in self.nsfw_classifier(img) if x["label"] == "nsfw"][0]
        if nsfw_score >= NSFW_THRESHOLD:
            print("Generated image may contain NSFW content. Skipping save.")
            return None

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
        seed = param.get("seed", -1)
        print(f"Generating image for key {key}")

        # 如果启用 img2img 模式但未提供初始图像路径，跳过
        if img2img and not init_image_path:
            print(f"Skipping key {key}: img2img enabled but no init image provided.")
            continue

        # 生成图像
        img = generator.generate_image(prompt, seed, init_image=init_image_path if img2img else None)

        # 保存图像
        if img is not None:
            filename = os.path.join(OUTPUT_FOLDER, f"edit_{key}.jpg")
            img.save(filename)
            t1 = time.perf_counter()
            print(f"Saved image {filename}, took {t1 - t0:.1f}s.")
        else:
            print(f"Failed to generate image for key {key}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flux JSON-based Image Generator")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with input parameters.")
    parser.add_argument("--img2img", action="store_true", help="Enable Image-to-Image mode.")
    parser.add_argument("--init_image", type=str, default=None, help="Path to the initial image for img2img.")
    args = parser.parse_args()

    main(args.json_path, args.img2img, args.init_image)
