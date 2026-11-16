"""
Fallback: text-to-image with Hugging Face diffusers (PyTorch).
Use this if the KerasCV weights/versions give you trouble.

python src/generate_images_diffusers.py --prompt "a cute otter in a rainbow whirlpool, watercolor"
"""
import argparse
import os

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for i, img in enumerate(result.images):
        path = os.path.join(args.out_dir, f"diffusers_{i}.png")
        img.save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
