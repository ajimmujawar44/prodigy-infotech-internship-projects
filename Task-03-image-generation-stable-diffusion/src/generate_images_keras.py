"""
Text-to-image generation with KerasCV Stable Diffusion.

Examples
--------
python src/generate_images_keras.py --prompt "photograph of an astronaut riding a horse"
python src/generate_images_keras.py --prompts-file prompts/prompts.txt --batch-size 3 --fast
"""
import argparse
import os
import time

import keras_cv
from tensorflow import keras
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="KerasCV Stable Diffusion text-to-image")
    p.add_argument("--prompt", type=str, help="Single text prompt")
    p.add_argument("--prompts-file", type=str, help="Text file, one prompt per line")
    p.add_argument("--negative-prompt", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=3, help="Images per prompt")
    p.add_argument("--steps", type=int, default=50, help="Denoising steps")
    p.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    p.add_argument("--size", type=int, default=512, help="Image width/height (multiple of 128)")
    p.add_argument("--fast", action="store_true", help="Enable mixed precision + XLA")
    p.add_argument("--out-dir", type=str, default="outputs")
    return p.parse_args()


def load_prompts(args):
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.prompt:
        return [args.prompt]
    raise SystemExit("Provide --prompt or --prompts-file")


def slugify(text, n=40):
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:n].strip("_")


def main():
    args = parse_args()
    prompts = load_prompts(args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.fast:
        keras.mixed_precision.set_global_policy("mixed_float16")
    model = keras_cv.models.StableDiffusion(
        img_width=args.size, img_height=args.size, jit_compile=args.fast
    )

    for i, prompt in enumerate(prompts, 1):
        start = time.time()
        images = model.text_to_image(
            prompt,
            negative_prompt=args.negative_prompt,
            batch_size=args.batch_size,
            num_steps=args.steps,
            unconditional_guidance_scale=args.guidance,
            seed=args.seed,
        )
        elapsed = time.time() - start
        for j, img in enumerate(images):
            path = os.path.join(args.out_dir, f"{i:02d}_{slugify(prompt)}_{j}.png")
            Image.fromarray(img).save(path)
        print(f"[{i}/{len(prompts)}] {elapsed:.1f}s -> {prompt[:60]}")

    print("Done. Images saved in", args.out_dir)


if __name__ == "__main__":
    main()
