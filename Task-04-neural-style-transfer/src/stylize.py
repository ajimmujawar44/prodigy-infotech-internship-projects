"""
Command-line neural style transfer.

Examples
--------
python -m src.stylize                                   # sample dog + Kandinsky
python -m src.stylize --content images/me.jpg --style images/starry_night.jpg --steps 1000
python -m src.stylize --fast                            # TF-Hub fast version
"""
import argparse
import os

import tensorflow as tf

from src import config
from src.image_utils import (load_img, save_comparison, save_image,
                             save_loss_curves)


def parse_args():
    p = argparse.ArgumentParser(description="Neural style transfer (Gatys et al.)")
    p.add_argument("--content", help="Content image path (default: sample Labrador)")
    p.add_argument("--style", help="Style image path (default: Kandinsky Composition 7)")
    p.add_argument("--output", default=None, help="Output file (default: outputs/stylized.png)")
    p.add_argument("--steps", type=int, default=config.STEPS)
    p.add_argument("--max-dim", type=int, default=config.MAX_DIM)
    p.add_argument("--style-weight", type=float, default=config.STYLE_WEIGHT)
    p.add_argument("--content-weight", type=float, default=config.CONTENT_WEIGHT)
    p.add_argument("--tv-weight", type=float, default=config.TV_WEIGHT)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--fast", action="store_true",
                   help="Use the pre-trained TF-Hub model instead of optimisation")
    return p.parse_args()


def main():
    args = parse_args()
    content_path = args.content or tf.keras.utils.get_file(
        "YellowLabradorLooking_new.jpg", config.CONTENT_URL)
    style_path = args.style or tf.keras.utils.get_file(
        "kandinsky5.jpg", config.STYLE_URL)
    output = args.output or os.path.join(config.OUTPUT_DIR, "stylized.png")

    content_image = load_img(content_path, args.max_dim)
    style_image = load_img(style_path, args.max_dim)

    if args.fast:
        from src.fast_style_transfer import fast_style_transfer
        result, history = fast_style_transfer(content_image, style_image), None
    else:
        from src.style_transfer import run_style_transfer
        result, history = run_style_transfer(
            content_image, style_image, steps=args.steps,
            style_weight=args.style_weight, content_weight=args.content_weight,
            tv_weight=args.tv_weight, learning_rate=args.lr)

    save_image(result, output)
    comparison = os.path.join(os.path.dirname(output), "comparison.png")
    save_comparison(content_image, style_image, result, comparison)
    print("Saved:", output)
    print("Saved:", comparison)
    if history:
        print("Saved:", save_loss_curves(
            history, os.path.join(os.path.dirname(output), "loss_curves.png")))


if __name__ == "__main__":
    main()
