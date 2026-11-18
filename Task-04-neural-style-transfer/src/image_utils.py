"""Image loading, conversion and plotting helpers."""
import os

import matplotlib
matplotlib.use("Agg")  # safe in terminals without a display
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf

from src.config import MAX_DIM


def load_img(path_to_img, max_dim=MAX_DIM):
    """Load an image as a float32 tensor in [0, 1], shape (1, H, W, 3).

    The longest side is scaled down/up to `max_dim` pixels.
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]


def tensor_to_image(tensor):
    """Convert a [0, 1] float tensor (optionally with batch dim) to a PIL image."""
    tensor = np.array(tensor) * 255
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    if tensor.ndim > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def save_image(tensor, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tensor_to_image(tensor).save(path)
    return path


def save_comparison(content, style, result, path, result_title="Stylized Image"):
    """Save a 1x3 figure: content | style | result."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.figure(figsize=(18, 6))
    for i, (img, title) in enumerate(
            [(content, "Content Image"), (style, "Style Image"), (result, result_title)]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(tf.squeeze(img, axis=0) if len(img.shape) > 3 else img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_loss_curves(history, path):
    """Plot style / content / total-variation loss per logged step."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.figure(figsize=(12, 4))
    for name in ("style", "content", "tv"):
        plt.plot(history[name], label=f"{name} loss")
    plt.yscale("log")
    plt.xlabel("Logged step")
    plt.ylabel("Weighted loss (log scale)")
    plt.legend()
    plt.title("Style transfer loss curves")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path
