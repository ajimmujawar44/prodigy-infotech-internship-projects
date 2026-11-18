"""Gatys et al. neural style transfer by direct image optimisation."""
import time

import tensorflow as tf

from src.config import (BETA_1, CONTENT_LAYERS, CONTENT_WEIGHT, EPSILON,
                        LEARNING_RATE, STEPS, STYLE_LAYERS, STYLE_WEIGHT,
                        TV_WEIGHT)
from src.model import StyleContentModel


def clip_0_1(image):
    return tf.clip_by_value(image, 0.0, 1.0)


def run_style_transfer(content_image, style_image, steps=STEPS,
                       style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT,
                       tv_weight=TV_WEIGHT, learning_rate=LEARNING_RATE,
                       style_layers=STYLE_LAYERS, content_layers=CONTENT_LAYERS,
                       weights="imagenet", log_every=100, callback=None):
    """Optimise a copy of `content_image` so it takes on the style of `style_image`.

    Total loss = style_weight * style_loss + content_weight * content_loss
                 + tv_weight * total_variation_loss

    Returns (stylized_image_tensor, history_dict).
    """
    extractor = StyleContentModel(style_layers, content_layers, weights=weights)
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]

    image = tf.Variable(content_image)       # start from the content image
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=BETA_1,
                                   epsilon=EPSILON)
    num_style, num_content = len(style_layers), len(content_layers)

    @tf.function()
    def train_step(img):
        with tf.GradientTape() as tape:
            outputs = extractor(img)
            style_loss = tf.add_n([
                tf.reduce_mean((outputs["style"][n] - style_targets[n]) ** 2)
                for n in outputs["style"]]) * (style_weight / num_style)
            content_loss = tf.add_n([
                tf.reduce_mean((outputs["content"][n] - content_targets[n]) ** 2)
                for n in outputs["content"]]) * (content_weight / num_content)
            tv_loss = tv_weight * tf.reduce_sum(tf.image.total_variation(img))
            loss = style_loss + content_loss + tv_loss
        grad = tape.gradient(loss, img)
        opt.apply_gradients([(grad, img)])
        img.assign(clip_0_1(img))
        return style_loss, content_loss, tv_loss

    history = {"style": [], "content": [], "tv": []}
    start = time.time()
    for step in range(1, steps + 1):
        s, c, t = train_step(image)
        if step % log_every == 0 or step == steps:
            history["style"].append(float(s))
            history["content"].append(float(c))
            history["tv"].append(float(t))
            print(f"step {step}/{steps}  style={float(s):.2f}  "
                  f"content={float(c):.2f}  tv={float(t):.2f}")
            if callback:
                callback(step, image)
    print(f"Total time: {time.time() - start:.1f}s")
    return image.read_value(), history
