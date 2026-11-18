"""Optional fast style transfer with the pre-trained TF-Hub arbitrary-stylization model."""
import os

import tensorflow as tf

from src.config import HUB_MODEL_URL

os.environ.setdefault("TFHUB_MODEL_LOAD_FORMAT", "COMPRESSED")


def fast_style_transfer(content_image, style_image):
    """Stylise in a single forward pass (seconds instead of minutes)."""
    import tensorflow_hub as hub  # imported lazily so the rest works without it
    hub_model = hub.load(HUB_MODEL_URL)
    return hub_model(tf.constant(content_image), tf.constant(style_image))[0]
