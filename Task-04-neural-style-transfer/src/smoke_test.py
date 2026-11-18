"""
Fast OFFLINE test: no downloads, no pretrained weights needed.
Builds the VGG19 architecture with random weights and runs a few optimisation
steps on synthetic images to verify shapes, losses and gradients.
(The image it produces is not a real style transfer; use `python -m src.stylize`
for that.)

python -m src.smoke_test
"""
import numpy as np
import tensorflow as tf

from src.config import CONTENT_LAYERS, STYLE_LAYERS
from src.image_utils import tensor_to_image
from src.model import StyleContentModel, gram_matrix
from src.style_transfer import run_style_transfer


def main():
    rng = np.random.default_rng(0)
    content = tf.constant(rng.random((1, 128, 160, 3)), dtype=tf.float32)
    style = tf.constant(rng.random((1, 128, 128, 3)), dtype=tf.float32)

    # 1. Gram matrix shape and symmetry
    feats = tf.random.uniform([1, 8, 8, 4])
    g = gram_matrix(feats)
    assert g.shape == (1, 4, 4), g.shape
    assert np.allclose(g.numpy(), np.transpose(g.numpy(), (0, 2, 1)), atol=1e-5)
    print("Gram matrix shape/symmetry OK")

    # 2. Extractor outputs
    extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS, weights=None)
    out = extractor(content)
    assert set(out["style"]) == set(STYLE_LAYERS)
    assert set(out["content"]) == set(CONTENT_LAYERS)
    print("StyleContentModel outputs OK")

    # 3. A few optimisation steps
    result, history = run_style_transfer(content, style, steps=3, log_every=1,
                                         weights=None)
    assert result.shape == content.shape, result.shape
    assert 0.0 <= float(tf.reduce_min(result)) and float(tf.reduce_max(result)) <= 1.0
    assert not np.allclose(result.numpy(), content.numpy()), "image did not change"
    assert len(history["style"]) == 3
    tensor_to_image(result)
    print("Optimisation loop OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
