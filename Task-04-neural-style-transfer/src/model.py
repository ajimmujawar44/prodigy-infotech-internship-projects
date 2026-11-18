"""VGG19 feature extractor producing style (Gram matrices) and content features."""
import tensorflow as tf


def vgg_layers(layer_names, weights="imagenet"):
    """Return a frozen VGG19 that outputs the activations of `layer_names`.

    weights=None builds the same architecture with random weights (only useful
    for offline smoke tests - the result is NOT a meaningful style transfer).
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights=weights)
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)


def gram_matrix(input_tensor):
    """Gram matrix G[c, d] = sum_ij F[i,j,c] * F[i,j,d] / (I*J)  -> (batch, C, C)."""
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(tf.keras.models.Model):
    """Returns {'content': {...feature maps}, 'style': {...Gram matrices}}."""

    def __init__(self, style_layers, content_layers, weights="imagenet"):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers, weights=weights)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0, 1]."""
        inputs = inputs * 255.0
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        style_outputs = [gram_matrix(s) for s in style_outputs]
        content_dict = dict(zip(self.content_layers, content_outputs))
        style_dict = dict(zip(self.style_layers, style_outputs))
        return {"content": content_dict, "style": style_dict}
