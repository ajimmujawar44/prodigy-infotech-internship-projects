# Theory: Neural Style Transfer

## 1. The idea
Neural style transfer (Gatys, Ecker, Bethge, 2015, *A Neural Algorithm of Artistic Style*) takes two images:
- a **content image** (what is in the picture, e.g. a dog), and
- a **style image** (how it looks, e.g. a painting by Kandinsky),

and creates a new image that keeps the content of the first but is "painted" in the style of the second. No network is trained: a **pre-trained image classifier (VGG19)** is used as a fixed feature extractor, and the **pixels of the output image itself are optimised** with gradient descent.

## 2. Why a classification network works
A CNN trained for ImageNet classification must understand images. Early layers respond to edges and colours/textures; deeper layers respond to object parts and arrangements. By reading activations at different depths we get separate descriptions of:
- **content** -> high-level layers (shapes and objects, ignoring exact pixels)
- **style** -> statistics of features over many layers (textures, colours, brush strokes)

## 3. Content representation
The content is the raw **feature map** of a deep layer (`block5_conv2` in this project). The content loss is the mean squared error between the feature maps of the generated image and the content image:

`L_content = mean( (F_generated - F_content)^2 )`

## 4. Style representation: Gram matrix
Style is *where textures occur together*, not *where they are located*. It is captured by the **Gram matrix** of a layer's feature maps: the correlations between channels, averaged over all spatial positions.

`G[c, d] = sum_{i,j} F[i, j, c] * F[i, j, d] / (I * J)`

Because positions are summed out, the Gram matrix discards layout and keeps texture statistics. In code: `tf.linalg.einsum('bijc,bijd->bcd', F, F) / (I*J)`.

Style layers used: `block1_conv1` ... `block5_conv1` (the first conv of each block). Using several layers captures fine strokes (shallow) and larger patterns (deep).

`L_style = (1 / N_layers) * sum_layers mean( (G_generated - G_style)^2 )`

## 5. Total variation (TV) loss
Pure style + content optimisation produces **high-frequency noise** (grainy artifacts). A regulariser penalises large differences between neighbouring pixels:

`L_tv = sum |x[i, j+1] - x[i, j]| + |x[i+1, j] - x[i, j]|`

(`tf.image.total_variation`). This is essentially an edge detector, so minimising it smooths the image.

## 6. Total objective
`L = style_weight * L_style + content_weight * L_content + tv_weight * L_tv`

| Parameter | Default | Effect when increased |
|---|---|---|
| `style_weight` | 1e-2 | More painterly, content shapes get distorted |
| `content_weight` | 1e4 | Output stays closer to the original photo |
| `tv_weight` | 30 | Smoother image, less noise (too high = blurry) |
| `steps` | 1000 | More steps = stronger, cleaner stylisation (diminishing returns) |
| `max_dim` | 512 | Larger = more detail but slower and more memory |

Only the **ratio** style_weight / content_weight matters for the look.

## 7. Optimisation
- The output image is a `tf.Variable` initialised with the content image (faster convergence than random noise).
- Each step: forward pass through VGG19 -> compute losses -> gradient **with respect to the image pixels** (network weights are frozen) -> Adam update -> clip pixels to [0, 1].
- The original paper used **L-BFGS**; Adam (lr 0.02, beta1 0.99, epsilon 0.1) works well and is easy in TensorFlow.
- Input preprocessing: scale to [0, 255] and apply `vgg19.preprocess_input` (BGR + mean subtraction).

## 8. Fast / feed-forward style transfer
The optimisation approach takes minutes per image. Faster alternatives:
- **Perceptual-loss networks** (Johnson et al., 2016): train one feed-forward network per style; one forward pass at test time.
- **Arbitrary style transfer** (Ghiasi et al. 2017 / Magenta; AdaIN, Huang & Belongie 2017): one network handles *any* style image. This project includes it via the TF-Hub model `arbitrary-image-stylization-v1-256` for comparison.

Trade-off: the optimisation method gives more control and often higher fidelity to a given style; fast methods are up to ~1000x faster.

## 9. Limitations and ethics
- Gram matrices ignore spatial structure, so large-scale composition of the style image is not transferred.
- Results depend heavily on hyper-parameters and image pairs; high-resolution output needs lots of GPU memory.
- Use images you have the rights to. Many famous paintings are in the public domain, but photographs of them and modern artworks may be copyrighted. Do not present stylised work as by the original artist.

## 10. References
- Gatys, Ecker, Bethge. *A Neural Algorithm of Artistic Style* (2015) - https://arxiv.org/abs/1508.06576
- Johnson, Alahi, Fei-Fei. *Perceptual Losses for Real-Time Style Transfer and Super-Resolution* (2016)
- Huang & Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (2017)
- Simonyan & Zisserman. *Very Deep Convolutional Networks (VGG)* (2014)
- TensorFlow tutorial: https://www.tensorflow.org/tutorials/generative/style_transfer
