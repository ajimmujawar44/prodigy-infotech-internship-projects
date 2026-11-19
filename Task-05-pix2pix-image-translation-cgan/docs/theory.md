# Theory: Image-to-Image Translation with pix2pix (cGAN)

## 1. GAN refresher
A Generative Adversarial Network has two networks trained against each other:
- **Generator (G):** produces fake samples from noise.
- **Discriminator (D):** classifies samples as real or fake.

G tries to fool D; D tries not to be fooled. At equilibrium, G's samples are indistinguishable from real data.

## 2. Conditional GAN (cGAN)
A cGAN (Mirza & Osindero, 2014) feeds extra information (a condition `x`) to both networks, so we learn `p(y | x)` instead of `p(y)`. In pix2pix the condition is an **input image** and the output is the **translated image**.

Objective: `G* = arg min_G max_D  L_cGAN(G, D) + lambda * L_L1(G)`

## 3. Image-to-image translation
Learn a mapping between two image domains using **paired** data (input, target). Examples from the paper: labels -> photo (this project: architecture labels -> building facade), edges -> photo, B&W -> colour, day -> night, map -> aerial image, sketch -> photo.

## 4. Generator: U-Net
An encoder-decoder with **skip connections**.
- **Encoder (downsample):** Conv(4x4, stride 2) -> BatchNorm -> LeakyReLU. 256x256 -> 1x1 bottleneck.
- **Decoder (upsample):** TransposedConv -> BatchNorm -> Dropout (first 3 blocks) -> ReLU. 1x1 -> 256x256.
- **Skip connections:** each decoder layer is concatenated with the mirrored encoder layer, so low-level details (edges, positions) bypass the bottleneck. Without skips, fine structure would be lost.
- Output layer: Conv2DTranspose with **tanh**, so pixels are in [-1, 1].

## 5. Discriminator: PatchGAN
Instead of one real/fake score for the whole image, the discriminator outputs a **30x30 grid**; each cell judges a **70x70 patch** of the input. This:
- focuses on local texture/high-frequency realism,
- has fewer parameters and is faster,
- works on any image size.

It receives **two images concatenated on the channel axis**: `[input, target]` (should be real) or `[input, generated]` (should be fake). Conditioning on the input checks that the output actually *matches* the input.

## 6. Losses
| Loss | Formula | Purpose |
|---|---|---|
| Generator GAN loss | BCE(ones, D(input, G(input))) | Fool the discriminator (realistic look) |
| Generator L1 loss | mean(abs(target - G(input))) | Stay close to the ground truth (correct structure, less blur than L2) |
| **Total generator loss** | `gan_loss + 100 * l1_loss` | `LAMBDA = 100` from the paper |
| Discriminator loss | BCE(ones, D(input, target)) + BCE(zeros, D(input, G(input))) | Tell real from fake |

L1 captures low-frequency correctness; the GAN term adds sharp high-frequency detail.

## 7. Training details
- **Data augmentation (random jitter):** resize 256 -> 286, random-crop 256, random horizontal flip. Applied identically to input and target.
- **Normalisation:** pixels scaled to [-1, 1].
- **Batch size 1**, **Adam** (lr = 2e-4, beta1 = 0.5) for both networks.
- **Dropout at inference/training** acts as the noise source (no explicit z vector).
- `training=True` is used when generating test images so BatchNorm uses batch statistics (as in the paper).
- Facades: ~40k steps for decent results (tutorial shows 200 epochs = 80k steps). About 15 s/epoch on a V100.

## 8. Reading the training logs
- `log(2) = 0.69` is the reference value: the discriminator is equally uncertain about real and fake.
- If `disc_loss` -> near 0, the discriminator is winning; if `gen_gan_loss` is very low while `disc_loss` is high, the generator is winning. Either way training is unbalanced.
- `gen_l1_loss` should decrease steadily.
- GAN losses are not a quality metric; always **look at the generated images**.

## 9. Limitations
- Needs **paired** training data (unpaired translation needs CycleGAN).
- Deterministic-looking outputs (little diversity); small dataset (400 images) limits quality.
- Can show artifacts/blur on unusual inputs.

## 10. References
- Isola, Zhu, Zhou, Efros. *Image-to-Image Translation with Conditional Adversarial Networks* (2017) - https://arxiv.org/abs/1611.07004
- Mirza & Osindero. *Conditional Generative Adversarial Nets* (2014) - https://arxiv.org/abs/1411.1784
- Ronneberger et al. *U-Net* (2015) - https://arxiv.org/abs/1505.04597
- Goodfellow et al. *Generative Adversarial Networks* (2014)
- TensorFlow tutorial: https://www.tensorflow.org/tutorials/generative/pix2pix
