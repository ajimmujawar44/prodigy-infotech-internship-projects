# Theory: How Text-to-Image Diffusion Works

## 1. Generative models in one paragraph
A generative model learns the data distribution p(x) so it can sample new, realistic data. Main families: GANs (generator vs. discriminator), VAEs (encoder-decoder with a latent space), autoregressive models (DALL-E 1) and **diffusion models** (Stable Diffusion, DALL-E 2/3, Imagen). Diffusion models are currently the most stable to train and give the best image quality and diversity.

## 2. Diffusion models (DDPM idea)
- **Forward process:** gradually add Gaussian noise to a real image over T steps until it becomes pure noise. This is fixed (no learning).
- **Reverse process:** train a neural network to undo one step of noise. In practice the network predicts the **noise** added to a noisy image at timestep *t*; the training loss is the MSE between predicted and true noise.
- **Sampling:** start from random noise and repeatedly ask the network to remove a little noise (about 20-50 steps with modern samplers). A brand-new image appears.
- This is the same idea as super-resolution/denoising: "denoise pure noise" and the model *hallucinates* details that fit its training distribution.

## 3. Latent diffusion (why Stable Diffusion is efficient)
Running diffusion on 512x512x3 pixels is expensive. Latent Diffusion Models (Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv 2112.10752, 2021) first compress images with a **VAE** into a small latent (64x64x4), run diffusion there, and decode back to pixels. This cuts compute by roughly an order of magnitude with little quality loss.

## 4. The three components of Stable Diffusion
| Part | Role |
|---|---|
| **Text encoder** (CLIP text transformer, frozen) | Turns the prompt into a sequence of embeddings (77 tokens x 768) |
| **Diffusion model** (U-Net with ResNet + attention blocks) | Repeatedly denoises the 64x64 latent, conditioned on text and timestep |
| **Decoder** (VAE decoder) | Converts the final 64x64 latent into a 512x512 RGB image |

## 5. Conditioning on text (cross-attention)
Prompt embeddings are injected into the U-Net through **cross-attention** layers: image-latent features are the *queries*, text embeddings are the *keys/values*. This is how words steer the picture. The model is trained on huge {image, caption} pair datasets (LAION).

## 6. Classifier-free guidance (guidance scale)
At every step the U-Net is run twice: with the prompt (conditional) and with an empty prompt (unconditional). The final noise estimate is:

`eps = eps_uncond + s * (eps_cond - eps_uncond)`

`s` is the **guidance scale** (default 7.5). Higher = follows the prompt more strictly but is less diverse and can show artifacts; lower = more creative but less faithful. A **negative prompt** replaces the empty prompt, pushing the image *away* from unwanted concepts.

## 7. Key generation parameters
| Parameter | Effect |
|---|---|
| `num_steps` (default 50) | More steps = cleaner image but slower; gains flatten after about 30-50 |
| `unconditional_guidance_scale` | Prompt adherence vs. creativity |
| `seed` | Same seed + same prompt = reproducible image |
| `negative_prompt` | Concepts to avoid |
| `batch_size` | Number of images generated per call |
| image size | SD v1 was trained at 512x512; much larger sizes cause duplicated or odd structures |

## 8. Speed optimisations used in KerasCV
- **Graph mode + XLA (`jit_compile=True`)**: compiles the model into fused GPU kernels.
- **Mixed precision (`mixed_float16`)**: computes in float16 for fast tensor-core kernels while keeping weights in float32 for numerical stability.
- Combined, the reference tutorial reports about a 2x speed-up (15.0s to 7.5s for 3 images on an A100).

## 9. Prompt engineering tips
Subject + style + medium + quality/lighting keywords, for example *"a cute magical flying dog, fantasy art, golden color, highly detailed, sharp focus, concept art, digital painting"*. Be specific, iterate, and fix the seed while comparing prompt changes.

## 10. Limitations and ethics
- Can reproduce biases from web-scale training data; struggles with text in images, hands, counting and precise spatial relations.
- Model licence: **CreativeML Open RAIL-M** forbids harmful uses (deception, harassment, non-consensual imagery, etc.).
- Consider copyright and consent of training data, and label AI-generated images when sharing.

## 11. Further reading
- Ho et al., *Denoising Diffusion Probabilistic Models* (2020)
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2021)
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022)
- Radford et al., *CLIP* (2021)
- Keras.io: Denoising Diffusion Implicit Models; KerasCV Stable Diffusion guide
