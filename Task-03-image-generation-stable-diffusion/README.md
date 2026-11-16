# Image Generation with Pre-trained Models (Stable Diffusion)

**Author:** [Ajim Mujawar]
**Track:** Generative AI - Task 02

Generate images from text prompts using a pre-trained **Stable Diffusion** model (KerasCV implementation), and study the theory behind latent diffusion models.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Sample Output](#sample-output)
- [Ethics and Limitations](#ethics-and-limitations)
- [References](#references)

## Overview
Stable Diffusion is an open-source text-to-image model. This project:
1. Loads the pre-trained model with `keras_cv.models.StableDiffusion`.
2. Generates images from simple and detailed prompts.
3. Explores `num_steps`, guidance scale, seed and negative prompts.
4. Benchmarks speed with mixed precision and XLA compilation.

## How It Works
Stable Diffusion is a **latent diffusion model** with three parts:

| Component | Purpose |
|---|---|
| Text encoder (CLIP, frozen) | Converts the prompt into embeddings |
| Diffusion model (U-Net) | Denoises a 64x64 latent over ~50 steps, guided by the text |
| Decoder (VAE) | Converts the final latent into a 512x512 image |

Generation starts from random noise; the model removes noise step by step while being steered by the prompt (cross-attention + classifier-free guidance). Detailed notes: [`docs/theory.md`](docs/theory.md).

## Project Structure
```
image-generation-stable-diffusion/
├── notebooks/
│   └── stable_diffusion_image_generation.ipynb   # Full walkthrough + experiments
├── src/
│   ├── generate_images_keras.py                  # CLI script (KerasCV)
│   └── generate_images_diffusers.py              # Optional PyTorch fallback
├── prompts/
│   └── prompts.txt                               # Example prompts
├── docs/
│   └── theory.md                                 # Theory notes
├── outputs/                                      # Generated images (git-ignored)
├── requirements.txt
├── requirements-diffusers.txt
├── .gitignore
└── README.md
```

## Installation
A **GPU** is strongly recommended (e.g. free Google Colab T4).

```bash
git clone <your-repo-url>
cd image-generation-stable-diffusion
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Model weights (~4 GB) download automatically on first run.

## Usage
**Notebook (recommended):** open `notebooks/stable_diffusion_image_generation.ipynb` in Jupyter or upload it to Google Colab (GPU runtime) and run all cells.

**Command line:**
```bash
# One prompt
python src/generate_images_keras.py --prompt "photograph of an astronaut riding a horse"

# All example prompts, faster with mixed precision + XLA
python src/generate_images_keras.py --prompts-file prompts/prompts.txt --fast

# Control parameters
python src/generate_images_keras.py --prompt "a cozy cabin in snow" --steps 30 --guidance 9 --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--prompt` / `--prompts-file` | - | Text prompt, or file with one prompt per line |
| `--negative-prompt` | None | Concepts to avoid |
| `--batch-size` | 3 | Images per prompt |
| `--steps` | 50 | Denoising steps |
| `--guidance` | 7.5 | Classifier-free guidance scale |
| `--seed` | None | Reproducibility |
| `--fast` | off | Mixed precision + XLA |
| `--out-dir` | outputs | Where images are saved |

**PyTorch fallback** (if KerasCV versions cause trouble):
```bash
pip install -r requirements-diffusers.txt
python src/generate_images_diffusers.py --prompt "a cute otter in a rainbow whirlpool, watercolor"
```

## Experiments
| Experiment | What changes | What to observe |
|---|---|---|
| Steps (5/15/30/50) | `num_steps` | Quality improves, then plateaus |
| Guidance (1/4/7.5/15) | `unconditional_guidance_scale` | Creativity vs. prompt fidelity |
| Seeds | `seed` | Same seed = same image |
| Negative prompt | `negative_prompt` | Removes unwanted traits |
| Speed | Mixed precision + XLA | Faster inference (reference reports ~2x on A100) |

## Sample Output
After running, your images are saved in `outputs/`. Add your favourites here, e.g.:

```markdown
![Astronaut riding a horse](outputs/01_astronaut_horse_0.png)
```

## Ethics and Limitations
- Models inherit biases from web-scale training data and can struggle with text, hands, and counting.
- Usage is subject to the **CreativeML Open RAIL-M** licence; do not generate harmful, deceptive or non-consensual content.
- Label AI-generated images when sharing.

## References
- Keras tutorial: *Generate images with Stable Diffusion* (fchollet, lukewood, divamgupta) - https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* - https://arxiv.org/abs/2112.10752
- Ho et al., *Denoising Diffusion Probabilistic Models* (2020)
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022)

## Author
**[Ajim Mujawar]**
