# GPT-2 Text Generation — Fine-Tuning Project

Fine-tune GPT-2 on a small custom text corpus, then generate new text from a prompt. This repository is designed as a beginner-friendly Generative AI learning project: the code is intentionally small, while the README explains the theory behind each step.

> **Important:** The included corpus is a tiny demonstration dataset, not a production training dataset. Results will be much better with a larger, clean, consistent dataset. This project demonstrates the workflow; it does not train GPT-2 from scratch.

## Project goals

- Understand autoregressive text generation and the Transformer decoder.
- Fine-tune a pretrained GPT-2 causal language model on custom text.
- Generate text from a user-supplied prompt.
- Experiment with sampling settings such as `temperature`, `top_k`, and `top_p`.
- Evaluate outputs critically for relevance, repetition, factuality, and bias.

## Repository structure

```text
gpt2-text-generation/
├── data/
│   └── sample_train.txt       # Small example corpus; replace with your own text
├── outputs/gpt2-custom/        # Fine-tuned inference model and tokenizer (weights via Git LFS)
├── src/
│   ├── train.py               # Fine-tune pretrained GPT-2
│   └── generate.py            # Generate text from a prompt
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 1. Environment setup

Python 3.10 or 3.11 is recommended.

```bash
# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# python3 -m venv .venv
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the GPT-2 tokenizer and model weights from Hugging Face, so an internet connection is required. A GPU is helpful but not mandatory. CPU fine-tuning can be slow.

## 2. Prepare your custom dataset

The included `data/sample_train.txt` is a tiny example corpus about learning and responsible AI. For a real experiment, replace it with text that you have permission to use.

- Use clean, relevant text in a consistent style and domain.
- Remove private or sensitive information.
- Remove duplicated, corrupted, and irrelevant text.
- Keep a backup of the original dataset.
- A larger corpus usually works better than the included demo file.
- Do not put secrets, API keys, personal records, or copyrighted material you are not authorized to use into a training corpus.

The script treats each non-empty line as one training example. Keep each line self-contained and reasonably short for this beginner project.

## 3. Fine-tune GPT-2

From the repository root:

```bash
python src/train.py --data_path data/sample_train.txt --output_dir outputs/gpt2-custom --epochs 3
```

Optional settings:

```bash
python src/train.py \
  --data_path data/sample_train.txt \
  --output_dir outputs/gpt2-custom \
  --model_name gpt2 \
  --epochs 3 \
  --batch_size 2 \
  --max_length 128 \
  --learning_rate 5e-5
```

On Windows PowerShell, the one-line command is simplest if line continuations cause problems.

The training script saves the model and tokenizer into `outputs/gpt2-custom/`. Use more epochs only after checking for overfitting and output quality. The tiny sample corpus is for verifying that the pipeline runs, not for judging model quality.

## 4. Generate text

The fine-tuned inference model is included in `outputs/gpt2-custom/`. Its model weights are stored with Git LFS; install Git LFS before cloning to download them:

```bash
git lfs install
git clone https://github.com/ajimmujawar44/prodigy-infotech-internship-projects.git
```

```bash
python src/generate.py \
  --model_path outputs/gpt2-custom \
  --prompt "Generative AI can help people learn because"
```

Try different sampling controls:

```bash
python src/generate.py --model_path outputs/gpt2-custom --prompt "A responsible AI system" --max_new_tokens 80 --temperature 0.8 --top_k 50 --top_p 0.92
```

If you want to compare against the original pretrained GPT-2 without fine-tuning:

```bash
python src/generate.py --model_path gpt2 --prompt "Generative AI can help people learn because"
```

## 5. How it works: theory

### What is Generative AI?

Generative AI learns patterns from examples and uses those patterns to produce new content, such as text, images, audio, or code. A language model learns statistical relationships between tokens. It does not store a perfect database of facts and can generate incorrect or fabricated statements.

### What is GPT-2?

GPT-2 is a pretrained, decoder-only Transformer language model. “Pretrained” means it learned general language patterns from a large text corpus before this project. Fine-tuning continues training on a smaller domain-specific dataset so its generations may better reflect that dataset's style and vocabulary.

### Tokenization

The model does not directly read words as humans do. A tokenizer converts text into token IDs. A token can be a whole word, part of a word, punctuation, or whitespace-related text. The model predicts token IDs, and the tokenizer converts generated IDs back into readable text.

### Autoregressive next-token prediction

Given a token sequence \(x_1, x_2, \ldots, x_t\), a causal language model estimates the next token:

\[
P(x_{t+1}\mid x_1, x_2, \ldots, x_t)
\]

It generates one token at a time. The newly generated token becomes part of the context for predicting the next token. This is why the prompt matters and why long generations can drift away from the original topic.

### Transformer and causal attention

GPT-2 uses Transformer decoder blocks. Self-attention lets each token use information from earlier tokens. A causal mask prevents a token from attending to future tokens during training, preserving the left-to-right prediction setup. Multi-head attention learns different relationships; feed-forward layers transform token representations; residual connections and normalization help optimization.

### Fine-tuning and loss

Training examples are converted to token IDs. The model predicts each next token and compares its predictions with the actual next tokens using cross-entropy loss. Backpropagation computes gradients, and an optimizer updates model weights. Fine-tuning starts from pretrained weights rather than random initialization, so it usually needs less data and compute than training from scratch.

The training script uses a **causal language modeling** objective. `DataCollatorForLanguageModeling(mlm=False)` prepares labels for next-token prediction; this is not masked-language modeling like BERT.

### Generation controls

- **`max_new_tokens`**: maximum number of new tokens, not counting the prompt.
- **`temperature`**: rescales next-token probabilities. Lower values are more conservative; higher values are more varied and can become less coherent.
- **`top_k`**: samples only from the `k` most likely candidate tokens when greater than zero.
- **`top_p`**: nucleus sampling; samples from the smallest candidate set whose cumulative probability reaches `p`.
- **`do_sample`**: enables sampling. When enabled, results can differ between runs.

No single setting guarantees coherence. Compare multiple outputs and evaluate them rather than assuming fluent text is accurate.

## 6. Experiment log (complete after running)

Record your experiments so the project demonstrates understanding, not only code execution.

| Experiment | Dataset | Epochs | Temperature | Top-k | Top-p | Observations |
|---|---|---:|---:|---:|---:|---|
| Baseline | Pretrained GPT-2 | — | 0.8 | 50 | 0.92 | Fill in after generation |
| Fine-tuned | Your corpus | 3 | 0.8 | 50 | 0.92 | Fill in after generation |

Suggested questions:
1. Does fine-tuning change vocabulary, tone, or structure?
2. What happens when temperature changes from 0.4 to 1.2?
3. Does a very small dataset cause the model to repeat training phrases?
4. Does increasing epochs improve relevance or cause overfitting?
5. Which output is more relevant to the prompt, and how did you judge it?

## 7. Limitations and responsible use

- A small dataset can cause memorization, repetition, and poor generalization.
- GPT-2 can produce plausible but false or biased text.
- Fine-tuning does not guarantee factual accuracy or safety.
- Use data only when you have permission and remove sensitive information.
- Do not treat generated text as professional, medical, legal, or financial advice without qualified review.
- Evaluate output quality with human review and, for larger projects, suitable quantitative metrics.

## Troubleshooting

**`ModuleNotFoundError`**: activate the intended virtual environment and run `pip install -r requirements.txt`.

**CUDA out-of-memory**: lower `--batch_size` and `--max_length`; use CPU if necessary.

**Very poor generations**: the included corpus is intentionally tiny. Replace it with a larger, clean corpus and compare the pretrained baseline with the fine-tuned model.

**PowerShell line-continuation errors**: run the command on one line.

## Learning references

- Hugging Face text generation guide: https://huggingface.co/blog/how-to-generate
- Hugging Face Transformers GPT-2 documentation: https://huggingface.co/docs/transformers/model_doc/gpt2
- Hugging Face causal language modeling guide: https://huggingface.co/docs/transformers/tasks/language_modeling
- OpenAI GPT-2 announcement and research: https://openai.com/index/better-language-models/
- Original GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

## What to include in your GitHub submission

1. This README with setup instructions and theory.
2. Training and generation scripts.
3. Your dataset source and license/permission notes.
4. A short experiment log with prompts and generated outputs.
5. Screenshots only if they add evidence (for example, terminal training and generated output).
6. Do not commit intermediate training checkpoints, virtual environments, private data, or secrets. This repository includes only the final inference model and tokenizer; its large weight file is stored with Git LFS.

## License

The code in this repository is provided under the MIT License. Dataset and model weights remain subject to their own licenses and terms.

## Author

**Ajim Sikandar Mujawar**

Aspiring Data Scientist and AI Engineer passionate about Generative AI, Natural Language Processing, Machine Learning, and building practical AI applications.

This project demonstrates my learning of GPT-2 fine-tuning, Transformer-based language models, and prompt-driven text generation.

