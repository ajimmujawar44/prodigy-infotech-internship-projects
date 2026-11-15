"""Generate text using a pretrained GPT-2 model or a fine-tuned checkpoint."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="outputs/gpt2-custom")
    parser.add_argument("--prompt", required=True, help="Text prompt to continue")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--no_sample", action="store_true", help="Use greedy decoding")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.no_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not args.no_sample:
        generation_kwargs.update(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    print("\n--- Generated text ---\n")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print("\n--- Generation settings ---")
    print(f"Model: {args.model_path} | Device: {device}")
    print(
        f"New-token limit: {args.max_new_tokens} | Sampling: {not args.no_sample} | "
        f"Temperature: {args.temperature} | top_k: {args.top_k} | top_p: {args.top_p}"
    )


if __name__ == "__main__":
    main()
