"""
CLI for mlx_parallm.

Usage:
  python -m mlx_parallm.cli --model google/gemma-1.1-2b-it --prompt "Hello"
  python -m mlx_parallm.cli --model google/gemma-1.1-2b-it --batch prompts.txt --output out.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from mlx_parallm import load, batch_generate, generate
except Exception as e:
    print(f"Failed to import mlx_parallm: {e}", file=sys.stderr)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="MLX ParaLLM CLI")
    p.add_argument("--model", required=True, help="Hugging Face repo or local path")
    p.add_argument("--prompt", help="Single prompt string")
    p.add_argument("--batch", help="File with one prompt per line")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--output", help="Write JSON output to file")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not args.prompt and not args.batch:
        p.error("Provide --prompt or --batch")

    print(f"Loading model {args.model} ...")
    model, tokenizer = load(args.model)

    if args.prompt:
        out = generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            temp=args.temperature,
            top_p=args.top_p,
        )
        print(out)
        if args.output:
            Path(args.output).write_text(json.dumps({"prompt": args.prompt, "response": out}))
        return

    prompts = []
    batch_path = Path(args.batch)
    if batch_path.is_file():
        prompts = [line.strip() for line in batch_path.read_text().splitlines() if line.strip()]
    else:
        prompts = [args.batch]

    print(f"Generating for {len(prompts)} prompts ...")
    results = []
    # Collect final responses
    partials = ["" for _ in prompts]
    for step_segments in batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        temp=args.temperature,
        top_p=args.top_p,
    ):
        for i, seg in enumerate(step_segments):
            if seg:
                partials[i] += seg

    for prompt, response in zip(prompts, partials):
        results.append({"prompt": prompt, "response": response})
        print(f"\n---\nPrompt: {prompt}\nResponse: {response}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
