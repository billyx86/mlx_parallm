"""
CLI for mlx_parallm.

Usage:
  python -m mlx_parallm.cli --model google/gemma-1.1-2b-it --prompt "Hello"
  python -m mlx_parallm.cli --model google/gemma-1.1-2b-it --batch prompts.txt --output out.json

Prompt files are read as UTF-8 by default; a UTF-8 BOM is stripped when
present. Use --encoding (e.g. "latin-1", "utf-16") for other encodings.
Blank lines are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_prompts(batch_path: Path | str, encoding: str = "utf-8-sig") -> list[str]:
    """Read one prompt per line from ``batch_path``.

    The default ``utf-8-sig`` encoding strips a UTF-8 BOM when present and
    reads plain UTF-8 files otherwise. Pass a different ``encoding`` (for
    example ``"latin-1"`` or ``"utf-16"``) for files in other encodings.

    Lines that are empty or contain only whitespace are skipped.

    Raises:
        ValueError: if the file cannot be read or cannot be decoded with the
            requested encoding (with a hint about --encoding).
    """
    batch_path = Path(batch_path)
    try:
        raw = batch_path.read_text(encoding=encoding)
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Could not decode '{batch_path}' as {encoding} "
            f"(at byte {e.start}: {e.reason}). "
            f"The file may use a different encoding -- try --encoding latin-1, "
            f"utf-16, or cp1252."
        ) from e
    except OSError as e:
        raise ValueError(f"Could not read '{batch_path}': {e}") from e

    return [line.strip() for line in raw.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MLX ParaLLM CLI")
    p.add_argument("--model", required=True, help="Hugging Face repo or local path")
    p.add_argument("--prompt", help="Single prompt string")
    p.add_argument("--batch", help="File with one prompt per line (or a single prompt string)")
    p.add_argument(
        "--encoding",
        default="utf-8-sig",
        help=(
            "Encoding of the --batch prompt file. Default 'utf-8-sig' reads "
            "UTF-8 and strips a BOM when present; use e.g. 'latin-1' or "
            "'utf-16' for other encodings. (default: %(default)s)"
        ),
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--output", help="Write JSON output to file")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if not args.prompt and not args.batch:
        p.error("Provide --prompt or --batch")

    # Import the MLX-backed API only once it is actually needed, so that
    # --help and prompt-file handling work without the MLX runtime
    # (MLX ships real wheels for Apple Silicon only). Any import-time failure
    # (missing libmlx, mlx, or a dependency) is the same user-facing error.
    try:
        from mlx_parallm import batch_generate, generate, load
    except Exception as e:  # noqa: BLE001 - see note above
        print(f"Failed to import mlx_parallm: {e}", file=sys.stderr)
        return 1

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
        return 0

    batch_path = Path(args.batch)
    if batch_path.is_file():
        try:
            prompts = read_prompts(batch_path, args.encoding)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not prompts:
            print(
                f"error: no prompts found in '{batch_path}' "
                f"(file is empty or contains only blank lines)",
                file=sys.stderr,
            )
            return 1
    else:
        prompts = [args.batch]

    print(f"Generating for {len(prompts)} prompts ...")
    results = []
    # Collect final responses
    partials = [""] * len(prompts)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
