# MLX ParaLLM

Batched KV caching for fast parallel inference on Apple Silicon devices, via [MLX](https://github.com/ml-explore/mlx).

This repo heavily borrows from [`mlx_lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). It adds batched generation with a shared KV cache, yielding ~2-5× throughput for multiple prompts on Apple Silicon.

> **Status**: Revived 2026-08-10 with improved docs, tests, CLI, streaming support, and bug fixes.

## Quick start

```bash
pip install mlx-parallm
```

```python
from mlx_parallm import load, batch_generate

model, tokenizer = load("google/gemma-1.1-2b-it")
prompts = ["Explain quantum computing.", "Write a haiku about rain."]
responses = list(batch_generate(model, tokenizer, prompts, max_tokens=128, verbose=True))
for r in responses[-1]:
    print(r)
```

## CLI

```bash
# Single prompt
python -m mlx_parallm.cli --model google/gemma-1.1-2b-it --prompt "Hello world"

# Batch: one prompt per line; blank lines are ignored
python -m mlx_parallm.cli --model google/gemma-1.1-2b-it --batch prompts.txt --output out.json
```

Prompt files are read as UTF-8 by default; a UTF-8 BOM is stripped when
present. For files in another encoding, pass `--encoding` (e.g.
`--encoding latin-1` or `--encoding utf-16`).

## Features

Supported:
- `batch_generate` with streaming yields per step
- Per-token streaming callback (`token_callback`) for real-time UI updates
- Auto-padding with left padding
- Auto-formatting with prompt templates (`format_prompts=True`)
- Temperature sampling and nucleus sampling (`top_p`)
- Repetition penalty with context window
- Single-stream `generate` and `stream_generate`
- LoRA adapter loading
- CLI for interactive and batch use
- Proper package layout with `pyproject.toml`

Not yet supported:
- Dynamic batching for async requests
- Repetition penalties in single-stream mode (works in batch)

## Models

Tested models:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-1.1-2b-it`
- Quantized 4-bit variants via `mlx-community/*`

Float16 models perform fastest when RAM allows (~1300+ tok/s for gemma-2b on M3 Max 128GB).

Add new architectures by copying from `mlx_lm/models` and replacing `KVCache` with `BatchedKVCache`.

## Memory & performance

MLX shares the system's unified memory, so inference cost is dominated by
**model weights** plus the **KV cache**. The KV cache is the part that
scales with your workload:

```
kv_bytes ≈ 2 · batch_size · n_kv_heads · seq_len · head_dim · dtype_bytes
```

(`2` = keys + values; `dtype_bytes` is 2 for float16.) It grows linearly in
both `batch_size` and `seq_len`, so a large batch of long prompts is the
main memory risk. `BatchedKVCache` allocates the sequence dimension in
256-token chunks and frees it with `.reset()`, so reuse one cache across
runs rather than holding many.

Full guidance — weight/KV-cache math with worked examples, per-chip batch
size tables (M1 through Ultra-class), and profiling recipes — is in
[MEMORY.md](MEMORY.md).

## API reference

### `load(path_or_hf_repo, ...) -> (model, tokenizer)`
Load model and tokenizer from local path or Hugging Face Hub.

### `batch_generate(model, tokenizer, prompts, max_tokens=100, verbose=False, format_prompts=True, token_callback=None, **kwargs)`
Yields lists of partial strings per batch step. Use `list(...)` to collect final results.

- **`token_callback(token: str, prompt_index: int) -> None`** (optional): called once per newly-decoded token/segment for each prompt, in generation order. Enables real-time streaming into a UI (e.g. append `token` to the line for `prompt_index`). Defaults to `None` — original behavior with no callbacks.

```python
def on_token(token, prompt_index):
    print(f"[prompt {prompt_index}] {token}", end="", flush=True)

for step in batch_generate(model, tokenizer, prompts, max_tokens=128, token_callback=on_token):
    pass  # callback already fired per token
```

### `generate(model, tokenizer, prompt, max_tokens=100, **kwargs) -> str`
Single prompt generation.

### `stream_generate(model, tokenizer, prompt, max_tokens=100, **kwargs) -> Generator[str]`
Yield tokens as they are generated.

## Development

```bash
git clone https://github.com/billyx86/mlx_parallm.git
cd mlx_parallm
pip install -e ".[dev]"
pytest tests/
```

## License

MIT

## Acknowledgements

Built on [mlx-lm](https://github.com/ml-explore/mlx-lm) and [MLX](https://github.com/ml-explore/mlx).
