# Memory usage and performance tuning

MLX models run in the same unified memory pool as the rest of macOS. The two
big consumers during inference are **model weights** and the **KV cache**.
This document explains both, gives batch-size recommendations per chip, and
shows how to profile your own setup.

## 1. Model weight memory

Approximate resident weight memory is simply parameter count × bytes per
weight:

| Precision        | Bytes/weight | Example: 7B model | Example: 31B model |
|------------------|--------------|-------------------|--------------------|
| float16 (no quant) | 2         | ≈ 14 GB           | ≈ 62 GB            |
| 8-bit            | 1            | ≈ 7 GB            | ≈ 31 GB            |
| 4-bit            | 0.5          | ≈ 3.5 GB          | ≈ 15.5 GB          |

Add a few hundred MB for the tokenizer, buffers, and MLX overhead. As a rule
of thumb, **your chip's usable GPU memory must exceed the model weights with
room left over for the KV cache** — loading a 14 GB model on a 16 GB chip
leaves almost nothing for the cache and will either fail or thrash.

macOS exposes only a fraction of unified memory to the GPU (a system
default around 75% on recent OSes). Treat the "GPU" budget as
`unified_memory × ~0.75` when sizing, and lower the fraction for anything
beyond rough planning.

## 2. KV cache memory (and how this library allocates it)

For a decoder model the KV cache stores one keys and one values tensor per
token generated so far. Each has shape

```
(batch_size, n_kv_heads, seq_len, head_dim)
```

so the total KV cache size in bytes is

```
kv_bytes = 2 · batch_size · n_kv_heads · seq_len · head_dim · dtype_bytes
```

where `2` accounts for keys **and** values, and `dtype_bytes` is 2 for
float16 (the usual case) or 1 for float8/e4m3.

### Worked example

A 7B-class model with `n_kv_heads = 8`, `head_dim = 128`, float16,
`batch_size = 1`:

```
kv_bytes = 2 · 1 · 8 · seq_len · 128 · 2  =  4096 · seq_len  ≈ 4 KB/token
```

- 512-token context  → ~2 MB
- 2,048-token context → ~8 MB
- 32,768-token context → ~128 MB

KV cache cost grows **linearly in sequence length and batch size** — a long
batch of long prompts is the main memory risk in this library.

### How `BatchedKVCache` allocates

`mlx_parallm.models.base.BatchedKVCache` allocates the sequence dimension in
chunks of `step` tokens (default **256**):

* the buffer is always a multiple of `step`, so at most `step − 1` tokens of
  capacity are unused at the tail;
* when a growth is needed the *existing* buffer is first trimmed to the real
  `offset` before the new chunk is concatenated, so freed space is not
  carried forward;
* growth is per-batch (the whole `(batch, heads, seq, dim)` block), so
  `batch_size = N` multiplies the whole allocation by N;
* `reset()` frees both tensors (`keys = values = None`), so you can reuse one
  cache object across many runs instead of holding N caches alive.

Practical consequences:

* The **largest** KV allocation you will see is roughly
  `batch_size · step` tokens per growth event, plus the running total.
* If you process many short prompts in one batch, keep `batch_size` modest
  and let `step` stay at its default; if you process few very long sequences,
  memory is dominated by `seq_len`, not `step`.

## 3. Batch size recommendations

Guidance is per-chip **usable** memory (unified memory × ~0.75), assuming
float16 weights and a few KB of KV per token (7B-class model).

| Chip (examples)            | Usable ≈ | 4-bit 3B model  | 4-bit 7B model   | 4-bit 13–14B model | float16 7B model |
|----------------------------|----------|-----------------|------------------|--------------------|------------------|
| M1/M2 8 GB                 | ~6 GB    | batch 2–4, ctx 2k | won't fit        | won't fit          | won't fit        |
| M1/M2/M3 16 GB             | ~12 GB   | batch 4–8, ctx 4k | batch 1–2, ctx 4k| batch 1, ctx 2k    | batch 1, ctx 2k  |
| M1/M2 32 GB                | ~24 GB   | batch 8+, ctx 8k | batch 2–4, ctx 8k| batch 1–2, ctx 4k  | batch 1–2, ctx 4k|
| M3/M4 24–36 GB             | ~18–27 GB| generous        | batch 2–4, ctx 8k| batch 1–2, ctx 8k  | batch 2, ctx 8k  |
| M2 Max/M3 Max 64 GB        | ~48 GB   | generous        | batch 4–8, ctx 16k | batch 2–4, ctx 8k | batch 2–4, ctx 8k|
| Ultra-class 128 GB+        | ~96 GB+  | —               | batch 8–16, ctx 16k+ | generous      | batch 4–8, ctx 16k+|

Rules of thumb when in doubt:

1. **Start at `batch_size = 1`** and increase until memory usage plateaus
   comfortably below the GPU budget, or you see the process get killed.
2. **KV cache grows with `batch_size × seq_len`** — halve one if you double
   the other.
3. **Quantize first, batch second**: moving from float16 to 4-bit frees more
   memory than any batching change.
4. Process independent jobs in separate `batch_generate` calls and call
   `cache.reset()` (or let the cache go out of scope) between them rather
   than accumulating one giant batch.

## 4. Profiling your own setup

### a) Watch actual GPU memory

While a run is in flight, keep an eye on *Activity Monitor → Memory → Memory
Pressure* (the "wired memory" / GPU row), or poll it from the shell:

```bash
# macOS: wired (non-pageable, mostly GPU) memory in MB
while true; do
  vm_stat | awk '/page size of/{p=$5} /Pages wired down/{w=$3} END{printf "wired: %.1f MB\n", w*p/1024/1024}'
  sleep 2
done
```

Run your `batch_generate` in a terminal next to it and note the peak.

### b) Measure tokens/second end to end

```python
import time
from mlx_parallm import load, batch_generate

model, tokenizer = load("mlx-community/gemma-2b-4bit")
prompts = ["Explain KV caching in one paragraph."] * 4  # tune this

t0 = time.perf_counter()
n = 0
for step in batch_generate(model, tokenizer, prompts, max_tokens=128, verbose=True):
    n += sum(1 for seg in step if seg)  # tokens produced this step
dt = time.perf_counter() - t0
print(f"{n} tokens in {dt:.1f}s -> {n / dt:.1f} tok/s total")
```

Sweep `len(prompts)` (1, 2, 4, 8, …) with `max_tokens` fixed and plot
throughput vs. batch size — throughput rises while compute is the
bottleneck, then flattens (or OOMs) when memory is.

### c) Isolate the KV cache cost

You can size the cache directly to confirm the §2 formula:

```python
import mlx.core as mx

batch, heads, dim, seq = 2, 8, 128, 4096
# Allocate as float16 to match the usual runtime dtype (mx.zeros defaults to float32).
k = mx.zeros((batch, heads, seq, dim), mx.float16)
print(f"one K tensor: {k.nbytes / 1024 / 1024:.1f} MiB")
# total KV cache ≈ 2 × that (keys + values), before step-rounding
```

Compare against the §2 formula: `2 · batch · heads · seq · dim · dtype_bytes`
(here `dtype_bytes = 2` for float16).
