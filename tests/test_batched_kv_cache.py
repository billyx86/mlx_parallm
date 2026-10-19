"""Comprehensive edge-case tests for BatchedKVCache (issue #2).

These cover the acceptance criteria from the issue:

* ``reset()`` clearing state (and the cache being reusable afterwards)
* the ``step`` size parameter controlling allocation granularity
* memory growth over a long, multi-reallocation sequence
* batch-size handling (a 2-batch cache stores each batch independently)
* error handling on invalid input (a head_dim mismatch that cannot
  broadcast must be rejected)

All shapes are deliberately tiny (batch<=2, heads<=1, dim<=4, step<=4,
seq<=10) so the suite runs on any MLX device without meaningful memory
pressure. The expected values are hand-derived from the implementation in
``mlx_parallm/models/base.py``, so the assertions are deterministic.
"""

import pytest


def _make_kv(batch, heads, seq, dim, value):
    """Build keys and values tensors filled with a uniform ``value``."""
    import mlx.core as mx

    k = mx.full((batch, heads, seq, dim), value, dtype=mx.float32)
    v = mx.full((batch, heads, seq, dim), value, dtype=mx.float32)
    return k, v


def _col(k, batch, head, slot):
    """Scalar-per-sequence-position view used for value placement checks."""
    return k[batch, head, :, slot].tolist()


def test_reset_clears_state_and_cache_is_reusable():
    """reset() clears keys/values/offset and the cache works again after."""
    import mlx.core as mx
    from mlx_parallm.models.base import BatchedKVCache

    cache = BatchedKVCache(head_dim=4, n_kv_heads=1, batch_size=1, step=4)
    # Two updates accumulate to a total sequence length of 4.
    cache.update_and_fetch(*_make_kv(1, 1, 3, 4, value=1))
    cache.update_and_fetch(*_make_kv(1, 1, 1, 4, value=2))
    assert cache.offset == 4
    assert cache.keys is not None
    assert cache.values is not None

    # reset() must clear all accumulated state.
    cache.reset()
    assert cache.keys is None
    assert cache.values is None
    assert cache.offset == 0

    # The cache must be reusable: a fresh update after reset is treated as
    # a brand-new cache (no stale data from before the reset).
    rk, rv = cache.update_and_fetch(*_make_kv(1, 1, 2, 4, value=7))
    assert cache.offset == 2
    assert rk.shape == (1, 1, 2, 4)
    assert _col(rk, 0, 0, 0) == [7, 7]


def test_step_size_controls_allocation_granularity():
    """A smaller ``step`` yields a smaller (step-multiple) internal buffer,
    while the returned data is still trimmed to the real sequence length."""
    import mlx.core as mx
    from mlx_parallm.models.base import BatchedKVCache

    cache = BatchedKVCache(head_dim=2, n_kv_heads=1, batch_size=1, step=2)
    rk, rv = cache.update_and_fetch(*_make_kv(1, 1, 3, 2, value=5))

    # Returned tensors are always trimmed to the true sequence length...
    assert rk.shape == (1, 1, 3, 2)
    assert _col(rk, 0, 0, 0) == [5, 5, 5]
    assert cache.offset == 3

    # ...while the internal buffer is allocated in multiples of ``step``.
    assert cache.keys.shape[2] >= 3
    assert cache.keys.shape[2] % 2 == 0


def test_memory_growth_over_long_sequence():
    """Repeated updates across multiple reallocations grow the cache; the
    returned tensor equals the concatenation of every input in order, and
    each position holds the value of the update that wrote it."""
    import mlx.core as mx
    from mlx_parallm.models.base import BatchedKVCache

    cache = BatchedKVCache(head_dim=2, n_kv_heads=1, batch_size=1, step=2)
    rk = rv = None
    n_updates = 5
    for i in range(n_updates):
        # Each update appends 2 tokens, all filled with the update index.
        rk, rv = cache.update_and_fetch(*_make_kv(1, 1, 2, 2, value=i))

    # Total sequence length is 5 updates * 2 tokens.
    assert cache.offset == 10
    assert rk.shape == (1, 1, 10, 2)

    # Position j was written by update floor(j/2), so the per-position
    # values read [0,0,1,1,2,2,3,3,4,4].
    assert _col(rk, 0, 0, 0) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    # The backing buffer grew to at least the total sequence length.
    assert cache.keys.shape[2] >= 10


def test_batch_size_two_stored_independently():
    """A batch_size=2 cache stores each batch lane independently."""
    import mlx.core as mx
    from mlx_parallm.models.base import BatchedKVCache

    cache = BatchedKVCache(head_dim=2, n_kv_heads=1, batch_size=2, step=4)
    # batch 0 -> all 1s, batch 1 -> all 9s (built explicitly to avoid any
    # slice-assignment dependency).
    k = mx.array([[[[1.0, 1.0], [1.0, 1.0]]], [[[9.0, 9.0], [9.0, 9.0]]]])
    v = mx.array([[[[1.0, 1.0], [1.0, 1.0]]], [[[9.0, 9.0], [9.0, 9.0]]]])
    rk, rv = cache.update_and_fetch(k, v)

    assert rk.shape == (2, 1, 2, 2)
    assert _col(rk, 0, 0, 0) == [1, 1]
    assert _col(rk, 1, 0, 0) == [9, 9]
    assert cache.offset == 2


def test_empty_input_is_handled_gracefully():
    """A zero-length update does not crash; offset stays 0 and the cache
    remains usable for a subsequent real update."""
    import mlx.core as mx
    from mlx_parallm.models.base import BatchedKVCache

    cache = BatchedKVCache(head_dim=4, n_kv_heads=1, batch_size=1, step=4)
    k = mx.zeros((1, 1, 0, 4), dtype=mx.float32)
    v = mx.zeros((1, 1, 0, 4), dtype=mx.float32)
    rk, rv = cache.update_and_fetch(k, v)

    assert cache.offset == 0
    assert rk.shape == (1, 1, 0, 4)
    assert cache.keys is not None  # buffer allocated, nothing written

    # A subsequent real update still lands at the correct position.
    rk2, _ = cache.update_and_fetch(*_make_kv(1, 1, 2, 4, value=3))
    assert cache.offset == 2
    assert _col(rk2, 0, 0, 0) == [3, 3]


def test_invalid_head_dim_mismatch_raises():
    """Input whose last (head_dim) dimension is incompatible with the
    cache's head_dim cannot be broadcast into the buffer and must raise."""
    import mlx.core as mx
    from mlx_parallm.models.base import BatchedKVCache

    cache = BatchedKVCache(head_dim=4, n_kv_heads=1, batch_size=1, step=4)
    # Last dim is 2, but the cache allocates head_dim=4 slots: 2 and 4 are
    # not broadcast-compatible, so the store is a genuine shape error.
    k = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
    v = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
    with pytest.raises(Exception):
        cache.update_and_fetch(k, v)
