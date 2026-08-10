import sys
import pytest

def test_import():
    import mlx_parallm
    assert hasattr(mlx_parallm, "load")
    assert hasattr(mlx_parallm, "batch_generate")
    assert hasattr(mlx_parallm, "generate")
    assert hasattr(mlx_parallm, "stream_generate")

def test_batched_kv_cache():
    from mlx_parallm.models.base import BatchedKVCache
    import mlx.core as mx
    cache = BatchedKVCache(head_dim=64, n_kv_heads=8, batch_size=2)
    keys = mx.ones((2, 8, 4, 64))
    values = mx.ones((2, 8, 4, 64))
    k, v = cache.update_and_fetch(keys, values)
    assert k.shape == (2, 8, 4, 64)
    assert v.shape == (2, 8, 4, 64)
    assert cache.offset == 4

def test_top_p_sampling():
    from mlx_parallm.sample_utils import top_p_sampling
    import mlx.core as mx
    logits = mx.array([[1.0, 2.0, 3.0]])
    tokens = top_p_sampling(logits, top_p=0.9, temperature=1.0)
    assert tokens.shape == (1, 1)

def test_apply_repetition_penalty():
    from mlx_parallm.utils import apply_repetition_penalty
    import mlx.core as mx
    logits = mx.array([[0.1, 0.5, 0.2]])
    generated = mx.array([1, 2, 1])
    out = apply_repetition_penalty(logits, generated, penalty=1.1)
    assert out.shape == logits.shape

def test_model_args_from_dict():
    from mlx_parallm.models.llama import ModelArgs
    cfg = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32000,
    }
    args = ModelArgs.from_dict(cfg)
    assert args.hidden_size == 4096
    assert args.num_key_value_heads == 32

if __name__ == "__main__":
    pytest.main([sys.argv[0]])
