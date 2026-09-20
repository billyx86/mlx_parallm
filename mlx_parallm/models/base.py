import inspect
from dataclasses import dataclass

import mlx.core as mx

def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9

class BatchedKVCache:

    def __init__(self, head_dim, n_kv_heads, batch_size=1, step=256):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

    def reset(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        seq_len = keys.shape[2]

        if self.keys is None or (prev + seq_len) > self.keys.shape[2]:
            # Allocate in chunks to avoid excessive memory growth
            n_steps = max(1, (seq_len + self.step - 1) // self.step)
            # Ensure we have at least enough for current + one step
            total_needed = max(prev + seq_len, self.step)
            n_steps = max(n_steps, (total_needed + self.step - 1) // self.step)
            shape = (self.batch_size, self.n_kv_heads, n_steps * self.step, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                # Trim to current offset to avoid waste
                if prev > 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += seq_len
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )