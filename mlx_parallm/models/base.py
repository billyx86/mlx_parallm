import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import mlx.core as mx

_T = TypeVar("_T", bound="BaseModelArgs")


def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class BatchedKVCache:
    """Per-layer key/value cache for batched generation.

    The cache grows in ``step``-sized chunks along the sequence axis and
    serves a view of everything written so far from
    :meth:`update_and_fetch`. ``batch_size`` slots are stored side by side
    on the batch axis, so one cache instance serves the whole batch.
    """

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        batch_size: int = 1,
        step: int = 256,
    ) -> None:
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self.step: int = step

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
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
            if self.keys is not None and self.values is not None:
                # Trim to current offset to avoid waste
                if prev > 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += seq_len
        assert self.keys is not None and self.values is not None
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls: Type[_T], params: Dict[str, Any]) -> _T:
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
