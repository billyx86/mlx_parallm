"""MLX ParaLLM - Fast parallel inference for MLX models."""

import importlib

__version__ = "0.2.0"
__author__ = "Billy King"

__all__ = [
    "BaseModelArgs",
    "BatchedKVCache",
    "batch_generate",
    "generate",
    "load",
    "stream_generate",
]

# Public symbols are resolved lazily (PEP 562) so that importing the
# package does not require the MLX runtime. MLX only ships real wheels for
# Apple Silicon, so without this the package -- and the CLI's `--help` and
# pure-Python helpers -- are unusable on any other platform, and even a
# typo in the CLI would be reported as "Failed to import mlx_parallm".
_LAZY_IMPORTS = {
    "load": ".utils",
    "batch_generate": ".utils",
    "generate": ".utils",
    "stream_generate": ".utils",
    "BatchedKVCache": ".models.base",
    "BaseModelArgs": ".models.base",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache: subsequent accesses skip the import
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
