CHANGELOG
=========

0.2.0 - 2026-08-10
-----------------
Revived and modernised.

Features:
- Added pyproject.toml and proper packaging
- CLI tool `python -m mlx_parallm.cli`
- Tests in tests/test_core.py
- Improved README with quick start and API docs
- BatchedKVCache reset() method and improved allocation
- Safer top-p sampling with temperature <=0 handling
- Safer repetition penalty with vocab bounds check
- Demo v2 with better streaming display
- __init__ exports public API
- .gitignore updated

Bugfixes:
- Fixed BatchedKVCache memory growth logic
- Fixed top_p_sampling division by zero
- Fixed apply_repetition_penalty index out of bounds
- Improved error messages for model loading

0.1.0 - 2025-04-06
-----------------
Initial release with batch_generate and BatchedKVCache.
