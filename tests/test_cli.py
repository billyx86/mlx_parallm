"""Tests for CLI batch-mode prompt-file handling (issue #5).

These exercise the pure-Python ``read_prompts`` helper and the CLI argument
surface. They import only ``mlx_parallm.cli`` (whose MLX-backed imports are
deferred into ``main()``), so the suite runs on any platform -- including the
Linux dev box where MLX is not installable -- and in CI on macOS.

Acceptance criteria covered:
* detect encoding or allow --encoding flag   -> ``--encoding`` + tests below
* strip BOM if present                       -> utf-8-sig default + test
* handle empty lines gracefully              -> blank/whitespace-line tests
* add tests for CLI batch mode               -> this file
"""

import sys

import pytest

from mlx_parallm.cli import main, read_prompts

# --- read_prompts: plain and BOM files ------------------------------------


def test_read_prompts_plain_utf8(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("First prompt\nSecond prompt\n", encoding="utf-8")
    assert read_prompts(f) == ["First prompt", "Second prompt"]


def test_read_prompts_strips_utf8_bom(tmp_path):
    # utf-8-sig strips a leading BOM transparently.
    f = tmp_path / "bom.txt"
    f.write_bytes("\ufeffHello\nWorld\n".encode("utf-8"))
    assert read_prompts(f) == ["Hello", "World"]


def test_read_prompts_explicit_utf8_also_strips_bom(tmp_path):
    # Passing "utf-8-sig" explicitly behaves identically to the default.
    f = tmp_path / "bom2.txt"
    f.write_bytes("\ufeffA\nB\n".encode("utf-8"))
    assert read_prompts(f, encoding="utf-8-sig") == ["A", "B"]


# --- read_prompts: other encodings via the --encoding value ---------------


def test_read_prompts_latin1(tmp_path):
    f = tmp_path / "latin1.txt"
    f.write_bytes("caf\xe9 latte\nr\xe9sum\xe9\n".encode("latin-1"))
    assert read_prompts(f, encoding="latin-1") == ["café latte", "résumé"]


def test_read_prompts_utf16(tmp_path):
    f = tmp_path / "utf16.txt"
    f.write_text("one\ntwo\n", encoding="utf-16")
    assert read_prompts(f, encoding="utf-16") == ["one", "two"]


def test_read_prompts_wrong_encoding_raises_value_error(tmp_path):
    # latin-1 bytes that are invalid UTF-8 must surface a ValueError with a
    # helpful hint, not crash with a raw UnicodeDecodeError.
    f = tmp_path / "bad.txt"
    f.write_bytes(b"\xff\xfe\xff caf\xe9\n")
    with pytest.raises(ValueError) as excinfo:
        read_prompts(f, encoding="utf-8")
    assert "--encoding" in str(excinfo.value)


# --- read_prompts: empty / whitespace-only lines --------------------------


def test_read_prompts_skips_blank_lines(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("a\n\n   \n\t\nb\n", encoding="utf-8")
    assert read_prompts(f) == ["a", "b"]


def test_read_prompts_strips_surrounding_whitespace(tmp_path):
    f = tmp_path / "pad.txt"
    f.write_text("  leading  \ntrailing   \n", encoding="utf-8")
    assert read_prompts(f) == ["leading", "trailing"]


def test_read_prompts_empty_file_returns_empty_list(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert read_prompts(f) == []


def test_read_prompts_only_blank_lines_returns_empty_list(tmp_path):
    f = tmp_path / "onlyblank.txt"
    f.write_text("\n   \n\t\n", encoding="utf-8")
    assert read_prompts(f) == []


def test_read_prompts_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        read_prompts(tmp_path / "does-not-exist.txt")


# --- CLI argument surface (no model load; works without the MLX runtime) --


def test_cli_import_does_not_import_mlx():
    # The core of the platform-portability fix: importing the CLI (and the
    # package) must not pull in mlx.core, so --help and prompt-file handling
    # work on non-Apple-Silicon machines and the pure-Python helpers are
    # testable anywhere.
    #
    # Run in a fresh interpreter (subprocess) so the assertion is independent
    # of test ordering: sibling test modules (e.g. test_batched_kv_cache.py,
    # which runs earlier alphabetically) import mlx.core on platforms where
    # MLX is available and would otherwise pollute sys.modules in-process.
    #
    # This proves portability on both platforms:
    #   * Linux dev box -- mlx.core cannot even be imported (no libmlx.so),
    #     so if the CLI pulled it in transitively the import would crash.
    #   * macOS CI      -- mlx imports fine, so the assertion that it is NOT
    #     in sys.modules after importing the CLI is a real check.
    import os
    import subprocess

    code = (
        "import sys\n"
        "import mlx_parallm\n"
        "import mlx_parallm.cli\n"
        "assert 'mlx.core' not in sys.modules, 'CLI pulled in mlx.core'\n"
        "assert 'mlx_parallm.utils' not in sys.modules, 'CLI pulled in utils'\n"
        "print('IMPORTS_CLEAN')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert r.returncode == 0, f"import failed\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "IMPORTS_CLEAN" in r.stdout


def test_cli_main_no_prompt_or_batch_errors():
    # main() with neither --prompt nor --batch must exit via argparse error.
    with pytest.raises(SystemExit) as excinfo:
        main(["--model", "some/model"])
    assert excinfo.value.code != 0
