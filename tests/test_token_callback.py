"""Tests for the per-token ``token_callback`` on ``batch_generate`` (issue #3).

These run in CI (macOS, where MLX is importable) but do **not** download a
model: ``generate_step`` is monkeypatched to yield deterministic fake token
arrays and a fake tokenizer is used, so the test isolates and exercises the
callback-invocation logic in ``batch_generate`` itself.

Acceptance criteria covered:
* optional token_callback parameter added to batch_generate
* callback receives (token, prompt_index)
* backward compatibility (no callback -> unchanged yields, no crash)
"""

import mlx.core as mx

import mlx_parallm.utils as U

# --- Fake tokenizer ---------------------------------------------------------

class FakeUnderlyingTokenizer:
    padding_side = "right"

    def __call__(self, prompts, padding="longest", return_tensors="np", truncation=False):
        import numpy as np

        n = len(prompts)
        return {"input_ids": np.zeros((n, 4), dtype=np.int32)}


class FakeTokenizer:
    def __init__(self, pad_id=2, eos_id=0):
        self._tokenizer = FakeUnderlyingTokenizer()
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def decode(self, token_ids, skip_special_tokens=True):
        # Deterministic, per-token text so segments are non-empty and
        # predictable: token 1 -> "T1 ", 3 -> "T3 ", etc.
        return "".join(f"T{t} " for t in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return ""


# --- Fake generate_step: deterministic token schedule -----------------------
# batch_size = 2, three steps. eos_token_id = 0.
#   step 1: prompt0 -> token 1, prompt1 -> token 2
#   step 2: prompt0 -> token 3, prompt1 -> token 4
#   step 3: prompt0 -> token 0 (EOS), prompt1 -> token 5

def _fake_generate_step(prompts, model, **kwargs):
    yield mx.array([[1], [2]]), mx.array([[0.5], [0.5]])
    yield mx.array([[3], [4]]), mx.array([[0.5], [0.5]])
    yield mx.array([[0], [5]]), mx.array([[0.5], [0.5]])


def _run_batch(monkeypatch, token_callback=None):
    monkeypatch.setattr(U, "generate_step", _fake_generate_step)
    tok = FakeTokenizer()
    steps = list(U.batch_generate(
        model=None,
        tokenizer=tok,
        prompts=["a", "b"],
        max_tokens=3,
        format_prompts=False,
        token_callback=token_callback,
    ))
    return steps


def test_token_callback_receives_token_and_index(monkeypatch):
    calls = []

    def cb(token, prompt_index):
        calls.append((token, prompt_index))

    _run_batch(monkeypatch, token_callback=cb)

    # Exact (token, prompt_index) sequence, in generation order.
    assert calls == [
        ("T1 ", 0),
        ("T2 ", 1),
        ("T3 ", 0),
        ("T4 ", 1),
        ("T5 ", 1),
    ]
    # Every call is a (str, int) pair.
    assert all(isinstance(t, str) and isinstance(i, int) for t, i in calls)


def test_token_callback_indices_match_prompts(monkeypatch):
    by_prompt = {0: [], 1: []}

    def cb(token, prompt_index):
        by_prompt[prompt_index].append(token)

    _run_batch(monkeypatch, token_callback=cb)

    # prompt 0's decoded text (minus the dropped EOS) is T1 T3; prompt 1 is T2 T4 T5.
    assert "".join(by_prompt[0]) == "T1 T3 "
    assert "".join(by_prompt[1]) == "T2 T4 T5 "


def test_backward_compat_no_callback(monkeypatch):
    # Without a callback the generator still yields per-step segment lists and
    # does not crash.
    steps = _run_batch(monkeypatch, token_callback=None)

    assert steps == [
        ["T1 ", "T2 "],
        ["T3 ", "T4 "],
        [None, "T5 "],
    ]


def test_backward_compat_default_is_none(monkeypatch):
    import inspect

    sig = inspect.signature(U.batch_generate)
    assert "token_callback" in sig.parameters
    assert sig.parameters["token_callback"].default is None
