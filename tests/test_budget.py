"""Tests for the adaptive per-task token budget manager."""

import json

from slm_bench.backends.openai_backend import OpenAICompatibleBackend
from slm_bench.execution.budget import BudgetManager


def _manager(tmp_path, **kwargs):
    params = dict(
        path=str(tmp_path / "budgets.json"),
        model_id="model",
        quantization="GGUF_F16",
        backend="openai",
        thinking="off",
        min_start=1,
    )
    params.update(kwargs)
    return BudgetManager(**params)


def test_default_start_is_the_base(tmp_path):
    manager = _manager(tmp_path)
    assert manager.start_for("gsm8k", base=512, cap=4096) == 512


def test_learns_a_higher_start(tmp_path):
    manager = _manager(tmp_path)
    manager.record("gsm8k", base=512, start=512, cap=4096, final_tokens=500, truncated=False)
    assert manager.start_for("gsm8k", base=512, cap=4096) == 512


def test_decays_when_far_below_the_start(tmp_path):
    manager = _manager(tmp_path)
    manager.record("gsm8k", base=1, start=8, cap=4096, final_tokens=2, truncated=False)
    assert manager.start_for("gsm8k", base=1, cap=4096) == 4


def test_stays_when_close_to_the_start(tmp_path):
    manager = _manager(tmp_path)
    manager.record("gsm8k", base=1, start=8, cap=4096, final_tokens=8, truncated=False)
    assert manager.start_for("gsm8k", base=1, cap=4096) == 8


def test_truncation_keeps_the_current_cap(tmp_path):
    manager = _manager(tmp_path)
    manager.record("gsm8k", base=512, start=2048, cap=4096, final_tokens=2048, truncated=True, final_budget=4096)
    assert manager.start_for("gsm8k", base=512, cap=4096) == 4096


def test_cap_is_respected(tmp_path):
    manager = _manager(tmp_path)
    manager.record("gsm8k", base=512, start=4096, cap=4096, final_tokens=4000, truncated=False)
    assert manager.start_for("gsm8k", base=512, cap=4096) == 4096


def test_state_persists_across_instances(tmp_path):
    first = _manager(tmp_path)
    first.record("gsm8k", base=512, start=512, cap=4096, final_tokens=2000, truncated=False)

    second = _manager(tmp_path)
    assert second.start_for("gsm8k", base=512, cap=4096) == 2048


def test_state_marks_cap_exceeded(tmp_path):
    manager = _manager(tmp_path, min_start=1)
    manager.record(
        "gsm8k",
        base=128,
        start=1024,
        cap=1024,
        final_tokens=1024,
        truncated=True,
        final_budget=1024,
    )

    state = json.loads((tmp_path / "budgets.json").read_text())
    entry = state["model|GGUF_F16|openai|off|gsm8k"]
    assert entry["cap_exceeded"] is True
    assert entry["required_cap"] is None


def test_default_minimum_start_is_128(tmp_path):
    manager = _manager(tmp_path, min_start=128)
    manager.record("gsm8k", base=256, start=256, cap=4096, final_tokens=40, truncated=False)
    assert manager.start_for("gsm8k", base=256, cap=4096) == 128


def test_state_is_scoped_by_task_and_thinking(tmp_path):
    manager = _manager(tmp_path)
    manager.record("gsm8k", base=512, start=512, cap=4096, final_tokens=2000, truncated=False)

    assert manager.start_for("ifeval", base=512, cap=4096) == 512
    other_thinking = _manager(tmp_path, thinking="auto")
    assert other_thinking.start_for("gsm8k", base=512, cap=4096) == 512


def test_backend_starts_from_the_learned_budget(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager.record("fake", base=1, start=8, cap=4096, final_tokens=2, truncated=False)

    captured = []

    def fake_post(path, payload):
        captured.append(payload["max_tokens"])
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5},
        }

    backend = OpenAICompatibleBackend("model", budget_manager=manager)
    backend.current_task_id = "fake"
    monkeypatch.setattr(backend, "_post", fake_post)

    backend.chat([{"role": "user", "content": "q"}], max_new_tokens=1)

    assert captured == [4]
