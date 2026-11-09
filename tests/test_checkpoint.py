"""
Tests for replay checkpointing of inference calls.
"""

import json

from slm_bench.backends.base import BaseInferenceBackend, GenerationResult
from slm_bench.benchmarks.suite import BenchmarkSuite
from slm_bench.models.schema import HardwareInfo, ModelMetadata, QuestionResult, TaskMetric


class CountingBackend(BaseInferenceBackend):
    """Backend that counts real generation calls."""

    def __init__(self, model_id="test/model", quantization="GGUF_Q4_K_M"):
        super().__init__(model_id, quantization)
        self.calls = 0

    def load(self):
        self.is_loaded = True

    def generate(self, prompt, max_new_tokens=512, temperature=0.0, stop_sequences=None):
        self.calls += 1
        return GenerationResult(f"generated:{prompt}", 10.0, 4)

    def chat(self, messages, max_new_tokens=512, temperature=0.0, stop_sequences=None):
        self.calls += 1
        return GenerationResult(f"chat:{messages[-1]['content']}", 10.0, 4)


class FakeTask:
    task_id = "fake"
    category = "reasoning"

    def __init__(self, prompt="question one", max_new_tokens=8):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def run(self, backend, sample_limit=None, on_sample=None, cancel_event=None, seed=42):
        gen = backend.chat(
            [{"role": "user", "content": self.prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="PARTIAL",
            raw_score=1.0,
            normalized_score=100.0,
            total_samples=1,
            passed_samples=1,
            avg_latency_ms=gen.latency_ms,
            avg_tokens_per_sec=gen.tokens_per_sec,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    raw_prompt=self.prompt,
                    raw_response=gen.text,
                    is_correct=True,
                    reused=gen.reused,
                )
            ],
        )


class GenerateFakeTask(FakeTask):
    def run(self, backend, sample_limit=None, on_sample=None, cancel_event=None, seed=42):
        gen = backend.generate(self.prompt, max_new_tokens=self.max_new_tokens, temperature=0.0)
        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="PARTIAL",
            raw_score=1.0,
            normalized_score=100.0,
            total_samples=1,
            passed_samples=1,
            avg_latency_ms=gen.latency_ms,
            avg_tokens_per_sec=gen.tokens_per_sec,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    raw_prompt=self.prompt,
                    raw_response=gen.text,
                    is_correct=True,
                    reused=gen.reused,
                )
            ],
        )


class TransformedPromptTask(FakeTask):
    """Mimics RULER: records a truncated raw_prompt rather than the sent prompt."""

    def run(self, backend, sample_limit=None, on_sample=None, cancel_event=None, seed=42):
        gen = backend.chat(
            [{"role": "user", "content": self.prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="PARTIAL",
            raw_score=1.0,
            normalized_score=100.0,
            total_samples=1,
            passed_samples=1,
            avg_latency_ms=gen.latency_ms,
            avg_tokens_per_sec=gen.tokens_per_sec,
            question_results=[
                QuestionResult(
                    question_id="q1",
                    raw_prompt=self.prompt[:4] + "... [truncated]",
                    raw_response=gen.text,
                    is_correct=True,
                    reused=gen.reused,
                )
            ],
        )


def _meta(model_id="test/model", backend="mock", reasoning_effort=None):
    return ModelMetadata(
        model_id=model_id,
        model_slug="model",
        param_count_b=1.0,
        param_tier="<3B",
        quantization="GGUF_Q4_K_M",
        backend=backend,
        reasoning_effort=reasoning_effort,
    )


def _run(tmp_path, backend, task=None, model_id="test/model", meta_backend="mock", reasoning_effort=None):
    suite = BenchmarkSuite()
    suite.resolve_tasks = lambda **kwargs: {"fake": task or FakeTask()}
    hw = HardwareInfo(os="Linux", cpu="Xeon", cpu_cores=8, total_ram_gb=16.0)
    return suite.run_suite(
        backend,
        _meta(model_id, meta_backend, reasoning_effort),
        hw,
        profile=None,
        checkpoint_path=str(tmp_path / "checkpoint.jsonl"),
        show_progress=False,
    )


def test_replays_cached_calls_without_touching_backend(tmp_path):
    cache = tmp_path / "checkpoint.jsonl"
    first_backend = CountingBackend()
    first = _run(tmp_path, first_backend)

    assert first_backend.calls == 1
    assert len(cache.read_text().splitlines()) == 1
    assert first.provenance["checkpoint"]["reused_calls"] == 0
    assert first.provenance["reused_samples"] == 0

    second_backend = CountingBackend()
    second = _run(tmp_path, second_backend)

    assert second_backend.calls == 0
    assert second.tasks["fake"].question_results[0].reused is True
    assert second.tasks["fake"].question_results[0].raw_response == "chat:question one"
    assert second.tasks["fake"].sub_metrics["reused_samples"] == 1
    assert second.provenance["reused_samples"] == 1
    assert second.provenance["checkpoint"]["reused_calls"] == 1
    assert second.provenance["checkpoint"]["loaded_entries"] == 1


def test_generate_calls_are_cached(tmp_path):
    _run(tmp_path, CountingBackend(), task=GenerateFakeTask())

    backend = CountingBackend()
    result = _run(tmp_path, backend, task=GenerateFakeTask())

    assert backend.calls == 0
    assert result.tasks["fake"].question_results[0].raw_response == "generated:question one"


def test_different_model_does_not_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), model_id="test/model")

    backend = CountingBackend(model_id="other/model")
    _run(tmp_path, backend, model_id="other/model")

    assert backend.calls == 1


def test_different_generation_params_do_not_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), task=FakeTask(max_new_tokens=8))

    backend = CountingBackend()
    _run(tmp_path, backend, task=FakeTask(max_new_tokens=16))

    assert backend.calls == 1


def test_malformed_checkpoint_lines_are_ignored(tmp_path):
    cache = tmp_path / "checkpoint.jsonl"
    cache.write_text('not json\n{"key": "broken"}\n')

    backend = CountingBackend()
    result = _run(tmp_path, backend)

    assert backend.calls == 1
    assert result.provenance["checkpoint"]["loaded_entries"] == 0


def test_checkpoint_appends_new_calls(tmp_path):
    _run(tmp_path, CountingBackend(), task=FakeTask(prompt="one"))

    backend = CountingBackend()
    result = _run(tmp_path, backend, task=FakeTask(prompt="two"))

    assert backend.calls == 1
    assert result.provenance["checkpoint"]["path"].endswith("checkpoint.jsonl")
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 2


def test_full_replay_omits_fabricated_performance(tmp_path):
    _run(tmp_path, CountingBackend())

    backend = CountingBackend()
    result = _run(tmp_path, backend)

    assert backend.calls == 0
    assert result.provenance["telemetry"]["total_calls"] == 0
    assert result.local_efficiency is None
    assert result.tasks["fake"].avg_tokens_per_sec == 0.0


def test_transformed_prompts_still_mark_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), task=TransformedPromptTask())

    backend = CountingBackend()
    result = _run(tmp_path, backend, task=TransformedPromptTask())

    assert backend.calls == 0
    assert result.tasks["fake"].question_results[0].reused is True
    assert result.tasks["fake"].sub_metrics["reused_samples"] == 1


def test_different_backend_does_not_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), meta_backend="mock")

    backend = CountingBackend()
    _run(tmp_path, backend, meta_backend="openai")

    assert backend.calls == 1


def test_different_reasoning_effort_does_not_reuse(tmp_path):
    _run(tmp_path, CountingBackend(), reasoning_effort=None)

    backend = CountingBackend()
    _run(tmp_path, backend, reasoning_effort="high")

    assert backend.calls == 1


def test_invalid_record_fields_are_ignored(tmp_path):
    cache = tmp_path / "checkpoint.jsonl"
    base = {
        "model_id": "test/model",
        "quantization": "GGUF_Q4_K_M",
        "backend": "mock",
        "backend_version": None,
        "reasoning_effort": None,
        "server": "",
        "evaluator_version": "2.0.0",
    }
    records = [
        {**base, "key": ["not", "a", "hash"], "response": "x", "tokens_generated": 1},
        {**base, "key": "a" * 64, "response": None, "tokens_generated": 1},
        {**base, "key": "b" * 64, "response": "x", "tokens_generated": "abc"},
        "not a dict",
    ]
    cache.write_text("\n".join(json.dumps(item) if not isinstance(item, str) else item for item in records) + "\n")

    backend = CountingBackend()
    result = _run(tmp_path, backend)

    assert backend.calls == 1
    assert result.provenance["checkpoint"]["loaded_entries"] == 0


def test_invalid_utf8_checkpoint_is_ignored(tmp_path):
    (tmp_path / "checkpoint.jsonl").write_bytes(b"\xff\xfe not valid utf8\n")

    backend = CountingBackend()
    result = _run(tmp_path, backend)

    assert backend.calls == 1
    assert result.provenance["checkpoint"]["loaded_entries"] == 0


def test_truncated_final_line_is_repaired(tmp_path):
    cache = tmp_path / "checkpoint.jsonl"
    cache.write_text('{"model_id": "other/model", "key": "x"}')

    backend = CountingBackend()
    _run(tmp_path, backend)

    lines = cache.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["model_id"] == "other/model"


def test_duplicate_records_count_once(tmp_path):
    from slm_bench.execution.checkpoint import CheckpointStore

    cache = tmp_path / "checkpoint.jsonl"
    record = {
        "key": "a" * 64,
        "model_id": "m",
        "quantization": "q",
        "backend": "b",
        "backend_version": None,
        "reasoning_effort": None,
        "server": "",
        "evaluator_version": "2.0.0",
        "response": "x",
        "tokens_generated": 1,
    }
    cache.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    store = CheckpointStore(str(cache), "m", "q", backend="b")

    assert store.loaded == 1


def test_non_ascii_prompts_roundtrip(tmp_path):
    _run(tmp_path, CountingBackend(), task=FakeTask(prompt="你好 🚀"))

    backend = CountingBackend()
    result = _run(tmp_path, backend, task=FakeTask(prompt="你好 🚀"))

    assert backend.calls == 0
    assert result.tasks["fake"].question_results[0].raw_response == "chat:你好 🚀"
