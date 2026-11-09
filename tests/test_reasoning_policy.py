"""Tests for reasoning-model normalization, token budgets, and truncation retries."""

from slm_bench.backends.openai_backend import OpenAICompatibleBackend
from slm_bench.benchmarks.suite import BenchmarkSuite
from slm_bench.models.schema import QuestionResult, TaskMetric


def _response(content, reasoning="", finish=None, tokens=10, reasoning_tokens=0):
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": reasoning},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "completion_tokens": tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def test_normalizes_reasoning_and_truncation(monkeypatch):
    backend = OpenAICompatibleBackend("model", truncation_policy="invalid")
    monkeypatch.setattr(
        backend,
        "_post",
        lambda path, payload: _response("", "thinking...", "length", 1024, 1024),
    )

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=1024)

    assert result.text == ""
    assert result.reasoning_text == "thinking..."
    assert result.reasoning_tokens == 1024
    assert result.finish_reason == "length"
    assert result.truncated is True


def test_retries_empty_truncated_and_doubles_budget(monkeypatch):
    budgets = []
    responses = [
        _response("", "thinking...", "length", 1024, 1024),
        _response("answer", "", "stop", 20, 0),
    ]

    def fake_post(path, payload):
        budgets.append(payload["max_tokens"])
        return responses.pop(0)

    backend = OpenAICompatibleBackend("model", max_token_budget=4096, max_retries=2)
    monkeypatch.setattr(backend, "_post", fake_post)

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=1024)

    assert budgets == [1024, 2048]
    assert result.text == "answer"
    assert result.truncated is False
    assert result.tokens_generated == 1044
    assert result.reasoning_tokens == 1024


def test_non_empty_truncated_answer_is_retried(monkeypatch):
    budgets = []
    responses = [
        _response("partial #### A", "", "length", 50),
        _response("complete #### A", "", "stop", 60),
    ]

    def fake_post(path, payload):
        budgets.append(payload["max_tokens"])
        return responses.pop(0)

    backend = OpenAICompatibleBackend("model")
    monkeypatch.setattr(backend, "_post", fake_post)

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=1024)

    assert budgets == [1024, 2048]
    assert result.text == "complete #### A"
    assert result.truncated is False


def test_invalid_policy_does_not_retry(monkeypatch):
    calls = []
    backend = OpenAICompatibleBackend("model", truncation_policy="invalid")
    monkeypatch.setattr(
        backend,
        "_post",
        lambda path, payload: calls.append(1) or _response("", "thinking", "length", 1024, 1024),
    )

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=1024)

    assert len(calls) == 1
    assert result.truncated is True


def test_retry_budget_is_capped(monkeypatch):
    budgets = []

    def fake_post(path, payload):
        budgets.append(payload["max_tokens"])
        return _response("", "thinking", "length", 1024, 1024)

    backend = OpenAICompatibleBackend("model", max_token_budget=2048, max_retries=5)
    monkeypatch.setattr(backend, "_post", fake_post)

    backend.chat([{"role": "user", "content": "q"}], max_new_tokens=1024)

    assert budgets == [1024, 2048]


def test_max_new_tokens_override_applies(monkeypatch):
    budgets = []

    def fake_post(path, payload):
        budgets.append(payload["max_tokens"])
        return _response("ok", "", "stop", 5)

    backend = OpenAICompatibleBackend("model", max_new_tokens=2048)
    monkeypatch.setattr(backend, "_post", fake_post)

    backend.chat([{"role": "user", "content": "q"}], max_new_tokens=512)

    assert budgets == [2048]


def test_thinking_off_is_sent_to_the_server(monkeypatch):
    payloads = []

    def fake_post(path, payload):
        payloads.append(payload)
        return _response("ok", "", "stop", 5)

    backend = OpenAICompatibleBackend("model", thinking="off")
    monkeypatch.setattr(backend, "_post", fake_post)

    backend.chat([{"role": "user", "content": "q"}], max_new_tokens=512)

    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_missing_usage_and_finish_reason_are_tolerated(monkeypatch):
    backend = OpenAICompatibleBackend("model")
    monkeypatch.setattr(
        backend,
        "_post",
        lambda path, payload: {"choices": [{"message": {"content": "hello there"}}]},
    )

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=512)

    assert result.text == "hello there"
    assert result.finish_reason is None
    assert result.truncated is False
    assert result.tokens_generated == 2


def test_question_result_serializes_truncation_fields():
    from slm_bench.models.schema import QuestionResult

    question = QuestionResult(
        question_id="q1",
        raw_prompt="q",
        raw_response="",
        is_correct=False,
        truncated=True,
        cap_exceeded=True,
        finish_reason="length",
        reasoning_tokens=1024,
    )

    assert question.truncated is True
    assert question.cap_exceeded is True
    assert question.finish_reason == "length"
    assert question.reasoning_tokens == 1024


def test_suite_counts_truncated_questions():
    metric = TaskMetric(
        task_id="fake",
        category="reasoning",
        status="PARTIAL",
        raw_score=0.0,
        normalized_score=0.0,
        total_samples=2,
        passed_samples=0,
        question_results=[
            QuestionResult(question_id="q1", raw_prompt="a", raw_response="", is_correct=False, truncated=True, cap_exceeded=True),
            QuestionResult(question_id="q2", raw_prompt="b", raw_response="", is_correct=False, truncated=False),
        ],
    )

    BenchmarkSuite._annotate_question_results(metric)

    assert metric.sub_metrics["truncated_samples"] == 1
    assert metric.sub_metrics["cap_exceeded_samples"] == 1


def test_doubles_to_the_hard_cap_without_repeating_at_cap(monkeypatch):
    budgets = []

    def fake_post(path, payload):
        budgets.append(payload["max_tokens"])
        return _response("", "thinking", "length", payload["max_tokens"], payload["max_tokens"])

    backend = OpenAICompatibleBackend("model", max_token_budget=1024, max_retries=10)
    monkeypatch.setattr(backend, "_post", fake_post)

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=128)

    assert budgets == [128, 256, 512, 1024]
    assert result.truncated is True
    assert result.cap_exceeded is True


def test_cap_not_exceeded_when_generation_finishes(monkeypatch):
    budgets = []
    responses = [
        _response("", "thinking", "length", 128, 128),
        _response("answer", "", "stop", 10, 0),
    ]

    def fake_post(path, payload):
        budgets.append(payload["max_tokens"])
        return responses.pop(0)

    backend = OpenAICompatibleBackend("model", max_token_budget=4096)
    monkeypatch.setattr(backend, "_post", fake_post)

    result = backend.chat([{"role": "user", "content": "q"}], max_new_tokens=128)

    assert budgets == [128, 256]
    assert result.truncated is False
    assert result.cap_exceeded is False
