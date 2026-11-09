"""
Official BIG-Bench Hard (BBH) Benchmark Adapter.
Evaluates algorithmic, multi-step symbolic deduction, and logical reasoning across 23 official subtasks.
Preserves task-specific evaluation and strict exact matching.
"""

import re
import time
from typing import Optional, List, Dict, Any, Tuple
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager, BBH_TASKS
from ...utils.logger import logger


def normalize_bbh_target(target: str) -> str:
    """Normalize official target string."""
    t = target.strip()
    # If target is option like (A), extract letter
    m = re.match(r"^\(([A-Z])\)$", t)
    if m:
        return m.group(1).upper()
    return t.lower()


def extract_bbh_answer(text: str, task_name: str) -> Optional[str]:
    """
    Task-specific answer extraction for BBH subtasks.
    Deterministically captures multiple-choice letter, boolean, or exact short answer.
    """
    clean = text.strip()
    if not clean:
        return None

    # Check for boxed answer
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", clean)
    if boxed:
        cand = boxed[-1].strip()
        m = re.search(r"\b([A-Z])\b", cand)
        return m.group(1).upper() if m else cand.lower()

    # Check for #### format
    if "####" in clean:
        after_hash = clean.split("####")[-1].strip()
        lines = [l.strip() for l in after_hash.splitlines() if l.strip()]
        if lines:
            line = lines[0]
            m = re.search(r"\b([A-Z])\b", line)
            return m.group(1).upper() if m else line.lower()

    # Task is boolean / logic
    if any(k in task_name for k in ["boolean", "causal", "sports", "web_of_lies", "formal_fallacies"]):
        matches = re.findall(r"\b(true|false|valid|invalid|yes|no)\b", clean, re.IGNORECASE)
        if matches:
            return matches[-1].lower()

    # Check "the answer is (A)" or "the answer is B"
    mc_phrase = re.findall(r"(?:the (?:correct )?answer is|choice is)\s*[:=]?\s*\(?([A-Z])\)?", clean, re.IGNORECASE)
    if mc_phrase:
        return mc_phrase[-1].upper()

    # Freeform phrase
    phrase = re.findall(r"(?:the (?:correct )?answer is|solution is)\s*[:=]?\s*([^\n\.]+)", clean, re.IGNORECASE)
    if phrase:
        ans = phrase[-1].strip()
        m = re.search(r"\b([A-Z])\b", ans)
        return m.group(1).upper() if m else ans.lower()

    # Last line fallback
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        m = re.search(r"\b([A-Z])\b", last)
        if m:
            return m.group(1).upper()
        return last.lower()

    return None


class BBHBenchmark(BaseBenchmarkTask):
    """Evaluates 23 BIG-Bench Hard algorithmic and symbolic tasks."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="bbh",
            category="reasoning",
            description="BIG-Bench Hard (BBH) 23 algorithmic and multi-step deduction tasks.",
            raw_metric_name="accuracy",
            weight=1.0,
        )
        self.dm = dataset_manager or DatasetManager()

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        seed: int = 42,
    ) -> TaskMetric:
        # Load all 23 official BBH tasks
        tasks_data = self.dm.load_bbh()

        # Flatten into list of items with task tags
        all_items: List[Dict[str, Any]] = []
        for t_name, examples in tasks_data.items():
            for ex in examples:
                all_items.append(ex)

        if sample_limit and sample_limit < len(all_items):
            dataset, selected_ids = self.dm.sample_stratified(
                all_items,
                sample_size=sample_limit,
                seed=seed,
                stratify_key="task",
            )
            run_status = "PARTIAL"
        else:
            dataset = all_items
            run_status = "FULL"

        total = len(dataset)
        passed = 0
        invalid_count = 0
        task_stats: Dict[str, Dict[str, int]] = {}
        question_results: List[QuestionResult] = []

        system_prompt = (
            "Follow the instructions and solve the problem step by step. "
            "Conclude your reasoning by outputting '#### <final answer>' on the last line."
        )

        total_latency_ms = 0.0
        total_tokens = 0

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("BBH evaluation cancelled.")

            subtask = item.get("task", "general")
            task_stats.setdefault(subtask, {"total": 0, "passed": 0})
            task_stats[subtask]["total"] += 1

            prompt = f"Task: {subtask}\nInput:\n{item['input']}\n\nAnswer:"

            start_t = time.perf_counter()
            gen = backend.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=512,
                temperature=0.0,
            )
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += latency_ms
            total_tokens += gen.tokens_generated

            extracted = extract_bbh_answer(gen.text, task_name=subtask)
            norm_expected = normalize_bbh_target(item["target"])

            is_correct = False
            if extracted is None:
                invalid_count += 1
            else:
                norm_pred = normalize_bbh_target(extracted)
                if norm_pred == norm_expected:
                    is_correct = True
                    passed += 1
                    task_stats[subtask]["passed"] += 1

            q_res = QuestionResult(
                question_id=item["id"],
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=extracted,
                expected_answer=item["target"],
                is_correct=is_correct,
                reused=gen.reused,
                evaluation_error=None if extracted else "Unparsable or ambiguous answer",
                tokens_prompt=len(prompt.split()),
                tokens_generated=gen.tokens_generated,
                latency_ms=round(latency_ms, 2),
            )
            question_results.append(q_res)

            if on_sample:
                on_sample(self.task_id, idx + 1, total, is_correct)

        accuracy = (passed / total) * 100.0 if total > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(passed, total)

        sub_breakdown = {}
        for s_name, stats in task_stats.items():
            s_tot = stats["total"]
            s_pas = stats["passed"]
            s_acc = (s_pas / s_tot * 100.0) if s_tot > 0 else 0.0
            sub_breakdown[s_name] = {
                "accuracy": round(s_acc, 2),
                "passed": s_pas,
                "total": s_tot,
            }

        avg_latency = total_latency_ms / max(1, total)
        avg_tps = (total_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status=run_status,
            raw_score=round(passed / max(1, total), 4),
            raw_metric_name="accuracy",
            normalized_score=round(accuracy, 2),
            total_samples=total,
            passed_samples=passed,
            failed_samples=total - passed,
            invalid_responses=invalid_count,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            dataset_name="suzgunmirac/BIG-Bench-Hard",
            dataset_revision="main",
            split="test",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"by_subtask": sub_breakdown},
            question_results=question_results,
        )
