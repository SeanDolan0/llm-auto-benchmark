"""
Official MATH-500 Benchmark Adapter.
Evaluates high-school competition mathematics across 500 test questions from HuggingFaceH4/MATH-500.
Employs SymPy-based mathematical equivalence, LaTeX normalization, and rejects substring shortcuts.
Reports accuracy by mathematical subject and difficulty level.
"""

import time
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager
from .math_equivalence import extract_math_answer, is_math_equivalent
from ...utils.logger import logger


class MATH500Benchmark(BaseBenchmarkTask):
    """Evaluates challenging competition math across 7 subjects and 5 difficulty levels."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="math500",
            category="mathematics",
            description="MATH-500 test set (500 competition math problems across 7 subjects).",
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
        # Load official MATH-500 dataset (500 questions)
        full_dataset = self.dm.load_math500()

        if sample_limit and sample_limit < len(full_dataset):
            dataset, selected_ids = self.dm.sample_stratified(
                full_dataset,
                sample_size=sample_limit,
                seed=seed,
                stratify_key="subject",
            )
            run_status = "PARTIAL"
        else:
            dataset = full_dataset
            run_status = "FULL"

        total = len(dataset)
        passed = 0
        invalid_count = 0
        subject_stats: Dict[str, Dict[str, int]] = {}
        level_stats: Dict[str, Dict[str, int]] = {}
        question_results: List[QuestionResult] = []

        system_prompt = (
            "You are an expert mathematical problem solver. Solve the problem thoroughly.\n"
            "At the end of your solution, express your final answer enclosed strictly within \\boxed{...}."
        )

        total_latency_ms = 0.0
        total_tokens = 0

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("MATH-500 evaluation cancelled.")

            subject = item.get("subject", "General")
            subject_stats.setdefault(subject, {"total": 0, "passed": 0})
            subject_stats[subject]["total"] += 1

            level = str(item.get("level", 1))
            level_stats.setdefault(f"Level {level}", {"total": 0, "passed": 0})
            level_stats[f"Level {level}"]["total"] += 1

            prompt = f"Problem:\n{item['problem']}\n\nSolution:"

            start_t = time.perf_counter()
            gen = backend.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=1024,
                temperature=0.0,
            )
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += latency_ms
            total_tokens += gen.tokens_generated

            extracted = extract_math_answer(gen.text)
            expected = item.get("answer")

            is_correct = False
            if extracted is None:
                invalid_count += 1
            elif expected:
                is_correct = is_math_equivalent(extracted, expected)
                if is_correct:
                    passed += 1
                    subject_stats[subject]["passed"] += 1
                    level_stats[f"Level {level}"]["passed"] += 1

            q_res = QuestionResult(
                question_id=item["question_id"],
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=extracted,
                expected_answer=expected,
                is_correct=is_correct,
                reused=gen.reused,
                finish_reason=gen.finish_reason,
                reasoning_tokens=gen.reasoning_tokens,
                truncated=gen.truncated,
                budget_start=gen.budget_start,
                completion_budget=gen.final_budget,
                cap_exceeded=gen.cap_exceeded,
                evaluation_error=None if extracted else "Unparsable or missing \\boxed{} mathematical answer",
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
        for s_name, stats in subject_stats.items():
            s_tot = stats["total"]
            s_pas = stats["passed"]
            sub_breakdown[s_name] = {
                "accuracy": round((s_pas / s_tot * 100.0) if s_tot > 0 else 0.0, 2),
                "passed": s_pas,
                "total": s_tot,
            }

        lvl_breakdown = {}
        for l_name, stats in level_stats.items():
            l_tot = stats["total"]
            l_pas = stats["passed"]
            lvl_breakdown[l_name] = {
                "accuracy": round((l_pas / l_tot * 100.0) if l_tot > 0 else 0.0, 2),
                "passed": l_pas,
                "total": l_tot,
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
            dataset_name="HuggingFaceH4/MATH-500",
            dataset_revision="main",
            split="test",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={
                "by_subject": sub_breakdown,
                "by_level": lvl_breakdown,
            },
            question_results=question_results,
        )
