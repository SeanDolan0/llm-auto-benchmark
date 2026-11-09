"""
Official MMLU-Pro Benchmark Adapter.
Advanced multi-discipline reasoning across 14 academic subjects evaluated with 10 answer options (A-J).
Uses the genuine official dataset (TIGER-Lab/MMLU-Pro) and strict deterministic answer extraction.
"""

import re
import time
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager
from ...utils.logger import logger


def extract_mmlu_pro_answer(text: str, num_options: int = 10) -> Optional[str]:
    """
    Deterministically extract the selected multiple-choice letter (A through J).
    Rejects ambiguous, malformed, or out-of-range choices.
    Never gives credit based on keywords in the explanation.
    """
    if not text:
        return None
    clean = text.strip()

    valid_letters = [chr(ord('A') + i) for i in range(min(num_options, 10))]
    valid_str = "".join(valid_letters)

    # 1. Official boxed marker: \boxed{A}
    boxed = re.findall(rf"\\boxed\{{\s*([{valid_str}])\s*\}}", clean, re.IGNORECASE)
    if boxed:
        return boxed[-1].upper()

    # 2. Final answer marker: #### A
    if "####" in clean:
        after_hash = clean.split("####")[-1].strip()
        match = re.search(rf"\b([{valid_str}])\b", after_hash, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 3. Explicit choice phrase at the end:
    # "The answer is (A)", "The correct answer is B", "Option C"
    phrases = re.findall(
        rf"(?:the (?:correct )?answer is|the best option is|choice is|option is)\s*[:=]?\s*\(?([{valid_str}])\)?(?:\.|\s|$)",
        clean,
        re.IGNORECASE,
    )
    if phrases:
        return phrases[-1].upper()

    # 4. Strict terminal line match: e.g. "Answer: A" or line starting with "(A)"
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        m = re.search(rf"^(?:answer\s*[:=]?\s*)?\(?([{valid_str}])\)?\.?$", last_line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None


class MMLUProBenchmark(BaseBenchmarkTask):
    """Evaluates multi-discipline knowledge and reasoning across 10 answer options."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="mmlu_pro",
            category="reasoning",
            description="MMLU-Pro 10-choice professional reasoning across academic domains.",
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
        # Load official MMLU-Pro test dataset
        full_dataset = self.dm.load_mmlu_pro()

        if sample_limit and sample_limit < len(full_dataset):
            dataset, selected_ids = self.dm.sample_stratified(
                full_dataset,
                sample_size=sample_limit,
                seed=seed,
                stratify_key="category",
            )
            run_status = "PARTIAL"
        else:
            dataset = full_dataset
            run_status = "FULL"

        total = len(dataset)
        passed = 0
        invalid_count = 0
        subject_stats: Dict[str, Dict[str, int]] = {}
        question_results: List[QuestionResult] = []

        system_prompt = (
            "You are an expert academic reasoning assistant. Analyze the problem carefully.\n"
            "At the end of your explanation, write '#### <letter>' indicating your single final choice.\n"
            "Do not output ambiguous text."
        )

        total_latency_ms = 0.0
        total_tokens = 0

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("MMLU-Pro evaluation cancelled.")

            category = item.get("category", "general")
            subject_stats.setdefault(category, {"total": 0, "passed": 0})
            subject_stats[category]["total"] += 1

            options = item.get("options", [])
            options_str = "\n".join([f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)])
            prompt = f"Question:\n{item['question']}\n\nOptions:\n{options_str}\n\nReasoning and final answer:"

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

            extracted = extract_mmlu_pro_answer(gen.text, num_options=len(options))
            expected = item.get("answer")
            if not expected and "answer_index" in item:
                expected = chr(ord('A') + int(item["answer_index"]))

            is_correct = False
            if extracted is None:
                invalid_count += 1
            elif expected and extracted.upper() == expected.upper():
                is_correct = True
                passed += 1
                subject_stats[category]["passed"] += 1

            q_res = QuestionResult(
                question_id=item["question_id"],
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=extracted,
                expected_answer=expected,
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
        for cat, stats in subject_stats.items():
            c_tot = stats["total"]
            c_pas = stats["passed"]
            c_acc = (c_pas / c_tot * 100.0) if c_tot > 0 else 0.0
            sub_breakdown[cat] = {
                "accuracy": round(c_acc, 2),
                "passed": c_pas,
                "total": c_tot,
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
            dataset_name="TIGER-Lab/MMLU-Pro",
            dataset_revision="4b6efcb",
            split="test",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"by_subject": sub_breakdown},
            question_results=question_results,
        )
