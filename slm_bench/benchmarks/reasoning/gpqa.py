"""
Official GPQA Diamond Benchmark Adapter.
Evaluates Google-proof, expert-validated graduate-level science reasoning across Biology, Chemistry, and Physics.
Uses the genuine 198-question GPQA Diamond dataset from idavidrein/gpqa.
Choices are randomized with a reproducible seed and the answer key is remapped before evaluation.
"""

import random
import time
import re
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager
from .mmlu_pro import extract_mmlu_pro_answer
from ...utils.logger import logger


def extract_gpqa_choice(text: str) -> Optional[str]:
    """
    Deterministic extraction of choice A, B, C, or D.
    Strictly prohibits granting credit for mentioning answer words in explanations.
    """
    return extract_mmlu_pro_answer(text, num_options=4)


class GPQADiamondBenchmark(BaseBenchmarkTask):
    """Evaluates graduate-level domain reasoning on expert-validated scientific questions."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="gpqa_diamond",
            category="reasoning",
            description="GPQA Diamond expert scientific reasoning across biology, chemistry, and physics.",
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
        # Load official 198-question GPQA Diamond dataset
        full_dataset = self.dm.load_gpqa_diamond()

        if sample_limit and sample_limit < len(full_dataset):
            dataset, selected_ids = self.dm.sample_stratified(
                full_dataset,
                sample_size=sample_limit,
                seed=seed,
                stratify_key="high_level_domain",
            )
            run_status = "PARTIAL"
        else:
            dataset = full_dataset
            run_status = "FULL"

        total = len(dataset)
        passed = 0
        invalid_count = 0
        domain_stats: Dict[str, Dict[str, int]] = {}
        question_results: List[QuestionResult] = []

        system_prompt = (
            "What is the correct answer to this question? "
            "Think step by step and then finish your answer by stating '#### <choice>' "
            "where <choice> is one of A, B, C, or D."
        )

        total_latency_ms = 0.0
        total_tokens = 0

        # Reproducible random generator for choice shuffling
        shuffle_rng = random.Random(seed)

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("GPQA Diamond evaluation cancelled.")

            domain = item.get("high_level_domain") or item.get("subdomain") or "General Science"
            domain_stats.setdefault(domain, {"total": 0, "passed": 0})
            domain_stats[domain]["total"] += 1

            # Prepare choices and shuffle order reproducibly
            correct_text = item["correct_answer"]
            incorrects = [item["incorrect_1"], item["incorrect_2"], item["incorrect_3"]]
            choices = [correct_text] + incorrects
            shuffle_rng.shuffle(choices)

            # Determine remapped correct option letter
            correct_index = choices.index(correct_text)
            expected_letter = chr(ord('A') + correct_index)

            options_str = "\n".join([f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)])
            prompt = f"Question: {item['question']}\n\nChoices:\n{options_str}\n\nAnswer:"

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

            extracted = extract_gpqa_choice(gen.text)
            is_correct = False
            if extracted is None:
                invalid_count += 1
            elif extracted.upper() == expected_letter:
                is_correct = True
                passed += 1
                domain_stats[domain]["passed"] += 1

            q_res = QuestionResult(
                question_id=item["question_id"],
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=extracted,
                expected_answer=expected_letter,
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
        for d_name, stats in domain_stats.items():
            d_tot = stats["total"]
            d_pas = stats["passed"]
            d_acc = (d_pas / d_tot * 100.0) if d_tot > 0 else 0.0
            sub_breakdown[d_name] = {
                "accuracy": round(d_acc, 2),
                "passed": d_pas,
                "total": d_tot,
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
            dataset_name="idavidrein/gpqa",
            dataset_revision="diamond-4b24",
            split="diamond",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={"by_domain": sub_breakdown},
            question_results=question_results,
        )
