"""
Official GSM8K Benchmark Adapter.
Evaluates multi-step arithmetic reasoning on the official test split (1,319 problems).
Uses strict deterministic numerical extraction from ground truth '#### <number>' format.
Never awards credit merely because a number appears somewhere in the explanation.
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


def normalize_numeric_string(s: str) -> Optional[str]:
    """Normalize numeric string by removing commas, currency signs, and trailing zero decimals."""
    if not s:
        return None
    clean = s.strip()
    clean = clean.replace("$", "").replace(",", "").replace("%", "").strip()

    # Match integer or float
    m = re.match(r"^-?\d+(?:\.\d+)?$", clean)
    if not m:
        return None

    try:
        val = float(clean)
        # If integer representation, return as integer string
        if val.is_integer():
            return str(int(val))
        return str(val)
    except ValueError:
        return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Deterministically extract the final numerical answer from model response.
    Prioritizes:
    1. '#### <number>'
    2. '\\boxed{<number>}'
    3. 'the answer is <number>'
    4. Terminal line number
    Never accepts arbitrary numbers mentioned in reasoning.
    """
    if not text:
        return None
    clean = text.strip()

    # 1. Official #### marker
    if "####" in clean:
        after_hash = clean.split("####")[-1].strip()
        num_m = re.search(r"(-?\$?\d[\d,]*(?:\.\d+)?)", after_hash)
        if num_m:
            norm = normalize_numeric_string(num_m.group(1))
            if norm:
                return norm

    # 2. Boxed notation
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", clean)
    if boxed:
        num_m = re.search(r"(-?\$?\d[\d,]*(?:\.\d+)?)", boxed[-1].strip())
        if num_m:
            norm = normalize_numeric_string(num_m.group(1))
            if norm:
                return norm

    # 3. Explicit final answer phrases
    phrases = re.findall(
        r"(?:the (?:final )?answer is|result is|the total is)\s*[:=]?\s*(-?\$?\d[\d,]*(?:\.\d+)?)",
        clean,
        re.IGNORECASE,
    )
    if phrases:
        norm = normalize_numeric_string(phrases[-1])
        if norm:
            return norm

    # 4. Strict last line match if standalone number
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    if lines:
        last = lines[-1].rstrip(".")
        num_m = re.search(r"^(?:answer\s*[:=]?\s*)?(-?\$?\d[\d,]*(?:\.\d+)?)$", last, re.IGNORECASE)
        if num_m:
            norm = normalize_numeric_string(num_m.group(1))
            if norm:
                return norm

    return None


class GSM8KBenchmark(BaseBenchmarkTask):
    """Evaluates multi-step mathematical word problem solving on official GSM8K."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="gsm8k",
            category="mathematics",
            description="GSM8K official test split (1,319 multi-step math word problems).",
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
        # Load official GSM8K test split (1,319 questions)
        full_dataset = self.dm.load_gsm8k()

        if sample_limit and sample_limit < len(full_dataset):
            dataset, selected_ids = self.dm.sample_stratified(
                full_dataset,
                sample_size=sample_limit,
                seed=seed,
            )
            run_status = "PARTIAL"
        else:
            dataset = full_dataset
            run_status = "FULL"

        total = len(dataset)
        passed = 0
        invalid_count = 0
        question_results: List[QuestionResult] = []

        system_prompt = (
            "You are a helpful math assistant. Solve the math word problem step by step. "
            "At the end of your explanation, write '#### <number>' indicating your final numerical answer."
        )

        total_latency_ms = 0.0
        total_tokens = 0

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("GSM8K evaluation cancelled.")

            prompt = f"Question: {item['question']}\n\nAnswer:"

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

            extracted = extract_gsm8k_answer(gen.text)

            # Ground truth answer is after ####
            expected_raw = item["answer"].split("####")[-1].strip()
            expected_norm = normalize_numeric_string(expected_raw) or expected_raw

            is_correct = False
            if extracted is None:
                invalid_count += 1
            elif extracted == expected_norm:
                is_correct = True
                passed += 1

            q_res = QuestionResult(
                question_id=item["question_id"],
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=extracted,
                expected_answer=expected_norm,
                is_correct=is_correct,
                reused=gen.reused,
                finish_reason=gen.finish_reason,
                reasoning_tokens=gen.reasoning_tokens,
                truncated=gen.truncated,
                budget_start=gen.budget_start,
                completion_budget=gen.final_budget,
                cap_exceeded=gen.cap_exceeded,
                evaluation_error=None if extracted else "Unparsable or missing final numerical answer",
                tokens_prompt=len(prompt.split()),
                tokens_generated=gen.tokens_generated,
                latency_ms=round(latency_ms, 2),
            )
            question_results.append(q_res)

            if on_sample:
                on_sample(self.task_id, idx + 1, total, is_correct)

        accuracy = (passed / total) * 100.0 if total > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(passed, total)

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
            dataset_name="openai/gsm8k",
            dataset_revision="main",
            split="test",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            question_results=question_results,
        )
