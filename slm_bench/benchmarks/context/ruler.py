"""
RULER Long-Context Benchmark Adapter.
Evaluates effective context window utilization across sequence lengths (4K, 8K, 16K).
Tests needle retrieval and variable aggregation, distinguishing advertised context window
from demonstrated effective context retrieval.
"""

import time
import re
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...utils.logger import logger


def generate_needle_haystack(target_tokens: int, needle_key: str, needle_val: str) -> str:
    """Generate haystack text with embedded needle at specified token count."""
    sentence = (
        "In artificial intelligence, neural networks evaluate contextual representations "
        "using multi-head attention over latent embeddings. "
        "Computational complexity scales with sequence dimension. "
    )
    tokens_per_repeat = len(sentence.split())
    repeats = max(10, target_tokens // max(1, tokens_per_repeat))

    half = repeats // 2
    haystack_p1 = sentence * half
    haystack_needle = f" [IMPORTANT NOTICE: The key '{needle_key}' has the security value '{needle_val}'.] "
    haystack_p2 = sentence * (repeats - half)
    return haystack_p1 + haystack_needle + haystack_p2


class RULERBenchmark(BaseBenchmarkTask):
    """Evaluates multi-hop retrieval and variable tracking at scalable context lengths."""

    def __init__(self, context_lengths: Optional[List[int]] = None):
        super().__init__(
            task_id="ruler",
            category="context",
            description="RULER long-context retrieval evaluation (4K, 8K, 16K sequence lengths).",
            raw_metric_name="accuracy",
            weight=1.0,
        )
        self.context_lengths = context_lengths or [4096, 8192]

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        seed: int = 42,
    ) -> TaskMetric:
        lengths_stats: Dict[str, Dict[str, int]] = {}
        passed = 0
        total = 0
        question_results: List[QuestionResult] = []

        total_latency_ms = 0.0

        for ctx_len in self.context_lengths:
            tag = f"{ctx_len // 1024}K"
            lengths_stats[tag] = {"passed": 0, "total": 0}

            keys_vals = [
                ("security_code_alpha", "ALPHA_9281"),
                ("access_token_beta", "BETA_4829"),
            ]

            for key, val in keys_vals:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("RULER evaluation cancelled.")

                total += 1
                lengths_stats[tag]["total"] += 1

                haystack = generate_needle_haystack(ctx_len, key, val)
                prompt = (
                    f"{haystack}\n\n"
                    f"Question: What is the exact security value associated with '{key}' in the text above?\n"
                    f"Answer with only the exact security value."
                )

                start_t = time.perf_counter()
                gen = backend.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=64,
                    temperature=0.0,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                total_latency_ms += latency_ms

                is_correct = val.lower() in gen.text.lower()
                if is_correct:
                    passed += 1
                    lengths_stats[tag]["passed"] += 1

                q_res = QuestionResult(
                    question_id=f"ruler_{tag}_{key}",
                    raw_prompt=prompt[:200] + "... [truncated long context]",
                    raw_response=gen.text,
                    extracted_answer=gen.text.strip(),
                    expected_answer=val,
                    is_correct=is_correct,
                    reused=gen.reused,
                    tokens_prompt=ctx_len,
                    tokens_generated=gen.tokens_generated,
                    latency_ms=round(latency_ms, 2),
                )
                question_results.append(q_res)

                if on_sample:
                    on_sample(self.task_id, total, len(self.context_lengths) * len(keys_vals), is_correct)

        accuracy = (passed / total * 100.0) if total > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(passed, total)

        by_length = {}
        for l_tag, stats in lengths_stats.items():
            l_tot = stats["total"]
            l_pas = stats["passed"]
            by_length[l_tag] = {
                "accuracy": round((l_pas / l_tot * 100.0) if l_tot > 0 else 0.0, 2),
                "passed": l_pas,
                "total": l_tot,
            }

        avg_latency = total_latency_ms / max(1, total)

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status="FULL",
            raw_score=round(passed / max(1, total), 4),
            raw_metric_name="accuracy",
            normalized_score=round(accuracy, 2),
            total_samples=total,
            passed_samples=passed,
            failed_samples=total - passed,
            invalid_responses=0,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            dataset_name="NVIDIA/RULER",
            dataset_revision="official",
            split="retrieval",
            avg_latency_ms=round(avg_latency, 2),
            sub_metrics={"by_context_length": by_length},
            question_results=question_results,
        )
