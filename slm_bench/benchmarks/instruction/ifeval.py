"""
Official IFEval (Instruction Following Evaluation) Benchmark Adapter.
Evaluates verifiable instruction-following capability across 541 official prompts.
Measures both prompt-level strict accuracy and instruction-level strict accuracy.
Preserves individual constraint failure information and uses deterministic checkers without LLM judges.
"""

import time
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager
from .ifeval_checkers import verify_instruction_constraint
from ...utils.logger import logger


class IFEvalBenchmark(BaseBenchmarkTask):
    """Evaluates verifiable instruction following using deterministic constraint checkers."""

    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id="ifeval",
            category="instruction",
            description="IFEval official dataset (541 verifiable instruction-following prompts).",
            raw_metric_name="prompt_level_strict_accuracy",
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
        # Load official IFEval dataset (541 prompts)
        full_dataset = self.dm.load_ifeval()

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
        prompt_passed = 0
        total_instructions = 0
        instruction_passed = 0
        failed_constraint_types: Dict[str, int] = {}
        question_results: List[QuestionResult] = []

        total_latency_ms = 0.0
        total_tokens = 0

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("IFEval evaluation cancelled.")

            prompt = item["prompt"]
            inst_ids = item.get("instruction_id_list", [])
            raw_kwargs_list = item.get("kwargs", [])

            start_t = time.perf_counter()
            gen = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=1024,
                temperature=0.0,
            )
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += latency_ms
            total_tokens += gen.tokens_generated

            # Evaluate each instruction constraint
            all_inst_passed = True
            constraint_failures = []

            for i_idx, inst_id in enumerate(inst_ids):
                total_instructions += 1
                kwargs = raw_kwargs_list[i_idx] if i_idx < len(raw_kwargs_list) else {}
                if isinstance(kwargs, str):
                    try:
                        import json
                        kwargs = json.loads(kwargs)
                    except Exception:
                        kwargs = {}

                passed_c, msg = verify_instruction_constraint(inst_id, kwargs or {}, gen.text)
                if passed_c:
                    instruction_passed += 1
                else:
                    all_inst_passed = False
                    constraint_failures.append(f"{inst_id}: {msg}")
                    failed_constraint_types[inst_id] = failed_constraint_types.get(inst_id, 0) + 1

            if all_inst_passed and len(inst_ids) > 0:
                prompt_passed += 1

            q_res = QuestionResult(
                question_id=str(item.get("key", idx)),
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=f"{len(inst_ids) - len(constraint_failures)}/{len(inst_ids)} constraints met",
                expected_answer="All constraints satisfied",
                is_correct=all_inst_passed,
                reused=gen.reused,
                evaluation_error="; ".join(constraint_failures) if constraint_failures else None,
                tokens_prompt=len(prompt.split()),
                tokens_generated=gen.tokens_generated,
                latency_ms=round(latency_ms, 2),
            )
            question_results.append(q_res)

            if on_sample:
                on_sample(self.task_id, idx + 1, total, all_inst_passed)

        prompt_acc = (prompt_passed / total * 100.0) if total > 0 else 0.0
        inst_acc = (instruction_passed / total_instructions * 100.0) if total_instructions > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(prompt_passed, total)

        avg_latency = total_latency_ms / max(1, total)
        avg_tps = (total_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status=run_status,
            raw_score=round(prompt_passed / max(1, total), 4),
            raw_metric_name="prompt_level_strict_accuracy",
            normalized_score=round(prompt_acc, 2),
            total_samples=total,
            passed_samples=prompt_passed,
            failed_samples=total - prompt_passed,
            invalid_responses=0,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            dataset_name="google/IFEval",
            dataset_revision="main",
            split="train",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={
                "instruction_level_strict_accuracy": round(inst_acc, 2),
                "total_instructions": total_instructions,
                "passed_instructions": instruction_passed,
                "failed_constraint_types": failed_constraint_types,
            },
            question_results=question_results,
        )
