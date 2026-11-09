"""
Official EvalPlus (HumanEval+ and MBPP+) Benchmark Adapter.
Evaluates Python code generation against expanded hidden test suites.
Enforces isolated container-based sandbox execution.
If Docker container isolation is unavailable, marks the evaluation as UNAVAILABLE
rather than executing untrusted model-generated code directly on the host process.
"""

import re
import time
from typing import Optional, List, Dict, Any
from ..base import BaseBenchmarkTask
from ...backends.base import BaseInferenceBackend
from ...models.schema import TaskMetric, QuestionResult
from ...metrics.confidence import wilson_score_interval
from ...datasets.manager import DatasetManager
from ...execution.sandbox import ContainerSandbox
from ...utils.logger import logger


def extract_python_code(text: str, entry_point: Optional[str] = None) -> str:
    """Extract Python code block from model response."""
    clean = text.strip()
    # Search for markdown python block
    matches = re.findall(r"```(?:python)?\s*([\s\S]*?)```", clean, re.IGNORECASE)
    if matches:
        # Choose block containing entry point if specified
        if entry_point:
            for m in reversed(matches):
                if entry_point in m:
                    return m.strip()
        return matches[-1].strip()

    # If no fences, filter code lines
    lines = clean.splitlines()
    code_lines = []
    started = False
    for line in lines:
        if line.startswith("def ") or line.startswith("import ") or line.startswith("from ") or started:
            started = True
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else clean


class EvalPlusBenchmark(BaseBenchmarkTask):
    """Evaluates Python code synthesis on HumanEval+ and MBPP+ with container sandbox."""

    def __init__(self, subset: str = "humanevalplus", dataset_manager: Optional[DatasetManager] = None):
        super().__init__(
            task_id=f"evalplus_{subset}" if subset != "humanevalplus" else "evalplus",
            category="coding",
            description=f"EvalPlus ({subset}) code synthesis evaluated against expanded hidden tests.",
            raw_metric_name="pass@1",
            weight=1.0,
        )
        self.subset = subset
        self.dm = dataset_manager or DatasetManager()
        self.sandbox = ContainerSandbox()

    def run(
        self,
        backend: BaseInferenceBackend,
        sample_limit: Optional[int] = None,
        on_sample: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        seed: int = 42,
    ) -> TaskMetric:
        # Security Gate: Require container sandbox
        if not self.sandbox.is_available():
            logger.warning(
                f"Docker daemon is not running. Marking {self.task_id} as UNAVAILABLE. "
                "Untrusted code execution on host is blocked."
            )
            return TaskMetric(
                task_id=self.task_id,
                category=self.category,
                status="UNAVAILABLE",
                raw_score=0.0,
                raw_metric_name="pass@1",
                normalized_score=0.0,
                total_samples=0,
                passed_samples=0,
                failed_samples=0,
                invalid_responses=0,
                ci_lower=0.0,
                ci_upper=0.0,
                dataset_name=f"evalplus/{self.subset}",
                dataset_revision="v0.3.0",
                split="test",
                sub_metrics={"error": "Docker container sandbox is unavailable. Host execution of untrusted code is prohibited."},
                question_results=[],
            )

        full_dataset = self.dm.load_evalplus(self.subset)

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
        error_categories: Dict[str, int] = {
            "syntax_error": 0,
            "assertion_error": 0,
            "timeout": 0,
            "runtime_error": 0,
        }
        question_results: List[QuestionResult] = []

        total_latency_ms = 0.0
        total_tokens = 0

        system_prompt = (
            "You are an expert Python software engineer. Implement the required function thoroughly.\n"
            "Return only the complete Python implementation inside a ```python ``` block."
        )

        for idx, item in enumerate(dataset):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("EvalPlus evaluation cancelled.")

            prompt = item.get("prompt", "")
            entry_point = item.get("entry_point")
            hidden_test = item.get("test", "")

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

            extracted_code = extract_python_code(gen.text, entry_point=entry_point)

            # Build isolated test runner script
            # Never expose hidden test in model prompt!
            test_runner = f"""
{hidden_test}
check({entry_point})
"""
            is_passed, err_msg, exec_time = self.sandbox.run_code(extracted_code, test_runner)

            if is_passed:
                passed += 1
            else:
                if "timed out" in err_msg.lower():
                    error_categories["timeout"] += 1
                elif "syntaxerror" in err_msg.lower():
                    error_categories["syntax_error"] += 1
                elif "assertionerror" in err_msg.lower():
                    error_categories["assertion_error"] += 1
                else:
                    error_categories["runtime_error"] += 1

            q_res = QuestionResult(
                question_id=item.get("task_id", f"{self.subset}_{idx}"),
                raw_prompt=prompt,
                raw_response=gen.text,
                extracted_answer=extracted_code,
                expected_answer="Official expanded hidden tests pass",
                is_correct=is_passed,
                reused=gen.reused,
                finish_reason=gen.finish_reason,
                reasoning_tokens=gen.reasoning_tokens,
                truncated=gen.truncated,
                budget_start=gen.budget_start,
                completion_budget=gen.final_budget,
                cap_exceeded=gen.cap_exceeded,
                evaluation_error=None if is_passed else err_msg,
                tokens_prompt=len(prompt.split()),
                tokens_generated=gen.tokens_generated,
                latency_ms=round(latency_ms, 2),
            )
            question_results.append(q_res)

            if on_sample:
                on_sample(self.task_id, idx + 1, total, is_passed)

        pass_at_1 = (passed / total * 100.0) if total > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(passed, total)

        avg_latency = total_latency_ms / max(1, total)
        avg_tps = (total_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        return TaskMetric(
            task_id=self.task_id,
            category=self.category,
            status=run_status,
            raw_score=round(passed / max(1, total), 4),
            raw_metric_name="pass@1",
            normalized_score=round(pass_at_1, 2),
            total_samples=total,
            passed_samples=passed,
            failed_samples=total - passed,
            invalid_responses=error_categories["syntax_error"],
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            dataset_name=f"evalplus/{self.subset}",
            dataset_revision="v0.3.0",
            split="test",
            avg_latency_ms=round(avg_latency, 2),
            avg_tokens_per_sec=round(avg_tps, 2),
            sub_metrics={
                "error_categories": error_categories,
                "subset": self.subset,
            },
            question_results=question_results,
        )
