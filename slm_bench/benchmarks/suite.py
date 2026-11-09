"""
Benchmark Suite Orchestrator, Provenance Tracking, and Multi-Track Aggregation.
Executes official, verifiable benchmarks, records fine-grained question logs,
and enforces strict leaderboard eligibility rules.
"""

import uuid
from typing import List, Dict, Optional, Any, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..backends.base import BaseInferenceBackend
from ..execution.inference_gateway import InferenceGateway
from ..metrics.telemetry import PerformanceCollector
from ..metrics.confidence import wilson_score_interval
from ..scoring.scoring_engine import ScoringEngine, DEFAULT_CATEGORY_WEIGHTS, EXTENDED_CATEGORY_WEIGHTS
from ..models.schema import BenchmarkResult, TaskMetric, CategoryScore, ModelMetadata, HardwareInfo
from .base import BaseBenchmarkTask
from .registry import get_task, get_profile_tasks, get_profile_sample_limits, BENCHMARK_REGISTRY
from ..datasets.manager import BENCHMARK_DATASET_META
from ..utils.logger import console, logger


class BenchmarkSuite:
    """Orchestrates official benchmark evaluation, telemetry, and composite index aggregation."""

    def __init__(self, category_weights: Optional[Dict[str, float]] = None):
        self.weights = category_weights or DEFAULT_CATEGORY_WEIGHTS
        self.scoring_engine = ScoringEngine(category_weights=self.weights)

    def resolve_tasks(
        self,
        selected_tasks: Optional[List[str]] = None,
        profile: Optional[str] = "standard",
    ) -> Dict[str, BaseBenchmarkTask]:
        """Resolve active official benchmark task instances based on profile or explicit list."""
        if selected_tasks:
            resolved: Dict[str, BaseBenchmarkTask] = {}
            for name in selected_tasks:
                clean_name = name.strip()
                try:
                    task_obj = get_task(clean_name)
                    resolved[task_obj.task_id] = task_obj
                except ValueError:
                    logger.warning(f"Skipping unrecognized official task: '{clean_name}'")
            return resolved

        profile_name = profile or "standard"
        profile_ids = get_profile_tasks(profile_name)
        resolved = {}
        for tid in profile_ids:
            try:
                task_obj = get_task(tid)
                resolved[task_obj.task_id] = task_obj
            except Exception as e:
                logger.warning(f"Could not load official task '{tid}' for profile '{profile_name}': {e}")
        return resolved

    def run_suite(
        self,
        backend: BaseInferenceBackend,
        model_meta: ModelMetadata,
        hardware_info: HardwareInfo,
        selected_tasks: Optional[List[str]] = None,
        profile: Optional[str] = "standard",
        sample_limit: Optional[int] = None,
        notes: Optional[str] = None,
        on_task_start: Optional[Any] = None,
        on_sample: Optional[Any] = None,
        on_task_complete: Optional[Any] = None,
        show_progress: bool = True,
        cancel_event: Optional[Any] = None,
        seed: int = 42,
    ) -> BenchmarkResult:
        """
        Execute evaluation suite and return standardized BenchmarkResult (Schema 2.0.0).
        """
        active_tasks = self.resolve_tasks(selected_tasks=selected_tasks, profile=profile)

        profile_sample_limits = get_profile_sample_limits(profile) or {}

        logger.info(
            f"Starting official evaluation suite for [bold green]{model_meta.model_id}[/bold green] "
            f"(profile: {profile or 'custom'}, tasks: {len(active_tasks)})..."
        )

        # Wrap backend in InferenceGateway with PerformanceCollector
        collector = PerformanceCollector()
        gateway = InferenceGateway(backend=backend, collector=collector)

        task_metrics: Dict[str, TaskMetric] = {}

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                main_bar = progress.add_task("[cyan]Running Benchmarks...", total=len(active_tasks))

                for task_id, task in active_tasks.items():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Benchmark cancelled by user.")
                    if on_task_start:
                        on_task_start(task.task_id, task.category)

                    # Log official benchmark provenance details
                    meta = BENCHMARK_DATASET_META.get(task.task_id, {})
                    ds_name = meta.get("name", task.task_id)
                    ds_split = meta.get("split", "test")
                    ds_rev = meta.get("revision", "official")
                    task_limit = sample_limit if sample_limit is not None else profile_sample_limits.get(task.task_id)
                    q_count = task_limit or meta.get("expected_count", "Full")
                    progress.update(
                        main_bar,
                        description=f"[cyan]Evaluating: {ds_name} ({ds_split}, {q_count} items)",
                    )

                    metric = task.run(
                        gateway,
                        sample_limit=task_limit,
                        on_sample=on_sample,
                        cancel_event=cancel_event,
                        seed=seed,
                    )

                    # Ensure confidence interval is calculated
                    if metric.ci_lower is None or metric.ci_upper is None:
                        ci_l, ci_u = wilson_score_interval(metric.passed_samples, metric.total_samples)
                        metric.ci_lower = ci_l
                        metric.ci_upper = ci_u

                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Benchmark cancelled by user.")
                    task_metrics[task.task_id] = metric
                    if on_task_complete:
                        on_task_complete(task.task_id, metric)
                    progress.advance(main_bar)
        else:
            for task_id, task in active_tasks.items():
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Benchmark cancelled by user.")
                if on_task_start:
                    on_task_start(task.task_id, task.category)

                metric = task.run(
                    gateway,
                    sample_limit=sample_limit if sample_limit is not None else profile_sample_limits.get(task.task_id),
                    on_sample=on_sample,
                    cancel_event=cancel_event,
                    seed=seed,
                )
                if metric.ci_lower is None or metric.ci_upper is None:
                    ci_l, ci_u = wilson_score_interval(metric.passed_samples, metric.total_samples)
                    metric.ci_lower = ci_l
                    metric.ci_upper = ci_u

                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Benchmark cancelled by user.")
                task_metrics[task.task_id] = metric
                if on_task_complete:
                    on_task_complete(task.task_id, metric)

        # 1. Scoring & Categorization
        scoring_weights = EXTENDED_CATEGORY_WEIGHTS if profile == "extended" else DEFAULT_CATEGORY_WEIGHTS
        engine = ScoringEngine(category_weights=scoring_weights)
        category_scores = engine.aggregate_categories(task_metrics)
        slm_index = engine.compute_slm_index(category_scores)
        coding_agentic_index = engine.compute_coding_agentic_index(category_scores)

        # Enforce strict leaderboard eligibility rules
        overall_status, is_eligible = engine.determine_run_status_and_eligibility(
            task_metrics=task_metrics,
            profile=profile,
            sample_limit=sample_limit,
        )

        # 2. Performance Telemetry & Efficiency Index
        telemetry_summary = collector.get_aggregate_summary()
        avg_tps = telemetry_summary.get("avg_decode_tokens_per_sec")
        if not avg_tps and task_metrics:
            valid_tps = [m.avg_tokens_per_sec for m in task_metrics.values() if m.avg_tokens_per_sec]
            avg_tps = (sum(valid_tps) / len(valid_tps)) if valid_tps else 0.0

        local_efficiency = engine.compute_local_efficiency_index(
            slm_index=slm_index,
            avg_tokens_per_sec=avg_tps,
            model_ram_gb=model_meta.model_ram_gb or hardware_info.model_ram_gb,
            peak_vram_gb=telemetry_summary.get("peak_vram_mb", 0.0) / 1024.0 if telemetry_summary.get("peak_vram_mb") else None,
        )

        eval_id = str(uuid.uuid4())
        logger.info(
            f"Evaluation complete. Status: [bold]{overall_status}[/bold] (Eligible: {is_eligible}). "
            f"Aggregate SLM-Index: [bold yellow]{slm_index}%[/bold yellow]"
        )

        return BenchmarkResult(
            schema_version="2.0.0",
            evaluation_id=eval_id,
            status=overall_status,
            eligible_for_leaderboard=is_eligible,
            is_legacy=False,
            profile=profile,
            sampling_seed=seed if (sample_limit is not None or profile_sample_limits) else None,
            model=model_meta,
            hardware=hardware_info,
            tasks=task_metrics,
            categories=category_scores,
            slm_index=slm_index,
            coding_agentic_index=coding_agentic_index,
            local_efficiency=local_efficiency,
            notes=notes,
            provenance={
                "profile": profile,
                "benchmark_count": len(task_metrics),
                "evaluator_version": "2.0.0",
                "telemetry": telemetry_summary,
                "dataset_provenance": {
                    tid: BENCHMARK_DATASET_META.get(tid, {}) for tid in task_metrics.keys()
                },
            },
        )
