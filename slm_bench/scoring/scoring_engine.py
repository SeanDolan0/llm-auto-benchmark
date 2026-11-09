"""
Consolidated Scoring Engine for SLM-Bench (Version 2.0).
Harmonizes official benchmark scoring categories, computes Wilson 95% confidence intervals,
separates Safety metrics from General Intelligence, and enforces strict leaderboard eligibility.
"""

from typing import Dict, List, Optional, Any
from ..models.schema import TaskMetric, CategoryScore
from ..metrics.confidence import wilson_score_interval


# Canonical category weights for General Intelligence (SLM-Index)
# Note: Safety is strictly evaluated as a separate pillar and NEVER inflates or reduces SLM-Index.
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "reasoning": 0.35,
    "mathematics": 0.35,
    "instruction": 0.30,
    "coding": 0.0,
    "tool_calling": 0.0,
    "context": 0.0,
}

EXTENDED_CATEGORY_WEIGHTS: Dict[str, float] = {
    "reasoning": 0.25,
    "mathematics": 0.25,
    "instruction": 0.20,
    "coding": 0.15,
    "tool_calling": 0.15,
    "context": 0.0,
}


class ScoringEngine:
    """Computes category-level aggregates and specialized leaderboard rankings."""

    def __init__(self, category_weights: Optional[Dict[str, float]] = None):
        self.category_weights = category_weights or DEFAULT_CATEGORY_WEIGHTS

    def aggregate_categories(self, task_metrics: Dict[str, TaskMetric]) -> Dict[str, CategoryScore]:
        """Group task metrics by category and compute category-level averages."""
        categories: Dict[str, CategoryScore] = {}

        # Collect metrics by category (excluding UNAVAILABLE tasks from active calculation)
        grouped: Dict[str, List[TaskMetric]] = {}
        for m in task_metrics.values():
            if m.status != "UNAVAILABLE":
                grouped.setdefault(m.category, []).append(m)

        all_cats = set(self.category_weights.keys()).union(grouped.keys())

        for cat_name in all_cats:
            metrics = grouped.get(cat_name, [])
            weight = self.category_weights.get(cat_name, 0.0)
            if metrics:
                avg_norm = sum(m.normalized_score for m in metrics) / len(metrics)
                categories[cat_name] = CategoryScore(
                    category=cat_name,
                    weight=weight,
                    normalized_score=round(avg_norm, 2),
                    task_count=len(metrics),
                    tasks=[m.task_id for m in metrics],
                )
            else:
                categories[cat_name] = CategoryScore(
                    category=cat_name,
                    weight=weight,
                    normalized_score=0.0,
                    task_count=0,
                    tasks=[],
                )

        return categories

    def compute_slm_index(self, categories: Dict[str, CategoryScore]) -> float:
        """
        Calculate weighted General Intelligence SLM-Index.
        Safety scores are intentionally excluded from general intelligence calculations.
        """
        # Exclude safety and context from the core general intelligence index
        active = [
            cs for cs in categories.values()
            if cs.task_count > 0 and cs.category != "safety" and cs.weight > 0
        ]
        if not active:
            # Fallback to equal weighting of active core pillars
            core_active = [cs for cs in categories.values() if cs.task_count > 0 and cs.category != "safety"]
            if not core_active:
                return 0.0
            return round(sum(cs.normalized_score for cs in core_active) / len(core_active), 2)

        total_w = sum(cs.weight for cs in active)
        weighted_sum = sum(cs.normalized_score * cs.weight for cs in active)
        return round(weighted_sum / total_w, 2)

    def compute_coding_agentic_index(self, categories: Dict[str, CategoryScore]) -> Optional[float]:
        """Calculate specialized Coding & Agentic Performance Index (if available)."""
        coding = categories.get("coding")
        tools = categories.get("tool_calling")

        has_coding = coding and coding.task_count > 0
        has_tools = tools and tools.task_count > 0

        if not has_coding and not has_tools:
            return None

        if has_coding and has_tools:
            return round(0.60 * coding.normalized_score + 0.40 * tools.normalized_score, 2)
        elif has_coding:
            return round(coding.normalized_score, 2)
        else:
            return round(tools.normalized_score, 2)

    def determine_run_status_and_eligibility(
        self,
        task_metrics: Dict[str, TaskMetric],
        profile: Optional[str] = "standard",
        sample_limit: Optional[int] = None,
    ) -> tuple[str, bool]:
        """
        Enforce strict leaderboard eligibility rules:
        - FULL: Complete official dataset, all samples evaluated, verified splits.
        - PARTIAL: Sampled evaluation (quick/small/medium profiles, or an explicit sample_limit). Ineligible for full leaderboard!
        - INVALID: Integrity or scoring failures.
        """
        if sample_limit is not None or profile in ("quick", "small", "medium"):
            return "PARTIAL", False

        for metric in task_metrics.values():
            if metric.status == "PARTIAL":
                return "PARTIAL", False
            if metric.status == "INVALID":
                return "INVALID", False

        # Only FULL results can contribute to official leaderboard
        return "FULL", True

    def compute_local_efficiency_index(
        self,
        slm_index: float,
        avg_tokens_per_sec: Optional[float],
        model_ram_gb: Optional[float],
        peak_vram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Local Hardware Efficiency metrics:
        - Score per GB of RAM/VRAM
        - Tokens/sec throughput
        - Hardware utilization score
        """
        effective_ram = peak_vram_gb or model_ram_gb or 8.0
        score_per_gb = round(slm_index / max(1.0, effective_ram), 2)
        tps = avg_tokens_per_sec or 0.0

        throughput_norm = min(100.0, (tps / 60.0) * 100.0)
        footprint_norm = min(100.0, (8.0 / max(1.0, effective_ram)) * 100.0)
        efficiency_score = round(0.5 * slm_index + 0.3 * throughput_norm + 0.2 * footprint_norm, 2)

        return {
            "score_per_gb": score_per_gb,
            "tokens_per_sec": round(tps, 2),
            "memory_footprint_gb": round(effective_ram, 2),
            "efficiency_score": efficiency_score,
        }
