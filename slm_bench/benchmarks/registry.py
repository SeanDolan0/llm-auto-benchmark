"""
Central Benchmark Registry and Official Evaluation Profiles.
Provides discovery and instantiation for established official benchmarks.
All custom handcrafted evaluations have been completely removed.
"""

from typing import Dict, List, Optional, Type, Callable, Any
from .base import BaseBenchmarkTask
from ..datasets.manager import BENCHMARK_DATASET_META

# Official benchmark adapters
from .reasoning.mmlu_pro import MMLUProBenchmark
from .reasoning.gpqa import GPQADiamondBenchmark
from .reasoning.bbh import BBHBenchmark
from .mathematics.gsm8k import GSM8KBenchmark
from .mathematics.math500 import MATH500Benchmark
from .instruction.ifeval import IFEvalBenchmark
from .coding.evalplus import EvalPlusBenchmark
from .coding.livecodebench import LiveCodeBenchBenchmark
from .tools.bfcl import BFCLBenchmark
from .context.ruler import RULERBenchmark
from .safety.safety_adapter import SafetyBenchmark


# Registry mapping canonical task IDs to Task classes or factory callables
BENCHMARK_REGISTRY: Dict[str, Any] = {
    # Priority 1: Reasoning & Knowledge
    "mmlu_pro": MMLUProBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "bbh": BBHBenchmark,

    # Priority 2: Mathematics
    "gsm8k": GSM8KBenchmark,
    "math500": MATH500Benchmark,

    # Priority 3: Coding
    "evalplus": lambda: EvalPlusBenchmark(subset="humanevalplus"),
    "evalplus_humanevalplus": lambda: EvalPlusBenchmark(subset="humanevalplus"),
    "evalplus_mbppplus": lambda: EvalPlusBenchmark(subset="mbppplus"),
    "livecodebench": LiveCodeBenchBenchmark,

    # Priority 4: Instruction Following
    "ifeval": IFEvalBenchmark,

    # Priority 5: Function Calling
    "bfcl": BFCLBenchmark,

    # Priority 6: Long-Context
    "ruler": RULERBenchmark,

    # Priority 7: Safety
    "xstest": SafetyBenchmark,
}

# User convenience aliases mapped to canonical official tasks
TASK_ALIASES: Dict[str, str] = {
    "gpqa": "gpqa_diamond",
    "math": "math500",
    "instruction": "ifeval",
    "coding": "evalplus",
    "code": "evalplus",
    "humaneval": "evalplus",
    "tools": "bfcl",
    "tool_calling": "bfcl",
    "safety": "xstest",
    "context": "ruler",
}

# Evaluation Profiles
_STANDARD_TASKS: List[str] = [
    "mmlu_pro",
    "gpqa_diamond",
    "bbh",
    "gsm8k",
    "math500",
    "ifeval",
]

EVALUATION_PROFILES: Dict[str, List[str]] = {
    # Quick: development & smoke testing only. Uses seeded samples, marked PARTIAL.
    "quick": [
        "mmlu_pro",
        "gpqa_diamond",
        "gsm8k",
        "math500",
        "ifeval",
    ],
    # Small: 25 samples per standard task. Rough sanity check, marked PARTIAL.
    "small": list(_STANDARD_TASKS),
    # Medium: 100 samples per standard task. Pre-flight smell test, marked PARTIAL.
    "medium": list(_STANDARD_TASKS),
    # Standard: complete official datasets for core reasoning, math, and instruction benchmarks.
    "standard": list(_STANDARD_TASKS),
    # Extended: complete official suite including coding, tools, long-context, and safety.
    "extended": _STANDARD_TASKS + [
        "evalplus",
        "bfcl",
        "ruler",
        "xstest",
    ],
}

# Target sample totals for sampled profiles. None means the full official split.
PROFILE_TOTAL_SAMPLES: Dict[str, Optional[int]] = {
    "quick": 25,
    "small": 1000,
    "medium": 10000,
    "standard": None,
    "extended": None,
}


def get_task(task_id: str) -> BaseBenchmarkTask:
    """Instantiate and return benchmark task instance by ID or alias."""
    clean_id = task_id.lower().strip()
    canonical_id = TASK_ALIASES.get(clean_id, clean_id)

    if canonical_id not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unrecognized benchmark task '{task_id}'. "
            f"Available official benchmarks: {list(BENCHMARK_REGISTRY.keys())}"
        )

    task_factory = BENCHMARK_REGISTRY[canonical_id]
    if callable(task_factory):
        return task_factory()
    return task_factory()


def list_available_tasks() -> List[str]:
    """Return all registered official benchmark task IDs."""
    return sorted(list(BENCHMARK_REGISTRY.keys()))


def get_profile_tasks(profile_name: str) -> List[str]:
    """Retrieve task IDs corresponding to an evaluation profile."""
    p_lower = profile_name.lower().strip()
    if p_lower not in EVALUATION_PROFILES:
        raise ValueError(f"Unknown evaluation profile '{profile_name}'. Available: {list(EVALUATION_PROFILES.keys())}")
    return EVALUATION_PROFILES[p_lower]


def get_profile_sample_limits(profile_name: Optional[str]) -> Optional[Dict[str, int]]:
    """Return per-task sample limits for a sampled profile, or None for a full split."""
    if not profile_name:
        return None
    p_lower = profile_name.lower().strip()
    total = PROFILE_TOTAL_SAMPLES.get(p_lower)
    if total is None:
        return None
    task_ids = EVALUATION_PROFILES.get(p_lower, [])
    capacities = {
        tid: int(BENCHMARK_DATASET_META.get(tid, {}).get("expected_count") or total)
        for tid in task_ids
    }
    return _water_fill(total, task_ids, capacities)


def _water_fill(total: int, task_ids: List[str], capacities: Dict[str, int]) -> Dict[str, int]:
    """Split a sample budget equally, capping each task at its dataset size."""
    if not task_ids:
        return {}
    caps = {tid: max(0, int(capacities.get(tid, 0))) for tid in task_ids}
    if total >= sum(caps.values()):
        return caps
    lo, hi = 0, max(caps.values() or [0])
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(cap, mid) for cap in caps.values()) <= total:
            lo = mid
        else:
            hi = mid - 1
    allocation = {tid: min(caps[tid], lo) for tid in task_ids}
    remainder = total - sum(allocation.values())
    while remainder > 0:
        progressed = False
        for tid in task_ids:
            if remainder <= 0:
                break
            if allocation[tid] < caps[tid]:
                allocation[tid] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return allocation
