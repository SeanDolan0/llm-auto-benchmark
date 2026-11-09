"""
Official Benchmarks package for SLM-Bench.
All custom handcrafted evaluations have been replaced with established, verifiable official benchmarks.
"""

from .base import BaseBenchmarkTask
from .registry import (
    BENCHMARK_REGISTRY,
    EVALUATION_PROFILES,
    PROFILE_TOTAL_SAMPLES,
    get_task,
    list_available_tasks,
    get_profile_tasks,
    get_profile_sample_limits,
)
from .suite import BenchmarkSuite

__all__ = [
    "BaseBenchmarkTask",
    "BenchmarkSuite",
    "BENCHMARK_REGISTRY",
    "EVALUATION_PROFILES",
    "PROFILE_TOTAL_SAMPLES",
    "get_task",
    "list_available_tasks",
    "get_profile_tasks",
    "get_profile_sample_limits",
]
