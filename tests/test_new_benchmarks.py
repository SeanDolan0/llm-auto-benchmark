"""
Tests for Official Benchmark Registry, Profiles, Stratified Sampling, and Statistical Confidence.
Verifies that all custom SLM benchmarks are completely absent from the registry,
sampling is deterministic and reproducible, and leaderboard eligibility is strictly enforced.
"""

import pytest
from slm_bench.benchmarks.registry import (
    get_task,
    list_available_tasks,
    get_profile_tasks,
    get_profile_sample_limits,
    BENCHMARK_REGISTRY,
    EVALUATION_PROFILES,
)
from slm_bench.benchmarks.suite import BenchmarkSuite
from slm_bench.backends.mock_backend import MockInferenceBackend
from slm_bench.models.schema import TaskMetric, ModelMetadata, HardwareInfo
from slm_bench.datasets.manager import DatasetManager, BENCHMARK_DATASET_META
from slm_bench.metrics.confidence import wilson_score_interval, compute_metric_confidence
from slm_bench.scoring.scoring_engine import ScoringEngine


def test_custom_slm_benchmarks_completely_removed():
    """Verify that all old custom SLM tasks are purged from the registry."""
    available = list_available_tasks()
    assert "slm_reasoning_cot" not in available
    assert "slm_coding_agent" not in available
    assert "slm_instruction_ifeval" not in available
    assert "slm_tool_calling_multiturn" not in available

    # Verify attempting to get old tasks raises ValueError
    with pytest.raises(ValueError):
        get_task("slm_reasoning_cot")
    with pytest.raises(ValueError):
        get_task("slm_coding_agent")


def test_official_benchmarks_registered():
    """Verify that only official, established benchmarks are registered."""
    available = list_available_tasks()
    official_expected = [
        "mmlu_pro",
        "gpqa_diamond",
        "bbh",
        "gsm8k",
        "math500",
        "ifeval",
        "evalplus",
        "bfcl",
        "xstest",
    ]
    for b_id in official_expected:
        assert b_id in available


def test_evaluation_profiles():
    quick = get_profile_tasks("quick")
    small = get_profile_tasks("small")
    medium = get_profile_tasks("medium")
    standard = get_profile_tasks("standard")
    extended = get_profile_tasks("extended")

    assert "mmlu_pro" in quick
    assert "gsm8k" in quick
    assert "ifeval" in quick

    assert "bbh" in standard
    assert len(standard) > len(quick)
    assert small == standard
    assert medium == standard

    assert "evalplus" in extended
    assert "bfcl" in extended
    assert "xstest" in extended
    assert len(extended) > len(standard)


def test_profile_sample_limits():
    quick = get_profile_sample_limits("quick")
    small = get_profile_sample_limits("small")
    medium = get_profile_sample_limits("medium")

    assert sum(quick.values()) == 25
    assert sum(small.values()) == 1000
    assert sum(medium.values()) == 10000

    assert small == {
        "mmlu_pro": 167,
        "gpqa_diamond": 167,
        "bbh": 167,
        "gsm8k": 167,
        "math500": 166,
        "ifeval": 166,
    }
    assert medium == {
        "mmlu_pro": 3721,
        "gpqa_diamond": 198,
        "bbh": 3721,
        "gsm8k": 1319,
        "math500": 500,
        "ifeval": 541,
    }
    assert get_profile_sample_limits("standard") is None
    assert get_profile_sample_limits("extended") is None
    assert get_profile_sample_limits(None) is None


def test_small_and_medium_are_partial_and_ineligible():
    engine = ScoringEngine()
    metric = TaskMetric(
        task_id="mmlu_pro",
        category="reasoning",
        status="FULL",
        raw_score=0.8,
        normalized_score=80.0,
        total_samples=100,
        passed_samples=80,
    )

    for profile in ("small", "medium"):
        status, eligible = engine.determine_run_status_and_eligibility(
            task_metrics={"mmlu_pro": metric},
            profile=profile,
        )
        assert status == "PARTIAL"
        assert eligible is False


def test_small_profile_applies_sample_limit():
    suite = BenchmarkSuite()
    backend = MockInferenceBackend("test/model")
    backend.load()
    meta = ModelMetadata(
        model_id="test/model-3B",
        model_slug="model-3B-BF16",
        param_count_b=3.0,
        param_tier="3B-8B",
        backend="mock",
    )
    hw = HardwareInfo(os="Linux", cpu="Xeon", cpu_cores=8, total_ram_gb=16.0)

    result = suite.run_suite(backend, meta, hw, profile="small", show_progress=False)

    assert result.profile == "small"
    assert result.status == "PARTIAL"
    assert result.eligible_for_leaderboard is False
    assert len(result.tasks) == 6
    assert sum(metric.total_samples for metric in result.tasks.values()) == 1000
    assert result.tasks["mmlu_pro"].total_samples == 167
    assert result.tasks["bbh"].total_samples == 167
    assert result.tasks["gpqa_diamond"].total_samples == 167
    assert result.tasks["math500"].total_samples == 166
    assert result.tasks["ifeval"].total_samples == 166


def test_explicit_sample_limit_overrides_profile_default():
    suite = BenchmarkSuite()
    backend = MockInferenceBackend("test/model")
    backend.load()
    meta = ModelMetadata(
        model_id="test/model-3B",
        model_slug="model-3B-BF16",
        param_count_b=3.0,
        param_tier="3B-8B",
        backend="mock",
    )
    hw = HardwareInfo(os="Linux", cpu="Xeon", cpu_cores=8, total_ram_gb=16.0)

    result = suite.run_suite(
        backend, meta, hw, profile="small", sample_limit=3, show_progress=False
    )

    assert result.status == "PARTIAL"
    for metric in result.tasks.values():
        assert metric.total_samples == 3


def test_custom_task_selection_does_not_inherit_profile_limit():
    suite = BenchmarkSuite()
    backend = MockInferenceBackend("test/model")
    backend.load()
    meta = ModelMetadata(
        model_id="test/model-3B",
        model_slug="model-3B-BF16",
        param_count_b=3.0,
        param_tier="3B-8B",
        backend="mock",
    )
    hw = HardwareInfo(os="Linux", cpu="Xeon", cpu_cores=8, total_ram_gb=16.0)

    result = suite.run_suite(
        backend,
        meta,
        hw,
        selected_tasks=["gpqa_diamond"],
        profile=None,
        show_progress=False,
    )

    assert list(result.tasks) == ["gpqa_diamond"]
    assert result.tasks["gpqa_diamond"].total_samples == 198
    assert result.status == "FULL"


def test_deterministic_stratified_sampling():
    dm = DatasetManager()
    dummy_data = [
        {"id": f"q_{i}", "category": "math" if i < 30 else ("physics" if i < 70 else "biology")}
        for i in range(100)
    ]

    # Two runs with same seed should yield exact same items
    sample1, ids1 = dm.sample_stratified(dummy_data, sample_size=10, seed=42, stratify_key="category")
    sample2, ids2 = dm.sample_stratified(dummy_data, sample_size=10, seed=42, stratify_key="category")

    assert ids1 == ids2
    assert len(sample1) == 10

    # Ensure categories are represented (stratification)
    cats = {item["category"] for item in sample1}
    assert "math" in cats
    assert "physics" in cats
    assert "biology" in cats

    # Different seed yields different deterministic order
    _, ids3 = dm.sample_stratified(dummy_data, sample_size=10, seed=999, stratify_key="category")
    assert ids1 != ids3


def test_wilson_confidence_intervals():
    ci_l, ci_u = wilson_score_interval(80, 100, confidence=0.95)
    assert 0.70 <= ci_l <= 0.75
    assert 0.85 <= ci_u <= 0.90
    assert ci_l < ci_u

    # Edge cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    assert wilson_score_interval(0, 10)[0] == 0.0
    assert wilson_score_interval(10, 10)[1] == 1.0


def test_leaderboard_eligibility_enforcement():
    engine = ScoringEngine()

    full_metric = TaskMetric(
        task_id="mmlu_pro",
        category="reasoning",
        status="FULL",
        raw_score=0.75,
        normalized_score=75.0,
        total_samples=100,
        passed_samples=75,
    )

    partial_metric = TaskMetric(
        task_id="gsm8k",
        category="mathematics",
        status="PARTIAL",
        raw_score=0.80,
        normalized_score=80.0,
        total_samples=10,
        passed_samples=8,
    )

    # 1. Full standard evaluation
    status_full, is_eligible_full = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric},
        profile="standard",
        sample_limit=None,
    )
    assert status_full == "FULL"
    assert is_eligible_full is True

    # 2. Sampled evaluation (quick profile)
    status_quick, is_eligible_quick = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric},
        profile="quick",
    )
    assert status_quick == "PARTIAL"
    assert is_eligible_quick is False

    # 3. Partial sample limit specified
    status_samp, is_eligible_samp = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric},
        profile="standard",
        sample_limit=50,
    )
    assert status_samp == "PARTIAL"
    assert is_eligible_samp is False

    # 4. Contains a partial task
    status_has_partial, is_eligible_has_partial = engine.determine_run_status_and_eligibility(
        task_metrics={"mmlu_pro": full_metric, "gsm8k": partial_metric},
        profile="standard",
        sample_limit=None,
    )
    assert status_has_partial == "PARTIAL"
    assert is_eligible_has_partial is False


def test_official_adapter_execution_with_mock():
    backend = MockInferenceBackend(model_id="mock-model", simulated_latency_ms=1.0)
    backend.load()

    # MMLU-Pro mock test with small sample
    mmlu = get_task("mmlu_pro")
    m_res = mmlu.run(backend, sample_limit=2)
    assert m_res.task_id == "mmlu_pro"
    assert m_res.status == "PARTIAL"  # Sample limit automatically marks status as PARTIAL!
    assert len(m_res.question_results) == 2

    # GPQA Diamond mock test with small sample
    gpqa = get_task("gpqa_diamond")
    g_res = gpqa.run(backend, sample_limit=2)
    assert g_res.task_id == "gpqa_diamond"
    assert g_res.status == "PARTIAL"
    assert len(g_res.question_results) == 2

    # GSM8K mock test with small sample
    gsm8k = get_task("gsm8k")
    gs_res = gsm8k.run(backend, sample_limit=2)
    assert gs_res.task_id == "gsm8k"
    assert gs_res.status == "PARTIAL"
    assert len(gs_res.question_results) == 2
