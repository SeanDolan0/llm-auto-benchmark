"""
Tests for Textual Terminal User Interface (TUI).
"""

import asyncio
import os
import pytest
from textual.containers import Container
from textual.widgets import Select, Button, Input, Checkbox, DataTable
from slm_bench.tui import SLMBenchTUI, REASONING_EFFORT_OPTIONS, _reasoning_effort_to_api
from slm_bench.models.schema import BenchmarkResult, HardwareInfo, ModelMetadata


def _make_result(eligible: bool) -> BenchmarkResult:
    metadata = ModelMetadata(
        model_id="test/model",
        model_slug="test-model",
        param_tier="3B-8B",
        backend="mock",
    )
    hardware = HardwareInfo(os="TestOS", cpu="CPU", cpu_cores=4, total_ram_gb=16.0)
    return BenchmarkResult(
        schema_version="2.0.0",
        evaluation_id="test-eval",
        status="FULL" if eligible else "PARTIAL",
        eligible_for_leaderboard=eligible,
        model=metadata,
        hardware=hardware,
        tasks={},
        categories={},
        slm_index=80.0,
    )


def test_tui_initialization_and_mounting():
    """Verify that the TUI app mounts widgets and exposes all configuration controls."""
    async def _test():
        # Ensure the test does not depend on or leak a caller's API key.
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            app = SLMBenchTUI(
                default_model="meta-llama/Llama-3.2-3B-Instruct",
                default_quant="BF16",
                default_effort="medium",
                default_output_dir="test-results",
                default_seed=123,
            )
            async with app.run_test() as pilot:
                assert app.eval_in_progress is False
                assert "sk_live_" not in app.default_api_key
                assert app.query_one("#input-model", Input).value == "meta-llama/Llama-3.2-3B-Instruct"
                assert app.query_one("#input-quant", Input).value == "BF16"
                assert app.query_one("#input-model-ram", Input).value == ""
                assert app.query_one("#input-seed", Input).value == "123"
                assert app.query_one("#input-output-dir", Input).value == "test-results"
                assert app.query_one("#select-profile", Select).value == "standard"
                assert app.query_one("#custom-tasks-container", Container).display is False
                assert app.query_one("#chk-task-mmlu_pro", Checkbox).value is True
                assert app.query_one("#chk-task-xstest", Checkbox).value is False
                assert app.query_one("#btn-run", Button) is not None
                assert app.query_one("#btn-stop", Button).disabled is True
                assert app.query_one("#btn-test-conn", Button) is not None
                assert app.query_one("#btn-submit", Button).disabled is True
                assert app.query_one("#live-stream-box") is not None
                assert app.query_one("#stream-log") is not None
                assert app.query_one("#results-table", DataTable) is not None
        finally:
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    asyncio.run(_test())


def test_tui_profile_selection():
    """Verify selecting an evaluation profile in the TUI."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test() as pilot:
            profile_sel = app.query_one("#select-profile", Select)
            profile_sel.value = "extended"
            await pilot.pause()
            assert app.query_one("#select-profile", Select).value == "extended"
            assert app.query_one("#custom-tasks-container", Container).display is False

    asyncio.run(_test())


def test_tui_profile_select_offers_small_and_medium():
    """Verify the TUI profile selector accepts the new sampled tiers."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test() as pilot:
            profile_sel = app.query_one("#select-profile", Select)
            for value in ("small", "medium"):
                profile_sel.value = value
                await pilot.pause()
                assert app.query_one("#select-profile", Select).value == value

    asyncio.run(_test())


def test_tui_custom_profile_shows_official_tasks():
    """Verify the custom profile reveals only the official benchmark task controls."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test() as pilot:
            profile_sel = app.query_one("#select-profile", Select)
            profile_sel.value = "custom"
            await pilot.pause()
            assert app.query_one("#custom-tasks-container", Container).display is True
            assert app.query_one("#chk-task-mmlu_pro", Checkbox).value is True
            assert app.query_one("#chk-task-xstest", Checkbox).value is False

    asyncio.run(_test())


def test_tui_invalid_seed_does_not_start_evaluation():
    """Verify invalid sampling seed is rejected before changing evaluation state."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test():
            app.query_one("#input-seed", Input).value = "not-a-number"
            app.start_evaluation()
            assert app.eval_in_progress is False
            assert app.query_one("#btn-run", Button).disabled is False

    asyncio.run(_test())


def test_tui_reasoning_effort_options_cover_common_levels():
    """Verify TUI exposes common reasoning levels and preserves explicit values."""
    values = [value for _, value in REASONING_EFFORT_OPTIONS]
    assert values == ["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]
    assert _reasoning_effort_to_api("default") is None
    assert _reasoning_effort_to_api("none") == "none"
    assert _reasoning_effort_to_api("xhigh") == "xhigh"
    assert _reasoning_effort_to_api("max") == "max"


def test_tui_reasoning_effort_aliases_are_normalized():
    """Verify common alias spellings resolve to canonical effort values."""
    assert SLMBenchTUI(default_effort="X-HIGH").default_effort == "xhigh"
    assert SLMBenchTUI(default_effort="maximum").default_effort == "max"
    assert SLMBenchTUI(default_effort="disabled").default_effort == "none"


def test_tui_reasoning_effort_rejects_unknown_value():
    """Verify unsupported reasoning values fail fast with a useful error."""
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        SLMBenchTUI(default_effort="ultra-high")


def test_tui_submission_requires_full_eligible_result():
    """Verify the submission button follows FULL leaderboard eligibility."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test():
            app._on_eval_finished(_make_result(eligible=False), "partial-result.json")
            assert app.query_one("#btn-submit", Button).disabled is True
            app._on_eval_finished(_make_result(eligible=True), "full-result.json")
            assert app.query_one("#btn-submit", Button).disabled is False

    asyncio.run(_test())


def test_tui_preset_selection():
    """Verify selecting a model preset updates the model ID input."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test() as pilot:
            select = app.query_one("#select-preset", Select)
            select.value = "Qwen/Qwen2.5-1.5B-Instruct"
            await pilot.pause()
            assert app.query_one("#input-model", Input).value == "Qwen/Qwen2.5-1.5B-Instruct"

    asyncio.run(_test())


def test_tui_cancellation_logic():
    """Verify stop_evaluation triggers cancellation event and UI update."""
    async def _test():
        app = SLMBenchTUI()
        async with app.run_test() as pilot:
            assert not app.cancel_event.is_set()
            app.eval_in_progress = True
            app.stop_evaluation()
            assert app.cancel_event.is_set()
            assert app.query_one("#btn-stop", Button).disabled is True

    asyncio.run(_test())


def test_tui_table_clearing_on_start():
    """Verify results table is cleared when starting a new evaluation."""
    async def _test():
        app = SLMBenchTUI(default_backend="mock")
        async with app.run_test() as pilot:
            table = app.query_one("#results-table", DataTable)
            table.add_row("Reasoning", "mmlu_pro", "100%", "5/5", "10ms", "50")
            assert table.row_count == 1

            # Calling start_evaluation clears table
            # We set cancel_event to stop the worker immediately
            app.cancel_event.set()
            app.start_evaluation()
            assert table.row_count == 0

    asyncio.run(_test())


def test_tui_streaming_and_sample_logging():
    """Verify in-place streaming updates live-stream-box and sample results are logged cleanly."""
    async def _test():
        from textual.widgets import Static, RichLog
        app = SLMBenchTUI()
        async with app.run_test() as pilot:
            app._append_token_chunk("Hello ")
            app._append_token_chunk("world!")
            live_box = app.query_one("#live-stream-box", Static)
            assert "Hello world!" in str(live_box.render())

            app._log_sample_result({
                "task_id": "test_task",
                "category": "reasoning",
                "sample_index": 1,
                "total_samples": 1,
                "passed": True,
                "latency_ms": 20.0,
                "tokens": 5,
                "prompt": "Say hello",
                "response": "Hello world!",
                "expected": "Hello world!",
                "extracted": "Hello world!",
            })
            log = app.query_one("#stream-log", RichLog)
            assert len(log.lines) > 0

    asyncio.run(_test())


def test_tui_profile_labels_include_small_and_medium():
    from slm_bench.tui import _profile_label

    assert "PARTIAL" in _profile_label("small")
    assert "PARTIAL" in _profile_label("medium")
    assert "1,000" in _profile_label("small")
    assert "10,000" in _profile_label("medium")

