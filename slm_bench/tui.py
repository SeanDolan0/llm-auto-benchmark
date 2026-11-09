"""
Interactive Terminal User Interface (TUI) for SLM-Bench.
Provides an easy-to-use graphical console with:
- Automatic hardware and environment detection
- Curated model presets and customizable configuration controls
- Endpoint pre-flight health checking ("Test Connection")
- Smooth in-place live generation streaming (without fragmented line breaks)
- Structured, readable evaluation log with full AI responses and clean formatting
- Cooperative cancellation (Stop button terminates active benchmarks cleanly)
- Scoreboard results table with automatic clearing on re-runs
- Integrated Pull Request submission modal to GitHub
- Keyboard shortcuts and Help overlay modal
"""

import os
import time
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, List

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Label,
    Input,
    Select,
    Button,
    Checkbox,
    ProgressBar,
    RichLog,
    DataTable,
)
from textual.screen import ModalScreen
from textual import work

from .models.hardware import get_hardware_info, is_endpoint_wsl_relayed, is_wsl_environment, generate_auto_system_notes
from .models.metadata import fetch_hf_model_metadata
from .models.memory_detector import auto_detect_model_ram_gb
from .backends.registry import get_backend
from .benchmarks.suite import BenchmarkSuite
from .benchmarks.registry import EVALUATION_PROFILES, PROFILE_TOTAL_SAMPLES
from .models.schema import BenchmarkResult
from .submission.git_manager import GitSubmissionManager
from .submission.pr_manager import PullRequestManager
from .utils.logger import setup_logger


REASONING_EFFORT_OPTIONS = [
    ("Backend / Model default", "default"),
    ("None (no reasoning)", "none"),
    ("Minimal (fastest)", "minimal"),
    ("Low (efficient)", "low"),
    ("Medium (balanced)", "medium"),
    ("High (deep reasoning)", "high"),
    ("XHigh (extended reasoning)", "xhigh"),
    ("Max (maximum reasoning)", "max"),
]
_REASONING_EFFORT_ALIASES = {
    "": "default",
    "unset": "default",
    "not-set": "default",
    "notset": "default",
    "null": "default",
    "server-default": "default",
    "model-default": "default",
    "off": "none",
    "disabled": "none",
    "disable": "none",
    "min": "minimal",
    "x-high": "xhigh",
    "extra-high": "xhigh",
    "minimum": "minimal",
    "maximum": "max",
}
_VALID_REASONING_EFFORTS = {value for _, value in REASONING_EFFORT_OPTIONS}


def _canonical_reasoning_effort(value: Optional[str]) -> str:
    """Normalize common reasoning-effort aliases to TUI select values."""
    normalized = (value or "default").strip().lower().replace("_", "-").replace(" ", "-")
    normalized = "-".join(normalized.split("-"))
    normalized = _REASONING_EFFORT_ALIASES.get(normalized, normalized.replace("-", ""))
    if normalized not in _VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Unsupported reasoning effort '{value}'. Supported values: "
            f"{', '.join(sorted(_VALID_REASONING_EFFORTS))}"
        )
    return normalized


def _reasoning_effort_to_api(value: str) -> Optional[str]:
    """Return the explicit OpenAI-compatible effort value, or None to omit it."""
    return None if value == "default" else value

OFFICIAL_TASK_OPTIONS = [
    ("MMLU-Pro (Reasoning)", "mmlu_pro"),
    ("GPQA Diamond (Reasoning)", "gpqa_diamond"),
    ("BIG-Bench Hard (Reasoning)", "bbh"),
    ("GSM8K (Mathematics)", "gsm8k"),
    ("MATH-500 (Mathematics)", "math500"),
    ("IFEval (Instruction)", "ifeval"),
    ("EvalPlus (Coding)", "evalplus"),
    ("LiveCodeBench (Coding)", "livecodebench"),
    ("BFCL v4 (Tool Calling)", "bfcl"),
    ("RULER (Long Context)", "ruler"),
    ("XSTest (Safety)", "xstest"),
]


def _profile_label(profile: str) -> str:
    total = PROFILE_TOTAL_SAMPLES.get(profile)
    if total is None:
        count = f"{len(EVALUATION_PROFILES.get(profile, []))} tasks"
        mode = "FULL"
    else:
        count = f"{total:,} samples"
        mode = "PARTIAL"
    return f"{profile.capitalize()} ({count} - {mode})"


def _default_results_dir() -> str:
    if Path("website/data/results").exists():
        return "website/data/results"
    if Path("Website/data/results").exists():
        return "Website/data/results"
    return "data/results"



MODEL_PRESETS = [
    ("Custom / Manual Entry", "custom"),
    ("Llama 3.2 1B Instruct (Meta)", "meta-llama/Llama-3.2-1B-Instruct"),
    ("Llama 3.2 3B Instruct (Meta)", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Qwen 2.5 1.5B Instruct (Alibaba)", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen 2.5 3B Instruct (Alibaba)", "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen 2.5 7B Instruct (Alibaba)", "Qwen/Qwen2.5-7B-Instruct"),
    ("Phi-3.5 Mini Instruct (Microsoft)", "microsoft/Phi-3.5-mini-instruct"),
    ("Gemma 2 2B IT (Google)", "google/gemma-2-2b-it"),
    ("Spark X2.6 4B INT4 (Local)", "XHToken/Spark-X2.6-4B-INT4"),
]


TUI_CSS = """
Screen {
    background: #0d1117;
    color: #e6edf3;
}

#layout {
    height: 100%;
}

#sidebar {
    width: 32%;
    min-width: 38;
    max-width: 48;
    height: 100%;
    background: #161b22;
    border-right: solid #30363d;
    padding: 1;
    overflow-y: scroll;
}

#main-content {
    height: 100%;
    padding: 1;
}

.box-title {
    color: #58a6ff;
    text-style: bold;
    margin-bottom: 1;
}

.section-label {
    color: #8b949e;
    margin-top: 1;
    text-style: bold;
}

.specs-card {
    background: #21262d;
    border: solid #30363d;
    padding: 1;
    margin-bottom: 1;
    color: #c9d1d9;
}

.status-card {
    background: #161b22;
    border: solid #30363d;
    padding: 1;
    margin-bottom: 1;
    height: auto;
}

#btn-run {
    width: 100%;
    margin-top: 1;
    background: #238636;
    color: white;
    text-style: bold;
}

#btn-run:hover {
    background: #2ea043;
}

#btn-stop {
    width: 100%;
    margin-top: 1;
    background: #da3633;
    color: white;
}

#btn-stop:hover {
    background: #f85149;
}

#btn-test-conn {
    width: 100%;
    margin-top: 1;
    margin-bottom: 1;
    background: #1f6feb;
    color: white;
}

#btn-test-conn:hover {
    background: #388bfd;
}

#btn-submit {
    width: 100%;
    margin-top: 1;
    background: #8957e5;
    color: white;
    text-style: bold;
}

#btn-submit:hover {
    background: #a371f7;
}

#btn-submit:disabled {
    background: #30363d;
    color: #8b949e;
}

#log-container {
    height: 1fr;
    border: solid #30363d;
    background: #0a0c10;
    margin-bottom: 1;
}

#live-stream-box {
    dock: top;
    height: auto;
    max-height: 8;
    background: #161b22;
    border-bottom: solid #30363d;
    padding: 1;
    color: #7ee787;
    overflow-y: scroll;
}

#stream-log {
    height: 1fr;
    background: #0a0c10;
    color: #e6edf3;
    overflow-y: scroll;
    padding: 0 1;
}

#progress-bar {
    margin-top: 1;
    margin-bottom: 1;
}

#results-table-container {
    height: 12;
    border: solid #30363d;
    background: #161b22;
}

.task-checkbox {
    margin-top: 0;
    margin-bottom: 0;
}

#custom-tasks-container {
    display: none;
    background: #161b22;
    border: solid #30363d;
    padding: 1;
    margin-top: 1;
    margin-bottom: 1;
}

#custom-tasks-title {
    color: #58a6ff;
    text-style: bold;
    margin-bottom: 1;
}

/* Modal styling */
.modal-container {
    width: 60;
    height: auto;
    background: #161b22;
    border: thick #58a6ff;
    padding: 2;
}

.modal-title {
    text-style: bold;
    color: #58a6ff;
    margin-bottom: 1;
    text-align: center;
}

.modal-actions {
    margin-top: 1;
    height: auto;
    align: right middle;
}

.modal-actions Button {
    margin-left: 1;
}
"""


class HelpModalScreen(ModalScreen):
    """Modal dialog displaying keyboard shortcuts and benchmark details."""

    def compose(self) -> ComposeResult:
        with Container(classes="modal-container"):
            yield Label("ℹ️ SLM-Bench Studio Help & Shortcuts", classes="modal-title")
            help_text = (
                "[bold cyan]Keyboard Shortcuts:[/bold cyan]\n"
                "  • [bold yellow]r[/bold yellow] : Start benchmark evaluation\n"
                "  • [bold yellow]s[/bold yellow] : Stop / cancel ongoing evaluation\n"
                "  • [bold yellow]b[/bold yellow] : Toggle left configuration sidebar\n"
                "  • [bold yellow]c[/bold yellow] : Clear generation stream log\n"
                "  • [bold yellow]?[/bold yellow] or [bold yellow]h[/bold yellow] : Show this help screen\n"
                "  • [bold yellow]q[/bold yellow] : Quit application\n\n"
                "[bold cyan]Evaluation Profiles:[/bold cyan]\n"
                "  • [bold yellow]Quick[/bold yellow]: 5 sampled tasks (MMLU-Pro, GPQA Diamond, GSM8K, MATH-500, IFEval). Always PARTIAL and leaderboard-ineligible.\n"
                "  • [bold yellow]Standard[/bold yellow] (Default): 6 full official datasets (MMLU-Pro, GPQA Diamond, BBH, GSM8K, MATH-500, IFEval)\n"
                "  • [bold yellow]Extended[/bold yellow]: 10 full datasets (Standard plus EvalPlus, BFCL v4, RULER, XSTest)\n"
                "  • [bold yellow]Custom[/bold yellow]: choose official task IDs and optionally set a seed-backed sample limit\n\n"
                "[bold cyan]Reasoning Effort:[/bold cyan]\n"
                "  • [bold yellow]Backend / Model default[/bold yellow] omits the effort field.\n"
                "  • Common levels: none, minimal, low, medium, high, xhigh, max.\n"
                "  • Level support is model-dependent; use your server's model-specific documentation.\n\n"
                "[bold cyan]Scoring Tracks:[/bold cyan]\n"
                "  • [bold green]General Intelligence (SLM-Index)[/bold green]: Standard uses Reasoning 35%, Mathematics 35%, Instruction 30%. Extended uses Reasoning 25%, Mathematics 25%, Instruction 20%, Coding 15%, Tool Calling 15%.\n"
                "  • [bold green]Coding & Agentic Index[/bold green]: 60% Code synthesis + 40% Tool execution\n"
                "  • [bold green]Local Efficiency[/bold green]: Score-per-GB RAM footprint + generation tokens/sec\n"
                "  • [bold green]Safety[/bold green]: Reported separately and never counted in SLM-Index\n\n"
                "[dim]Only FULL results are eligible for the official leaderboard. Results are saved to the Output Directory and must pass schema and eligibility checks before PR submission.[/dim]"
            )
            yield Static(help_text)
            with Horizontal(classes="modal-actions"):
                yield Button("Close", variant="primary", id="help-btn-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-btn-close":
            self.dismiss(None)


class SubmitModalScreen(ModalScreen):
    """Modal dialog for submitting benchmark results via GitHub Pull Request."""

    def __init__(self, result_path: str, model_slug: str, slm_index: float, **kwargs):
        super().__init__(**kwargs)
        self.result_path = result_path
        self.model_slug = model_slug
        self.slm_index = slm_index

    def compose(self) -> ComposeResult:
        with Container(classes="modal-container"):
            yield Label(f"🚀 Submit Benchmark: {self.model_slug}", classes="modal-title")
            yield Label(f"Composite SLM-Index: [bold yellow]{self.slm_index:.2f}%[/bold yellow]\n")

            yield Label("Target GitHub Repository (owner/repo):", classes="section-label")
            yield Input(value="SeanDolan0/llm-auto-benchmark", id="modal-input-repo")

            yield Label("GitHub Personal Access Token (PAT):", classes="section-label")
            yield Input(value=os.environ.get("GITHUB_TOKEN", ""), placeholder="ghp_...", password=True, id="modal-input-token")

            yield Checkbox("Dry Run (Validate without pushing branch or PR)", value=False, id="modal-chk-dryrun")

            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", variant="default", id="modal-btn-cancel")
                yield Button("Submit PR", variant="success", id="modal-btn-submit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "modal-btn-cancel":
            self.dismiss(None)
        elif event.button.id == "modal-btn-submit":
            repo = self.query_one("#modal-input-repo", Input).value.strip()
            token = self.query_one("#modal-input-token", Input).value.strip() or None
            dry_run = self.query_one("#modal-chk-dryrun", Checkbox).value
            self.dismiss({
                "repo": repo,
                "token": token,
                "dry_run": dry_run,
                "result_path": self.result_path,
                "model_slug": self.model_slug,
                "slm_index": self.slm_index,
            })


class SLMBenchTUI(App):
    """Modern interactive TUI application for SLM-Bench."""

    TITLE = "SLM-Bench Studio"
    SUB_TITLE = "Automated Evaluation & Live Model Verification"
    CSS = TUI_CSS

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_eval", "Run"),
        ("s", "stop_eval", "Stop"),
        ("b", "toggle_sidebar", "Toggle Sidebar"),
        ("c", "clear_log", "Clear Log"),
        ("?", "show_help", "Help"),
        ("h", "show_help", "Help"),
    ]

    def __init__(
        self,
        default_model: str = "XHToken/Spark-X2.6-4B-INT4",
        default_backend: str = "openai",
        default_endpoint: str = "http://127.0.0.1:8000/v1",
        default_api_key: Optional[str] = None,
        default_effort: str = "low",
        default_quant: str = "INT4",
        default_output_dir: Optional[str] = None,
        default_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_model = default_model
        self.default_backend = default_backend
        self.default_endpoint = default_endpoint
        self.default_api_key = default_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_effort = _canonical_reasoning_effort(default_effort)
        self.default_quant = default_quant
        self.default_output_dir = default_output_dir or _default_results_dir()
        self.default_seed = default_seed

        # Detected Hardware
        self.hw = get_hardware_info()
        self.eval_in_progress = False
        self.cancel_event = threading.Event()
        self.current_live_response = ""

        # Last completed evaluation result and persistence file
        self.last_result = None
        self.last_result_path = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="layout"):
            # Left Sidebar: Configuration Controls
            with ScrollableContainer(id="sidebar"):
                yield Label("⚡ Host System Hardware", classes="box-title")

                # Hardware Info Card
                cpu_short = self.hw.cpu.replace("Intel(R) Core(TM) ", "").replace("Processor", "").strip()
                gpu_short = (self.hw.gpu_name or "CPU Mode").replace("NVIDIA GeForce ", "")
                vram_str = f"{self.hw.gpu_vram_gb:.1f} GB VRAM" if self.hw.gpu_vram_gb else "No dedicated VRAM"
                os_str = "Windows 11 (WSL2)" if is_endpoint_wsl_relayed(self.default_endpoint) or is_wsl_environment() else self.hw.os.split("(")[0].strip()

                specs_text = (
                    f"[bold cyan]GPU:[/bold cyan] {gpu_short} ({vram_str})\n"
                    f"[bold cyan]CPU:[/bold cyan] {cpu_short}\n"
                    f"[bold cyan]RAM:[/bold cyan] {self.hw.total_ram_gb:.1f} GB | [bold cyan]OS:[/bold cyan] {os_str}"
                )
                yield Static(specs_text, classes="specs-card")

                yield Label("Model Presets:", classes="section-label")
                yield Select(
                    MODEL_PRESETS,
                    value=self.default_model if any(p[1] == self.default_model for p in MODEL_PRESETS) else "custom",
                    id="select-preset",
                )

                yield Label("Model ID / Hugging Face Repo:", classes="section-label")
                yield Input(value=self.default_model, placeholder="e.g. meta-llama/Llama-3.2-3B", id="input-model")

                yield Label("Quantization:", classes="section-label")
                yield Input(value=self.default_quant, placeholder="e.g. INT4, BF16, AWQ_4BIT", id="input-quant")

                yield Label("Model RAM / VRAM (GB, optional):", classes="section-label")
                yield Input(value="", placeholder="Leave blank to auto-detect or estimate", id="input-model-ram")

                yield Label("Inference Backend:", classes="section-label")
                yield Select(
                    [
                        ("OpenAI-Compatible (vLLM / Ollama)", "openai"),
                        ("Mock Engine (Test / Simulation)", "mock"),
                        ("HuggingFace Transformers", "transformers"),
                        ("Llama.cpp", "llamacpp"),
                    ],
                    value=self.default_backend,
                    id="select-backend",
                )

                yield Label("Server Endpoint URL:", classes="section-label")
                yield Input(value=self.default_endpoint, id="input-endpoint")
                yield Button("🔌 Test Connection", id="btn-test-conn")

                yield Label("API Key:", classes="section-label")
                yield Input(value=self.default_api_key, placeholder="(Optional for local backends)", password=True, id="input-apikey")

                yield Label("Reasoning Effort (level support is model-dependent):", classes="section-label")
                yield Select(
                    REASONING_EFFORT_OPTIONS,
                    value=self.default_effort,
                    id="select-effort",
                )

                yield Label("Evaluation Profile:", classes="section-label")
                yield Select(
                    [
                        (_profile_label("quick"), "quick"),
                        (_profile_label("small"), "small"),
                        (_profile_label("medium"), "medium"),
                        (_profile_label("standard"), "standard"),
                        (_profile_label("extended"), "extended"),
                        ("Custom (Choose Official Tasks)", "custom"),
                    ],
                    value="standard",
                    id="select-profile",
                )

                yield Label("Sample Limit:", classes="section-label")
                yield Select(
                    [
                        ("Profile Default", "default"),
                        ("1 sample (Quick test)", "1"),
                        ("3 samples (Fast check)", "3"),
                        ("5 samples (Standard)", "5"),
                        ("All samples (Full suite)", "all"),
                    ],
                    value="default",
                    id="select-limit",
                )

                yield Label("Sampling Seed:", classes="section-label")
                yield Input(value=str(self.default_seed), placeholder="42", id="input-seed")

                yield Label("Output Directory:", classes="section-label")
                yield Input(value=self.default_output_dir, placeholder="data/results", id="input-output-dir")

                yield Label("Evaluation Notes:", classes="section-label")
                yield Input(value="", placeholder="Optional notes (overrides auto-generated notes)", id="input-notes")

                with Container(id="custom-tasks-container"):
                    yield Label("Official Custom Tasks:", id="custom-tasks-title")
                    for label, task_id in OFFICIAL_TASK_OPTIONS:
                        yield Checkbox(label, value=task_id in EVALUATION_PROFILES["standard"], id=f"chk-task-{task_id}", classes="task-checkbox")

                yield Button("▶  Start Evaluation", variant="success", id="btn-run")
                yield Button("⏹  Stop", variant="error", id="btn-stop", disabled=True)

            # Right Main Panel: Live Telemetry, Streaming Output, and Results
            with Vertical(id="main-content"):
                with Container(classes="status-card"):
                    yield Label("[bold cyan]Status:[/bold cyan] Ready to benchmark", id="lbl-status")
                    yield ProgressBar(total=100, show_eta=False, id="progress-bar")

                # Live Streaming Terminal & Clean Output Log
                yield Label("📺 Live LLM Generation & Verification Stream", classes="box-title")
                with Container(id="log-container"):
                    yield Static("[dim italic]Model output will stream live here during evaluation...[/dim italic]", id="live-stream-box")
                    yield RichLog(id="stream-log", wrap=True, highlight=False, markup=True)

                # Results Summary Table
                yield Label("📊 Benchmark Scoreboard", classes="box-title")
                with Container(id="results-table-container"):
                    table = DataTable(id="results-table", zebra_stripes=True)
                    table.add_columns("Category", "Task ID", "Score", "Accuracy / Pass@1", "Latency (ms)", "Tokens/s")
                    yield table

                yield Button("🚀  Submit Results to Leaderboard (PR)", variant="primary", id="btn-submit", disabled=True)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize UI widgets upon mount."""
        self._set_custom_tasks_visibility(str(self.query_one("#select-profile", Select).value) == "custom")
        log = self.query_one("#stream-log", RichLog)
        log.write("[bold green]Welcome to SLM-Bench Interactive Studio![/bold green]")
        log.write("[dim]Configure your model and server on the left, then click 'Start Evaluation' (or press 'r').[/dim]")
        log.write("[dim]Press '?' or 'h' for shortcuts and help.[/dim]")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update model input and custom task panel visibility."""
        if event.select.id == "select-preset":
            val = str(event.value)
            if val != "custom":
                self.query_one("#input-model", Input).value = val
        elif event.select.id == "select-profile":
            self._set_custom_tasks_visibility(str(event.value) == "custom")

    def _set_custom_tasks_visibility(self, visible: bool) -> None:
        """Show or hide the official custom task checkbox panel."""
        self.query_one("#custom-tasks-container").display = visible

    def action_run_eval(self) -> None:
        """Trigger evaluation action."""
        if not self.eval_in_progress:
            self.start_evaluation()

    def action_stop_eval(self) -> None:
        """Trigger stop evaluation action."""
        if self.eval_in_progress:
            self.stop_evaluation()

    def action_toggle_sidebar(self) -> None:
        """Toggle left configuration sidebar visibility."""
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_clear_log(self) -> None:
        """Clear streaming output log and live preview."""
        log = self.query_one("#stream-log", RichLog)
        log.clear()
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[dim italic]Model output will stream live here during evaluation...[/dim italic]")

    def action_show_help(self) -> None:
        """Open the help and shortcuts modal."""
        self.push_screen(HelpModalScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn-run":
            self.start_evaluation()
        elif event.button.id == "btn-stop":
            self.stop_evaluation()
        elif event.button.id == "btn-test-conn":
            self.test_connection()
        elif event.button.id == "btn-submit":
            self.open_submit_modal()

    def test_connection(self) -> None:
        """Initiate endpoint connectivity pre-flight test."""
        endpoint_url = self.query_one("#input-endpoint", Input).value.strip() or self.default_endpoint
        api_key = self.query_one("#input-apikey", Input).value.strip()
        self.test_connection_worker(endpoint_url, api_key)

    @work(thread=True)
    def test_connection_worker(self, endpoint_url: str, api_key: str) -> None:
        """Pre-flight check to verify endpoint connectivity."""
        self.call_from_thread(self._update_status, "[bold yellow]Testing endpoint connection...[/bold yellow]")
        try:
            url = endpoint_url.rstrip("/")
            if not url.startswith("http"):
                url = f"http://{url}"

            test_url = f"{url}/models"
            req = urllib.request.Request(test_url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")

            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                latency_ms = (time.perf_counter() - t0) * 1000
                status_code = resp.status

            self.call_from_thread(
                self._update_status,
                f"[bold green]✓ Endpoint reachable[/bold green] (HTTP {status_code}, {latency_ms:.0f}ms)",
            )
            self.call_from_thread(
                self._log_info,
                f"[bold green]✓ Connection Test Passed:[/bold green] Reachable at {test_url} in {latency_ms:.1f}ms",
            )
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                msg = f"[bold yellow]⚠ Endpoint reachable (HTTP {e.code}: Authentication required)[/bold yellow]"
            elif e.code == 404:
                msg = "[bold green]✓ Server reachable[/bold green] (HTTP 404 on /models endpoint)"
            else:
                msg = f"[bold yellow]⚠ Server responded with HTTP {e.code}[/bold yellow]"
            self.call_from_thread(self._update_status, msg)
            self.call_from_thread(self._log_info, f"{msg} at {endpoint_url}")
        except Exception as e:
            err_msg = f"[bold red]✗ Connection failed: {e}[/bold red]"
            self.call_from_thread(self._update_status, err_msg)
            self.call_from_thread(
                self._log_info,
                f"[bold red]✗ Pre-flight check failed:[/bold red] Could not connect to {endpoint_url}. Verify that your local server (vLLM / Ollama) is active.",
            )

    def open_submit_modal(self) -> None:
        """Open the GitHub Pull Request submission modal."""
        if not self.last_result or not self.last_result_path:
            return
        if not self.last_result.eligible_for_leaderboard:
            self._update_status(
                f"[bold yellow]Result status is {self.last_result.status}; only FULL eligible results can be submitted to the official leaderboard.[/bold yellow]"
            )
            self._log_info(
                f"[bold yellow]Submission blocked:[/bold yellow] Status {self.last_result.status} is not leaderboard-eligible."
            )
            return

        def handle_submission(submission_data: Optional[Dict[str, Any]]) -> None:
            if submission_data:
                self.submit_pr_worker(submission_data)

        self.push_screen(
            SubmitModalScreen(
                result_path=self.last_result_path,
                model_slug=self.last_result.model.model_slug,
                slm_index=self.last_result.slm_index,
            ),
            handle_submission,
        )

    @work(thread=True)
    def submit_pr_worker(self, submission_data: Dict[str, Any]) -> None:
        """Execute Git commit and GitHub Pull Request creation in background."""
        repo = submission_data["repo"]
        token = submission_data.get("token")
        dry_run = submission_data.get("dry_run", False)
        res_path = Path(submission_data["result_path"])

        self.call_from_thread(self._update_status, "[bold yellow]Submitting results to GitHub...[/bold yellow]")
        try:
            json_content = res_path.read_text(encoding="utf-8")
            result = BenchmarkResult.model_validate_json(json_content)
            if not result.eligible_for_leaderboard:
                raise ValueError(
                    f"Result status is {result.status}; only FULL eligible results can be published to the official leaderboard."
                )
            model_slug = result.model.model_slug
            repo_url = f"https://github.com/{repo}.git" if not repo.startswith("http") else repo

            git_mgr = GitSubmissionManager(repo_url=repo_url, token=token, dry_run=dry_run)
            git_mgr.prepare_repo()
            branch_name = git_mgr.commit_and_push(
                model_slug=model_slug,
                json_content=json_content,
                slm_index=result.slm_index,
            )

            pr_mgr = PullRequestManager(repo_identifier=repo, token=token, dry_run=dry_run)
            pr_info = pr_mgr.create_pull_request(branch_name=branch_name, result=result)
            pr_url = pr_info.get("pr_url")

            self.call_from_thread(self._on_submit_finished, pr_url, dry_run, branch_name)
        except Exception as e:
            self.call_from_thread(self._on_submit_error, str(e))

    def _on_submit_finished(self, pr_url: Optional[str], dry_run: bool, branch_name: str) -> None:
        """Handle successful PR submission."""
        lbl = self.query_one("#lbl-status", Label)
        log = self.query_one("#stream-log", RichLog)
        if dry_run:
            lbl.update(f"[bold green]✓ Dry-run completed![/bold green] Branch: [cyan]{branch_name}[/cyan]")
            log.write(f"[bold green]✓ Dry-run submission completed successfully![/bold green]")
            log.write(f"Validated JSON schema and git branch: [bold cyan]{branch_name}[/bold cyan]")
            if pr_url:
                log.write(f"Simulated Pull Request: [bold cyan]{pr_url}[/bold cyan]")
        else:
            lbl.update(f"[bold green]✓ Pull Request Created![/bold green] {pr_url}")
            log.write(f"[bold green]🚀 PULL REQUEST OPENED SUCCESSFULLY![/bold green]")
            if pr_url:
                log.write(f"Leaderboard PR URL: [bold cyan]{pr_url}[/bold cyan]")
        self.query_one("#btn-submit", Button).disabled = True

    def _on_submit_error(self, err_msg: str) -> None:
        """Handle submission failure."""
        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold red]Submission failed: {err_msg}[/bold red]")
        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold red]✗ PR Submission Error:[/bold red] {err_msg}")

    def _form_error(self, message: str) -> None:
        """Surface a validation error without starting an evaluation."""
        self.query_one("#lbl-status", Label).update(f"[bold red]{message}[/bold red]")
        self.query_one("#live-stream-box", Static).update(f"[bold red]✖ {message}[/bold red]")
        self.query_one("#stream-log", RichLog).write(f"[bold red]✖ Input error:[/bold red] {message}")

    def start_evaluation(self) -> None:
        """Start benchmark run in background worker thread."""
        if self.eval_in_progress:
            return

        # Read and validate form configuration before changing UI state.
        model_id = self.query_one("#input-model", Input).value.strip() or self.default_model
        quantization = self.query_one("#input-quant", Input).value.strip() or self.default_quant
        backend_name = str(self.query_one("#select-backend", Select).value)
        endpoint_url = self.query_one("#input-endpoint", Input).value.strip() or self.default_endpoint
        api_key = self.query_one("#input-apikey", Input).value.strip() or self.default_api_key
        effort_val = str(self.query_one("#select-effort", Select).value)
        effort = _reasoning_effort_to_api(effort_val)

        ram_text = self.query_one("#input-model-ram", Input).value.strip()
        model_ram_gb = None
        if ram_text:
            try:
                model_ram_gb = float(ram_text)
                if model_ram_gb <= 0:
                    raise ValueError
            except ValueError:
                self._form_error("Model RAM / VRAM must be a positive number of GB.")
                return

        try:
            seed_text = self.query_one("#input-seed", Input).value.strip()
            seed = int(seed_text) if seed_text else self.default_seed
        except ValueError:
            self._form_error("Sampling seed must be a whole number.")
            return

        output_dir = self.query_one("#input-output-dir", Input).value.strip() or self.default_output_dir
        notes = self.query_one("#input-notes", Input).value.strip() or None

        limit_val = str(self.query_one("#select-limit", Select).value)
        sample_limit = None if limit_val in ("all", "default") else int(limit_val)

        profile_val = str(self.query_one("#select-profile", Select).value)
        profile = None if profile_val == "custom" else profile_val

        selected_tasks = None
        if profile_val == "custom":
            selected_tasks = [
                task_id
                for _, task_id in OFFICIAL_TASK_OPTIONS
                if self.query_one(f"#chk-task-{task_id}", Checkbox).value
            ]
            if not selected_tasks:
                self._form_error("Select at least one official benchmark for a custom evaluation.")
                return

        self.eval_in_progress = True
        self.cancel_event.clear()
        self.current_live_response = ""

        self.query_one("#btn-run", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False
        self.query_one("#btn-submit", Button).disabled = True

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[bold yellow]⚡ Initializing model and inference backend...[/bold yellow]")

        # Clear scoreboard for a fresh run
        table = self.query_one("#results-table", DataTable)
        table.clear()

        status_lbl = self.query_one("#lbl-status", Label)
        status_lbl.update("[bold yellow]Initializing benchmark backend and model metadata...[/bold yellow]")

        # Run worker
        self.run_eval_worker(
            model_id=model_id,
            quantization=quantization,
            backend_name=backend_name,
            endpoint_url=endpoint_url,
            api_key=api_key,
            reasoning_effort=effort,
            model_ram_gb=model_ram_gb,
            output_dir=output_dir,
            seed=seed,
            notes=notes,
            profile=profile,
            sample_limit=sample_limit,
            selected_tasks=selected_tasks,
        )

    def stop_evaluation(self) -> None:
        """Signal cooperative cancellation to running benchmark worker."""
        if not self.eval_in_progress:
            return

        self.cancel_event.set()
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#lbl-status", Label).update("[bold red]Stopping evaluation...[/bold red]")
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[bold red]⏹ Cancellation requested. Waiting for active operation to stop...[/bold red]")
        log = self.query_one("#stream-log", RichLog)
        log.write("[bold yellow]Cancellation signal sent. Halting tasks...[/bold yellow]")

    @work(thread=True)
    def run_eval_worker(
        self,
        model_id: str,
        quantization: str,
        backend_name: str,
        endpoint_url: str,
        api_key: str,
        reasoning_effort: Optional[str],
        model_ram_gb: Optional[float],
        output_dir: str,
        seed: int,
        notes: Optional[str],
        profile: Optional[str],
        sample_limit: Optional[int],
        selected_tasks: Optional[List[str]],
    ) -> None:
        """Background thread worker executing evaluation suite with cooperative cancellation."""
        try:
            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled before start.")

            # 1. Fetch metadata
            meta = fetch_hf_model_metadata(
                model_id,
                quantization=quantization,
                backend=backend_name,
                model_ram_gb=model_ram_gb,
            )
            if reasoning_effort:
                meta.reasoning_effort = reasoning_effort

            hw = get_hardware_info(model_ram_gb=meta.model_ram_gb)

            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled during hardware detection.")

            # 2. Instantiate backend
            backend = get_backend(
                backend_name,
                model_id=model_id,
                quantization=quantization,
                endpoint_url=endpoint_url,
                api_key=api_key,
                model_ram_gb=meta.model_ram_gb,
                reasoning_effort=reasoning_effort,
            )

            # Hook token stream callback to verify active LLM generation
            def on_token_chunk(token: str):
                self.call_from_thread(self._append_token_chunk, token)

            backend.set_stream_callback(on_token_chunk)
            backend.load()

            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled after backend loading.")

            # Auto-detect RAM only when the user did not provide a footprint
            endpoint_backends = ("openai", "openai_compatible", "openai-compatible", "ollama", "lmstudio", "endpoint", "localai")
            if model_ram_gb is None:
                detected_ram = auto_detect_model_ram_gb(
                    backend_instance=backend,
                    backend_name=backend_name,
                    model_id=model_id,
                    quantization=quantization,
                    param_count_b=meta.param_count_b,
                    endpoint_url=endpoint_url if backend_name.lower() in endpoint_backends else None,
                )
                if detected_ram:
                    meta.model_ram_gb = detected_ram
                    hw.model_ram_gb = detected_ram

            # Generate auto notes
            notes = generate_auto_system_notes(
                hw=hw,
                backend_name=backend_name,
                endpoint_url=endpoint_url,
                server_runtime=getattr(backend, "server_runtime", None),
                user_notes=notes,
            )

            # 3. Setup hooks for suite execution
            suite = BenchmarkSuite()
            resolved_tasks = suite.resolve_tasks(selected_tasks=selected_tasks, profile=profile)
            if not resolved_tasks:
                raise ValueError("No recognized benchmark tasks were resolved. Check your custom task selection.")
            total_tasks_count = len(resolved_tasks)
            current_task_idx = [0]

            def on_task_start(task_id: str, category: str):
                current_task_idx[0] += 1
                msg = f"Task [{current_task_idx[0]}/{total_tasks_count}]: [bold cyan]{task_id}[/bold cyan] ({category.title()})"
                pct = int((current_task_idx[0] - 1) / max(1, total_tasks_count) * 100)
                self.call_from_thread(self._update_task_progress, msg, pct)
                self.call_from_thread(self._log_task_header, task_id, category)

            def on_sample(info: Dict[str, Any]):
                self.call_from_thread(self._log_sample_result, info)

            def on_task_complete(task_id: str, metric: Any):
                self.call_from_thread(self._add_metric_row, task_id, metric)

            result = suite.run_suite(
                backend=backend,
                model_meta=meta,
                hardware_info=hw,
                selected_tasks=selected_tasks,
                profile=profile,
                sample_limit=sample_limit,
                notes=notes,
                on_task_start=on_task_start,
                on_sample=on_sample,
                on_task_complete=on_task_complete,
                show_progress=False,
                cancel_event=self.cancel_event,
                seed=seed,
            )

            if self.cancel_event.is_set():
                raise InterruptedError("Evaluation cancelled during benchmark execution.")

            # 4. Save result to the selected output directory
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            res_file = out_dir / f"{meta.model_slug}.json"
            res_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

            # Final UI notification
            self.call_from_thread(self._on_eval_finished, result, str(res_file))

        except InterruptedError:
            self.call_from_thread(self._on_eval_cancelled)
        except Exception as e:
            if self.cancel_event.is_set():
                self.call_from_thread(self._on_eval_cancelled)
            else:
                self.call_from_thread(self._on_eval_error, str(e))

    def _update_task_progress(self, msg: str, pct: int) -> None:
        """Update progress bar and status label from worker thread."""
        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold green]Running:[/bold green] {msg}")
        pbar = self.query_one("#progress-bar", ProgressBar)
        pbar.progress = pct

    def _append_token_chunk(self, token: str) -> None:
        """Update live generation view in-place without artificial line breaks."""
        self.current_live_response += token
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold cyan]⚡ Live Generation Stream:[/bold cyan]\n{self.current_live_response}▌")

    def _log_task_header(self, task_id: str, category: str) -> None:
        """Log prominent task header banner."""
        self.current_live_response = ""
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold cyan]⚡ Evaluating Task:[/bold cyan] [bold yellow]{task_id}[/bold yellow] ([italic]{category.title()}[/italic])...")

        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold magenta]════════════════════════════════════════════════════════════════════════════[/bold magenta]")
        log.write(f"[bold cyan]▶ STARTING TASK:[/bold cyan] [bold yellow]{task_id}[/bold yellow] ([italic]{category.title()}[/italic])")
        log.write(f"[bold magenta]────────────────────────────────────────────────────────────────────────────[/bold magenta]")

    def _log_sample_result(self, info: Dict[str, Any]) -> None:
        """Log structured sample evaluation and clean AI output."""
        self.current_live_response = ""
        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[dim italic]Model output will stream live here during evaluation...[/dim italic]")

        log = self.query_one("#stream-log", RichLog)
        idx = info.get("sample_index", 1)
        tot = info.get("total_samples", 1)
        passed = info.get("passed", False)
        lat = info.get("latency_ms", 0.0)
        toks = info.get("tokens", 0)
        tps = (toks / (lat / 1000.0)) if lat > 0 and toks > 0 else 0.0
        task_id = info.get("task_id", "")
        category = info.get("category", "")
        prompt = info.get("prompt", "")
        response = info.get("response", "")
        expected = info.get("expected", "")
        extracted = info.get("extracted", "")

        verdict_tag = "[bold green]✓ PASSED[/bold green]" if passed else "[bold red]✗ FAILED[/bold red]"

        log.write(f"[bold cyan]▶ [{category.title()}] {task_id}[/bold cyan] [dim](Sample {idx}/{tot})[/dim] {verdict_tag} [dim]| {lat:.0f}ms | {tps:.1f} tok/s[/dim]")
        if prompt:
            p_clean = prompt.strip().replace("\r\n", "\n")
            log.write(f"[dim]Prompt:[/dim] {p_clean}")
        if response:
            log.write("[bold green]AI Output:[/bold green]")
            for line in response.strip().splitlines():
                cleaned = line.rstrip()
                if cleaned:
                    log.write(f"  {cleaned}")
                else:
                    log.write("")
        if expected is not None or extracted is not None:
            log.write(f"[dim]Expected: {expected} | Extracted: {extracted}[/dim]")
        log.write("[dim]────────────────────────────────────────────────────────────────────────────[/dim]")

    def _add_metric_row(self, task_id: str, metric: Any) -> None:
        """Add finished metric to the results table."""
        table = self.query_one("#results-table", DataTable)
        score_styled = f"[bold green]{metric.normalized_score:.1f}%[/bold green]" if metric.normalized_score >= 70 else f"[bold yellow]{metric.normalized_score:.1f}%[/bold yellow]"
        table.add_row(
            metric.category.title(),
            task_id,
            score_styled,
            f"{metric.passed_samples}/{metric.total_samples}",
            f"{metric.avg_latency_ms:.1f}" if metric.avg_latency_ms else "N/A",
            f"{metric.avg_tokens_per_sec:.1f}" if metric.avg_tokens_per_sec else "N/A",
        )

    def _update_status(self, text: str) -> None:
        """Update top status message."""
        lbl = self.query_one("#lbl-status", Label)
        lbl.update(text)

    def _log_info(self, text: str) -> None:
        """Write an informational message to the stream log."""
        log = self.query_one("#stream-log", RichLog)
        log.write(text)

    def _on_eval_finished(self, result: Any, file_path: str) -> None:
        """Update UI upon evaluation completion."""
        self.eval_in_progress = False
        self.last_result = result
        self.last_result_path = file_path

        status = getattr(result, "status", "UNKNOWN")
        eligible = bool(getattr(result, "eligible_for_leaderboard", False))
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#btn-submit", Button).disabled = not eligible

        pbar = self.query_one("#progress-bar", ProgressBar)
        pbar.progress = 100

        lbl = self.query_one("#lbl-status", Label)
        lbl.update(
            f"[bold green]✓ Complete![/bold green] Status: [bold]{status}[/bold] | "
            f"SLM-Index: [bold yellow]{result.slm_index:.2f}%[/bold yellow] | Saved: [cyan]{file_path}[/cyan]"
        )

        live_box = self.query_one("#live-stream-box", Static)
        eligibility_text = "eligible" if eligible else "not leaderboard-eligible"
        live_box.update(
            f"[bold {'green' if eligible else 'yellow'}]✓ Evaluation completed![/bold {'green' if eligible else 'yellow'}] "
            f"Status: [bold]{status}[/bold] | SLM-Index: [bold yellow]{result.slm_index:.2f}%[/bold yellow] ({eligibility_text})"
        )

        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold green]════════════════════════════════════════════════════════════════════════════[/bold green]")
        log.write(f"[bold green]✓ BENCHMARK COMPLETED SUCCESSFULLY![/bold green]")
        log.write(f"Evaluation Status: [bold yellow]{status}[/bold yellow] | Eligible for Leaderboard: [bold]{eligible}[/bold]")
        log.write(f"Evaluation Profile: [bold yellow]{(result.profile or 'standard').upper()}[/bold yellow]")
        log.write(f"General Intelligence (SLM-Index): [bold yellow]{result.slm_index:.2f}%[/bold yellow]")
        if result.coding_agentic_index is not None:
            log.write(f"Coding & Agentic Index: [bold cyan]{result.coding_agentic_index:.2f}%[/bold cyan]")
        if result.local_efficiency:
            eff = result.local_efficiency
            pts_gb = eff.get("score_per_gb", "N/A")
            tps = eff.get("tokens_per_sec", "N/A")
            log.write(f"Local Efficiency: [bold magenta]{pts_gb} pts/GB RAM | {tps} tok/s[/bold magenta]")
        if eligible:
            log.write("[bold green]Result is FULL and eligible for official leaderboard submission.[/bold green]")
        else:
            log.write(
                f"[bold yellow]Result status is {status}; official leaderboard submission is blocked.[/bold yellow]"
            )
        log.write(f"Standardized Result JSON: [bold cyan]{file_path}[/bold cyan]")
        log.write(f"[bold green]════════════════════════════════════════════════════════════════════════════[/bold green]")

    def _on_eval_cancelled(self) -> None:
        """Handle clean cooperative cancellation."""
        self.eval_in_progress = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#btn-submit", Button).disabled = not bool(
            self.last_result and getattr(self.last_result, "eligible_for_leaderboard", False)
        )

        lbl = self.query_one("#lbl-status", Label)
        lbl.update("[bold yellow]Evaluation cancelled by user.[/bold yellow]")

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update("[bold yellow]⚠ Evaluation was cancelled by user.[/bold yellow]")

        log = self.query_one("#stream-log", RichLog)
        log.write("[bold yellow]⚠ Evaluation was cancelled by user. Background tasks stopped cleanly.[/bold yellow]")

    def _on_eval_error(self, err_msg: str) -> None:
        """Handle evaluation error."""
        self.eval_in_progress = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#btn-submit", Button).disabled = not bool(
            self.last_result and getattr(self.last_result, "eligible_for_leaderboard", False)
        )

        lbl = self.query_one("#lbl-status", Label)
        lbl.update(f"[bold red]Error: {err_msg}[/bold red]")

        live_box = self.query_one("#live-stream-box", Static)
        live_box.update(f"[bold red]✖ Evaluation error: {err_msg}[/bold red]")

        log = self.query_one("#stream-log", RichLog)
        log.write(f"[bold red]✖ Evaluation Error Encountered:[/bold red] {err_msg}")


def launch_tui(**kwargs) -> None:
    """Launch the interactive SLM-Bench TUI application."""
    setup_logger()
    app = SLMBenchTUI(**kwargs)
    app.run()
