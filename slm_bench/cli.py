"""
CLI interface for SLM-Bench evaluation harness and submission system (Version 2.0).
Provides verifiable evaluation using established official benchmarks:
- MMLU-Pro, GPQA Diamond, BBH, GSM8K, MATH-500, IFEval, EvalPlus, BFCL, RULER, XSTest.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Optional, List
import typer
from rich.table import Table
from rich.panel import Panel

from .models.hardware import get_hardware_info, generate_auto_system_notes
from .models.metadata import fetch_hf_model_metadata
from .models.memory_detector import auto_detect_model_ram_gb
from .models.schema import BenchmarkResult, TaskMetric
from .backends.registry import get_backend
from .benchmarks.suite import BenchmarkSuite
from .benchmarks.registry import BENCHMARK_REGISTRY, EVALUATION_PROFILES, get_task, list_available_tasks, get_profile_sample_limits
from .datasets.manager import DatasetManager, BENCHMARK_DATASET_META
from .submission.git_manager import GitSubmissionManager
from .submission.pr_manager import PullRequestManager
from .utils.logger import console, logger, setup_logger

app = typer.Typer(
    name="slm-bench",
    help="Automated Evaluation Harness for Small Local AI Using Established Official Benchmarks.",
    add_completion=False,
)


@app.command()
def info(
    model_id: Optional[str] = typer.Option(None, "--model-id", "-m", help="Hugging Face model repository ID to inspect"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization tag"),
    model_ram_gb: Optional[float] = typer.Option(None, "--model-ram-gb", "--ram-gb", help="Size in RAM / VRAM in GB that the model takes up"),
):
    """Display host hardware specifications and inspect Hugging Face model metadata."""
    setup_logger()
    hw = get_hardware_info(model_ram_gb=model_ram_gb)

    hw_table = Table(title="[Host Hardware & Runtime Environment]", show_header=True, header_style="bold magenta")
    hw_table.add_column("Property", style="dim")
    hw_table.add_column("Value", style="bold cyan")
    hw_table.add_row("Operating System", hw.os)
    hw_table.add_row("CPU", f"{hw.cpu} ({hw.cpu_cores} logical cores)")
    hw_table.add_row("System RAM", f"{hw.total_ram_gb} GB")
    if hw.model_ram_gb:
        hw_table.add_row("Model RAM Footprint", f"{hw.model_ram_gb} GB")
    hw_table.add_row("GPU Model", hw.gpu_name or "None detected (CPU mode)")
    hw_table.add_row("GPU VRAM", f"{hw.gpu_vram_gb} GB" if hw.gpu_vram_gb else "N/A")
    hw_table.add_row("CUDA Version", hw.cuda_version or "N/A")
    hw_table.add_row("Driver Version", hw.driver_version or "N/A")
    console.print(hw_table)

    if model_id:
        meta = fetch_hf_model_metadata(model_id, quantization=quantization, model_ram_gb=model_ram_gb)
        m_table = Table(title=f"[Model Metadata: {model_id}]", show_header=True, header_style="bold green")
        m_table.add_column("Field", style="dim")
        m_table.add_column("Value", style="bold yellow")
        m_table.add_row("Model Slug", meta.model_slug)
        m_table.add_row("Parameter Count", f"{meta.param_count_b:.2f}B" if meta.param_count_b else "Unknown")
        m_table.add_row("Parameter Tier", meta.param_tier)
        m_table.add_row("RAM / VRAM Footprint", f"{meta.model_ram_gb:.2f} GB" if meta.model_ram_gb else "Unknown")
        m_table.add_row("Architecture", meta.architecture or "Unknown")
        m_table.add_row("Context Window", f"{meta.context_length} tokens" if meta.context_length else "Unknown")
        m_table.add_row("License", meta.license or "Unknown")
        console.print(m_table)


@app.command()
def evaluate(
    model_id: str = typer.Option(..., "--model-id", "--repo-id", "-m", help="Hugging Face repo ID or local model path"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization: BF16, FP16, AWQ_4BIT, GGUF_Q4_K_M, NF4"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Inference backend: mock, transformers, llamacpp, vllm, openai (Ollama / LM Studio)"),
    endpoint_url: str = typer.Option("http://localhost:11434/v1", "--endpoint-url", "--endpoint", "--base-url", help="OpenAI-compatible server endpoint URL"),
    api_key: str = typer.Option("EMPTY", "--api-key", envvar="OPENAI_API_KEY", help="API key for custom or remote endpoint"),
    model_ram_gb: Optional[float] = typer.Option(None, "--model-ram-gb", "--ram-gb", help="Size in RAM / VRAM in GB that the model takes up"),
    output_dir: str = typer.Option("data/results", "--output-dir", "-o", help="Target directory for output JSON"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Read/write a JSONL replay checkpoint to reuse prior model calls"),
    profile: Optional[str] = typer.Option("standard", "--profile", "-p", help="Evaluation profile: quick, small, medium, standard, extended"),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", help="Specific single benchmark to run (e.g. gsm8k, gpqa_diamond)"),
    tasks: Optional[str] = typer.Option(None, "--tasks", "-t", help="Comma-separated list of tasks to run (mmlu_pro, gpqa_diamond, gsm8k, math500, ifeval)"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", "-n", help="Deterministic sample count (marks run as PARTIAL)"),
    seed: int = typer.Option(42, "--seed", help="Reproducible random sampling seed"),
    submit: bool = typer.Option(False, "--submit", "-s", help="Automatically submit results via Pull Request after eval"),
    github_repo: Optional[str] = typer.Option(None, "--github-repo", help="Target GitHub repository (e.g. owner/repo)"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN", help="GitHub Personal Access Token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Optional notes on runner configuration or hardware offloading"),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort", "--effort", help="Reasoning effort level (model-dependent; none, minimal, low, medium, high, xhigh, max)"),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens", help="Override the per-question answer token budget"),
    thinking: str = typer.Option("auto", "--thinking", help="Thinking mode for reasoning models: auto, on, off"),
    truncation_policy: str = typer.Option("retry", "--truncation-policy", help="On a length-truncated answer: retry, invalid, accept"),
    max_token_budget: int = typer.Option(16384, "--max-token-budget", help="Maximum answer-token budget when retrying truncation (0 = 32768)"),
    min_token_budget: int = typer.Option(128, "--min-token-budget", help="Minimum adaptive starting budget"),
    adaptive_budget: bool = typer.Option(False, "--adaptive-budget/--no-adaptive-budget", help="Remember per-task token caps and start there on the next run"),
    budget_state: str = typer.Option("~/.cache/slm-bench/token_budgets.json", "--budget-state", help="Path for adaptive token-budget state"),
):
    """Run official benchmark evaluation suite on a single model and save standardized JSON (Schema 2.0.0)."""
    setup_logger()
    if isinstance(reasoning_effort, typer.models.OptionInfo):
        reasoning_effort = None
    if isinstance(max_new_tokens, typer.models.OptionInfo):
        max_new_tokens = None
    if isinstance(thinking, typer.models.OptionInfo):
        thinking = "auto"
    if isinstance(truncation_policy, typer.models.OptionInfo):
        truncation_policy = "retry"
    if isinstance(max_token_budget, typer.models.OptionInfo):
        max_token_budget = 16384
    if isinstance(min_token_budget, typer.models.OptionInfo):
        min_token_budget = 128
    if isinstance(adaptive_budget, typer.models.OptionInfo):
        adaptive_budget = False
    if isinstance(budget_state, typer.models.OptionInfo):
        budget_state = "~/.cache/slm-bench/token_budgets.json"
    if thinking not in ("auto", "on", "off"):
        raise typer.BadParameter("--thinking must be one of: auto, on, off")
    if truncation_policy not in ("retry", "invalid", "accept"):
        raise typer.BadParameter("--truncation-policy must be one of: retry, invalid, accept")
    if max_token_budget <= 0:
        max_token_budget = 32768
    min_token_budget = max(1, min_token_budget)
    if isinstance(profile, typer.models.OptionInfo) or not profile:
        profile = "standard"
    if isinstance(model_ram_gb, typer.models.OptionInfo):
        model_ram_gb = None
    if isinstance(checkpoint, typer.models.OptionInfo):
        checkpoint = None
    if isinstance(benchmark, typer.models.OptionInfo):
        benchmark = None
    if isinstance(tasks, typer.models.OptionInfo):
        tasks = None
    if isinstance(sample_limit, typer.models.OptionInfo):
        sample_limit = None
    if isinstance(seed, typer.models.OptionInfo) or seed is None:
        seed = 42
    if isinstance(notes, typer.models.OptionInfo):
        notes = None
    if isinstance(github_repo, typer.models.OptionInfo):
        github_repo = None
    if isinstance(token, typer.models.OptionInfo):
        token = None

    # 1. Hardware and Model Metadata
    meta = fetch_hf_model_metadata(model_id, quantization=quantization, backend=backend, model_ram_gb=model_ram_gb)
    if reasoning_effort:
        meta.reasoning_effort = reasoning_effort
    hw = get_hardware_info(model_ram_gb=meta.model_ram_gb)

    # 2. Model Telemetry Logging
    if meta.param_count_b:
        logger.info(f"Model '{model_id}' evaluated at {meta.param_count_b:.1f}B parameters.")
    if meta.model_ram_gb:
        logger.info(f"Model RAM / VRAM footprint: [bold cyan]{meta.model_ram_gb:.2f} GB[/bold cyan]")

    # 3. Instantiate Backend
    budget_manager = None
    if adaptive_budget:
        from .execution.budget import BudgetManager

        budget_manager = BudgetManager(
            path=budget_state,
            model_id=model_id,
            quantization=quantization,
            backend=backend,
            thinking=thinking,
            min_start=min_token_budget,
        )
    try:
        engine = get_backend(
            backend,
            model_id=model_id,
            quantization=quantization,
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_ram_gb=meta.model_ram_gb,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
            thinking=thinking,
            truncation_policy=truncation_policy,
            max_token_budget=max_token_budget,
            budget_manager=budget_manager,
        )
        engine.load()
    except Exception as e:
        logger.error(f"Failed to initialize backend '{backend}': {e}")
        raise typer.Exit(code=1)

    # 4. Auto-detect Model RAM / VRAM if not explicitly specified
    if model_ram_gb is None:
        detected_ram = auto_detect_model_ram_gb(
            backend_instance=engine,
            backend_name=backend,
            model_id=model_id,
            quantization=quantization,
            param_count_b=meta.param_count_b,
            endpoint_url=endpoint_url if backend in ("openai", "ollama", "lmstudio", "endpoint", "localai") else None,
        )
        if detected_ram:
            meta.model_ram_gb = detected_ram
            hw.model_ram_gb = detected_ram

    final_notes = generate_auto_system_notes(
        hw=hw,
        backend_name=backend,
        endpoint_url=endpoint_url,
        server_runtime=getattr(engine, "server_runtime", None),
        user_notes=notes,
    )

    # 5. Resolve Selected Tasks
    task_list = None
    if benchmark:
        task_list = [benchmark.strip()]
    elif tasks:
        task_list = [t.strip() for t in tasks.split(",")]

    # Display execution banner for each benchmark
    suite = BenchmarkSuite()
    resolved = suite.resolve_tasks(selected_tasks=task_list, profile=profile if not task_list else None)

    banner_table = Table(title="[Active Official Benchmark Evaluation Suite]", show_header=True, header_style="bold green")
    banner_table.add_column("Benchmark", style="bold yellow")
    banner_table.add_column("Official Dataset", style="cyan")
    banner_table.add_column("Revision", style="dim")
    banner_table.add_column("Split", style="dim")
    banner_table.add_column("Questions Count", style="bold")
    banner_table.add_column("Profile Mode", style="magenta")

    for tid, tobj in resolved.items():
        d_meta = BENCHMARK_DATASET_META.get(tobj.task_id, {})
        d_name = d_meta.get("name", tobj.task_id)
        d_rev = d_meta.get("revision", "official")
        d_split = d_meta.get("split", "test")
        profile_limits = get_profile_sample_limits(profile) if not task_list else {}
        task_limit = sample_limit if sample_limit is not None else (profile_limits or {}).get(tobj.task_id)
        q_count = str(task_limit or d_meta.get("expected_count", "Full"))
        p_mode = "PARTIAL (Sampled)" if task_limit else "FULL (Official Split)"
        banner_table.add_row(d_name, d_meta.get("source", "Official upstream"), d_rev, d_split, q_count, p_mode)
    console.print(banner_table)

    # 6. Run Benchmark Suite
    result = suite.run_suite(
        backend=engine,
        model_meta=meta,
        hardware_info=hw,
        selected_tasks=task_list,
        profile=profile if not task_list else None,
        sample_limit=sample_limit,
        notes=final_notes,
        seed=seed,
        checkpoint_path=checkpoint,
        generation_policy=(
            f"thinking={thinking};truncation={truncation_policy};"
            f"min={min_token_budget};cap={max_token_budget};adaptive={adaptive_budget}"
        ),
    )

    # 7. Save JSON Output
    out_path = Path(output_dir)
    if output_dir == "data/results":
        if Path("website/data/results").exists():
            out_path = Path("website/data/results")
        elif Path("Website/data/results").exists():
            out_path = Path("Website/data/results")
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"{meta.model_slug}.json"
    json_str = result.model_dump_json(indent=2)
    result_file.write_text(json_str, encoding="utf-8")
    logger.info(f"Standardized result successfully saved to: [bold cyan]{result_file}[/bold cyan]")

    # 8. Render Result Summary Table
    summary_table = Table(
        title=f"[Benchmark Summary: {meta.model_slug} (Status: {result.status})]",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Benchmark Task", style="bold")
    summary_table.add_column("Category", style="dim")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Passed / Total")
    summary_table.add_column("Accuracy / Metric", style="bold yellow")
    summary_table.add_column("95% Wilson CI", style="dim")

    for tid, tm in result.tasks.items():
        ci_str = f"[{tm.ci_lower:.1f}%, {tm.ci_upper:.1f}%]" if tm.ci_lower is not None else "N/A"
        summary_table.add_row(
            tid,
            tm.category,
            tm.status,
            f"{tm.passed_samples}/{tm.total_samples}",
            f"{tm.normalized_score:.2f}%",
            ci_str,
        )
    summary_table.add_row(
        "[bold]Composite SLM-Index[/bold]",
        "Overall",
        result.status,
        "-",
        f"[bold green]{result.slm_index:.2f}%[/bold green]",
        "-",
    )
    console.print(summary_table)

    # Eligibility Notice
    if result.eligible_for_leaderboard:
        console.print("[bold green]✔ Result Status: FULL. Eligible for official leaderboard publication.[/bold green]")
    else:
        console.print(
            f"[bold yellow]⚠ Result Status: {result.status}. Not eligible for the full official leaderboard "
            "(sampled evaluation or partial split).[/bold yellow]"
        )

    # 9. Automated Submission if requested
    if submit:
        if not result.eligible_for_leaderboard:
            logger.warning("Result is PARTIAL/INVALID. Only FULL official evaluations can be submitted.")
        if not github_repo:
            logger.error("--submit requires --github-repo (e.g., 'owner/leaderboard-repo')")
            raise typer.BadParameter("Missing --github-repo")

        logger.info(f"Preparing automated Pull Request submission to {github_repo}...")
        repo_url = f"https://github.com/{github_repo}.git" if not github_repo.startswith("http") else github_repo
        git_mgr = GitSubmissionManager(repo_url=repo_url, token=token, dry_run=dry_run)
        git_mgr.prepare_repo()
        branch_name = git_mgr.commit_and_push(
            model_slug=meta.model_slug,
            json_content=json_str,
            slm_index=result.slm_index,
        )

        pr_mgr = PullRequestManager(repo_identifier=github_repo, token=token, dry_run=dry_run)
        pr_info = pr_mgr.create_pull_request(branch_name=branch_name, result=result)
        console.print(Panel(
            f"Pull Request URL: {pr_info['pr_url']}\nHead Branch: {branch_name}\nTarget: {github_repo}:main",
            title="🚀 Submission Completed",
            border_style="green",
        ))

    return result


@app.command(name="list-benchmarks")
def list_benchmarks_cmd():
    """List all registered official benchmarks, dataset provenance, splits, and requirements."""
    setup_logger()

    p_table = Table(title="[Evaluation Profiles]", show_header=True, header_style="bold magenta")
    p_table.add_column("Profile", style="bold cyan")
    p_table.add_column("Status Output", style="yellow")
    p_table.add_column("Tasks Included")
    p_table.add_row("QUICK", "PARTIAL", "5 samples/task for harness smoke testing (5 tasks: MMLU-Pro, GPQA, GSM8K, MATH-500, IFEval)")
    p_table.add_row("SMALL", "PARTIAL", "1,000 samples total, split evenly across the 6 standard tasks incl. BBH")
    p_table.add_row("MEDIUM", "PARTIAL", "10,000 samples total: full small splits + 3,721 each for MMLU-Pro and BBH")
    p_table.add_row("STANDARD", "FULL", "Complete official datasets for core reasoning, math, and instruction (MMLU-Pro, GPQA, BBH, GSM8K, MATH-500, IFEval)")
    p_table.add_row("EXTENDED", "FULL", "Standard + EvalPlus (Docker sandbox), BFCL v4, RULER, XSTest")
    console.print(p_table)

    t_table = Table(title="[Official Verifiable Benchmarks]", show_header=True, header_style="bold green")
    t_table.add_column("Task ID", style="bold yellow")
    t_table.add_column("Category", style="cyan")
    t_table.add_column("Official Source", style="dim")
    t_table.add_column("Split / Count")
    t_table.add_column("Execution Requirements")

    for tid, meta in BENCHMARK_DATASET_META.items():
        reqs = "Python environment"
        if tid == "evalplus":
            reqs = "Docker container sandbox"
        elif tid == "math500":
            reqs = "SymPy equivalence engine"
        count_str = f"{meta.get('expected_count', 'All')} ({meta.get('split', 'test')})"
        t_table.add_row(tid, meta.get("category", "general"), meta.get("name", tid), count_str, reqs)

    console.print(t_table)


@app.command()
def download(
    benchmark: Optional[str] = typer.Option(None, "--benchmark", "-b", help="Specific benchmark to download (e.g. mmlu_pro, gpqa_diamond)"),
    all_benchmarks: bool = typer.Option(False, "--all", "-a", help="Download all core benchmark datasets"),
):
    """Download and pre-cache official benchmark datasets locally."""
    setup_logger()
    dm = DatasetManager()

    benchmarks_to_download = []
    if all_benchmarks:
        benchmarks_to_download = ["mmlu_pro", "gpqa_diamond", "gsm8k", "math500", "ifeval", "xstest"]
    elif benchmark:
        benchmarks_to_download = [benchmark.strip()]
    else:
        logger.info("Specify --benchmark <id> or --all. Run 'slm-bench list-benchmarks' to see available IDs.")
        return

    for bid in benchmarks_to_download:
        logger.info(f"Downloading official dataset for [bold cyan]{bid}[/bold cyan]...")
        try:
            if bid == "mmlu_pro":
                items = dm.load_mmlu_pro()
            elif bid == "gpqa_diamond":
                items = dm.load_gpqa_diamond()
            elif bid == "gsm8k":
                items = dm.load_gsm8k()
            elif bid == "math500":
                items = dm.load_math500()
            elif bid == "ifeval":
                items = dm.load_ifeval()
            elif bid == "xstest":
                items = dm.load_xstest()
            elif bid == "bfcl":
                items = dm.load_bfcl()
            elif bid == "evalplus":
                items = dm.load_evalplus()
            else:
                logger.warning(f"Unrecognized benchmark: {bid}")
                continue
            logger.info(f"✔ Successfully cached [bold green]{bid}[/bold green] ({len(items)} items).")
        except Exception as e:
            logger.error(f"Failed to download {bid}: {e}")


@app.command(name="validate-datasets")
def validate_datasets_cmd():
    """Verify local cache integrity and availability of official datasets."""
    setup_logger()
    dm = DatasetManager()

    table = Table(title="[Benchmark Dataset Cache & Integrity Status]", show_header=True, header_style="bold cyan")
    table.add_column("Benchmark", style="bold")
    table.add_column("Expected Count", style="yellow")
    table.add_column("Cached Count", style="green")
    table.add_column("Cache File")
    table.add_column("Status", style="bold")

    checks = [
        ("mmlu_pro", "mmlu_pro_test.json", 12032),
        ("gpqa_diamond", "gpqa_diamond.json", 198),
        ("gsm8k", "gsm8k_test.json", 1319),
        ("math500", "math500_test.json", 500),
        ("ifeval", "ifeval.json", 541),
        ("xstest", "xstest.json", 450),
    ]

    for bid, fname, expected in checks:
        c_path = dm.cache_dir / fname
        if c_path.exists():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cnt = len(data)
                    status = "[bold green]VERIFIED[/bold green]" if cnt == expected else f"[yellow]PARTIAL ({cnt})[/yellow]"
                    table.add_row(bid, str(expected), str(cnt), str(c_path), status)
            except Exception as e:
                table.add_row(bid, str(expected), "Corrupted", str(c_path), f"[bold red]ERROR: {e}[/bold red]")
        else:
            table.add_row(bid, str(expected), "0", str(c_path), "[dim red]NOT CACHED[/dim red]")

    console.print(table)


@app.command(name="inspect-run")
def inspect_run_cmd(
    result_file: Path = typer.Option(..., "--result-file", "-f", help="Path to BenchmarkResult JSON file"),
    task_id: Optional[str] = typer.Option(None, "--task", "-t", help="Filter by specific task ID"),
    show_failures_only: bool = typer.Option(False, "--failed", help="Show only failed or incorrect questions"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum questions to display"),
):
    """Inspect fine-grained question prompts, model responses, extractions, and correctness."""
    setup_logger()
    if not result_file.exists():
        logger.error(f"File not found: {result_file}")
        raise typer.BadParameter("File not found")

    data = json.loads(result_file.read_text(encoding="utf-8"))
    tasks = data.get("tasks", {})

    target_tasks = [task_id] if task_id else list(tasks.keys())

    for tid in target_tasks:
        if tid not in tasks:
            continue
        tm = tasks[tid]
        q_results = tm.get("question_results") or []
        if not q_results:
            continue

        displayed = 0
        console.print(f"\n[bold green]=== Question Provenance: {tid} ===[/bold green]")
        for q in q_results:
            is_cor = q.get("is_correct", False)
            if show_failures_only and is_cor:
                continue

            status_icon = "✔ PASS" if is_cor else "✘ FAIL"
            status_style = "green" if is_cor else "red"

            console.print(f"\n[{status_style}]Question ID: {q.get('question_id')} | {status_icon}[/{status_style}]")
            console.print(f"[dim]Prompt:[/dim]\n{q.get('raw_prompt')}")
            console.print(f"[dim]Model Response:[/dim]\n{q.get('raw_response')}")
            console.print(f"[bold yellow]Extracted Answer:[/bold yellow] {q.get('extracted_answer')}")
            console.print(f"[bold cyan]Expected Answer:[/bold cyan] {q.get('expected_answer')}")
            if q.get("evaluation_error"):
                console.print(f"[red]Evaluation Error:[/red] {q.get('evaluation_error')}")

            displayed += 1
            if displayed >= limit:
                break


@app.command(name="compare")
def compare_models_cmd(
    file_a: Path = typer.Option(..., "--model-a", "-a", help="Path to first model result JSON"),
    file_b: Path = typer.Option(..., "--model-b", "-b", help="Path to second model result JSON"),
):
    """Compare evaluation metrics and category tradeoffs between two model runs."""
    setup_logger()
    if not file_a.exists() or not file_b.exists():
        logger.error("Both model result JSON files must exist.")
        raise typer.BadParameter("File not found")

    res_a = BenchmarkResult.model_validate_json(file_a.read_text(encoding="utf-8"))
    res_b = BenchmarkResult.model_validate_json(file_b.read_text(encoding="utf-8"))

    table = Table(
        title=f"[Model Comparison: {res_a.model.model_slug} vs {res_b.model.model_slug}]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Benchmark Pillar / Metric", style="bold")
    table.add_column(f"{res_a.model.model_slug} ({res_a.model.quantization})", style="yellow")
    table.add_column(f"{res_b.model.model_slug} ({res_b.model.quantization})", style="green")
    table.add_column("Delta (B - A)", style="bold magenta")

    # Composite SLM-Index
    delta_slm = res_b.slm_index - res_a.slm_index
    table.add_row("Composite SLM-Index", f"{res_a.slm_index:.2f}%", f"{res_b.slm_index:.2f}%", f"{delta_slm:+.2f}%")

    # Categories
    all_cats = sorted(set(list(res_a.categories.keys()) + list(res_b.categories.keys())))
    for c_name in all_cats:
        sc_a = res_a.categories.get(c_name)
        sc_b = res_b.categories.get(c_name)
        val_a = f"{sc_a.normalized_score:.2f}%" if sc_a else "N/A"
        val_b = f"{sc_b.normalized_score:.2f}%" if sc_b else "N/A"
        d_str = f"{(sc_b.normalized_score - sc_a.normalized_score):+.2f}%" if (sc_a and sc_b) else "-"
        table.add_row(f"Category: {c_name.title()}", val_a, val_b, d_str)

    # Common individual benchmark tasks
    common_tasks = sorted(set(res_a.tasks.keys()).intersection(set(res_b.tasks.keys())))
    for t_id in common_tasks:
        tm_a = res_a.tasks[t_id]
        tm_b = res_b.tasks[t_id]
        diff = tm_b.normalized_score - tm_a.normalized_score
        table.add_row(f"  └ Task: {t_id}", f"{tm_a.normalized_score:.2f}%", f"{tm_b.normalized_score:.2f}%", f"{diff:+.2f}%")

    console.print(table)


@app.command(name="export")
def export_reports_cmd(
    results_dir: Path = typer.Option(Path("data/results"), "--results-dir", "-d", help="Directory containing result JSON files"),
    format_type: str = typer.Option("csv", "--format", help="Export format: csv or json"),
    output_file: Path = typer.Option(Path("benchmark_report.csv"), "--out", "-o", help="Target output file"),
):
    """Export aggregated benchmark results to CSV or summary JSON report."""
    setup_logger()
    if not results_dir.exists():
        logger.error(f"Directory not found: {results_dir}")
        raise typer.BadParameter("Directory not found")

    files = list(results_dir.glob("*.json"))
    rows = []
    for f in files:
        try:
            res = BenchmarkResult.model_validate_json(f.read_text(encoding="utf-8"))
            s_dict = res.to_summary_dict()
            s_dict["filename"] = f.name
            rows.append(s_dict)
        except Exception:
            continue

    if not rows:
        logger.warning(f"No valid BenchmarkResult files found in {results_dir}.")
        return

    if format_type.lower() == "csv":
        fieldnames = list(rows[0].keys())
        with open(output_file, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"✔ Successfully exported CSV report ({len(rows)} models) to: [bold green]{output_file}[/bold green]")
    else:
        output_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info(f"✔ Successfully exported JSON report ({len(rows)} models) to: [bold green]{output_file}[/bold green]")


@app.command()
def submit(
    result_file: Path = typer.Option(..., "--result-file", "-f", help="Path to existing benchmark result JSON file"),
    github_repo: str = typer.Option(..., "--github-repo", "-r", help="Target GitHub repository (e.g., owner/repo)"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN", help="GitHub Personal Access Token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode"),
):
    """Submit an existing benchmark result JSON file to a GitHub repository via PR."""
    setup_logger()
    if not result_file.exists():
        logger.error(f"Result file not found: {result_file}")
        raise typer.BadParameter(f"Result file not found: {result_file}")

    data = json.loads(result_file.read_text(encoding="utf-8"))
    result = BenchmarkResult.model_validate(data)

    repo_url = f"https://github.com/{github_repo}.git" if not github_repo.startswith("http") else github_repo
    git_mgr = GitSubmissionManager(repo_url=repo_url, token=token, dry_run=dry_run)
    git_mgr.prepare_repo()
    branch_name = git_mgr.commit_and_push(
        model_slug=result.model.model_slug,
        json_content=result_file.read_text(encoding="utf-8"),
        slm_index=result.slm_index,
    )

    pr_mgr = PullRequestManager(repo_identifier=github_repo, token=token, dry_run=dry_run)
    pr_info = pr_mgr.create_pull_request(branch_name=branch_name, result=result)

    console.print(Panel(
        f"Pull Request URL: {pr_info['pr_url']}\nHead Branch: {branch_name}\nTarget: {github_repo}:main",
        title="🚀 Submission Completed",
        border_style="green",
    ))


@app.command()
def init_website(
    target_dir: Path = typer.Option(Path("./website"), "--target-dir", "-d", help="Directory where static website will be created"),
):
    """Generate or update static GitHub Pages leaderboard website template."""
    setup_logger()
    from .website_template import scaffold_website
    scaffold_website(target_dir)
    logger.info(f"Static website scaffolded successfully at [bold green]{target_dir}[/bold green]")


@app.command(name="compare-quantization")
def compare_quantization_cmd(
    baseline_file: Path = typer.Option(..., "--baseline", "-b", help="JSON result file of higher precision baseline"),
    quantized_file: Path = typer.Option(..., "--quantized", "-q", help="JSON result file of quantized model"),
):
    """Analyze capability retention, degradation percentage, and RAM savings between quantization levels."""
    from .scoring.quantization import compare_quantizations
    setup_logger()

    if not baseline_file.exists() or not quantized_file.exists():
        logger.error("Both baseline and quantized result files must exist.")
        raise typer.BadParameter("File not found")

    base_res = BenchmarkResult.model_validate_json(baseline_file.read_text(encoding="utf-8"))
    quant_res = BenchmarkResult.model_validate_json(quantized_file.read_text(encoding="utf-8"))

    comp = compare_quantizations(base_res, quant_res)

    table = Table(title=f"[Quantization Tradeoff: {comp['model_id']}]", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline Value", style="yellow")
    table.add_column("Quantized Value", style="green")
    table.add_column("Delta / Tradeoff", style="bold magenta")

    table.add_row("Format", comp["baseline_quantization"], comp["quantized_format"], "-")
    table.add_row("SLM-Index Score", f"{comp['baseline_slm_index']:.2f}%", f"{comp['quantized_slm_index']:.2f}%", f"{comp['score_delta']:+.2f}%")
    table.add_row("Degradation", "0.00%", f"{comp['degradation_pct']:.2f}%", f"{comp['retention_pct']:.2f}% retained")
    table.add_row("RAM / VRAM Footprint", f"{comp['baseline_ram_gb']:.2f} GB", f"{comp['quantized_ram_gb']:.2f} GB", f"-{comp['ram_savings_gb']:.2f} GB ({comp['ram_savings_pct']:.1f}% saved)")
    console.print(table)


@app.command()
def tui(
    model_id: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", "--model-id", "-m", help="Default model repository ID"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Default inference backend"),
    endpoint_url: str = typer.Option("http://127.0.0.1:8000/v1", "--endpoint-url", "--endpoint", help="Server endpoint URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization method"),
    reasoning_effort: str = typer.Option("low", "--reasoning-effort", "--effort", help="Default reasoning effort (model-dependent; none, minimal, low, medium, high, xhigh, max)"),
):
    """Launch the interactive Terminal User Interface (TUI) with real-time output verification."""
    from .tui import launch_tui
    launch_tui(
        default_model=model_id,
        default_backend=backend,
        default_endpoint=endpoint_url,
        default_api_key=api_key,
        default_quant=quantization,
        default_effort=reasoning_effort,
    )


@app.command()
def batch(
    batch_file: Path = typer.Option(..., "--batch-file", "-f", help="Text file with model IDs (one per line) or JSON file"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Inference backend"),
    quantization: str = typer.Option("BF16", "--quantization", "-q", help="Quantization method"),
    output_dir: str = typer.Option("data/results", "--output-dir", "-o", help="Target results directory"),
    profile: str = typer.Option("standard", "--profile", "-p", help="Evaluation profile: quick, small, medium, standard, extended"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", "-n", help="Sample limit per task"),
    submit: bool = typer.Option(False, "--submit", "-s", help="Submit PR for each model"),
    github_repo: Optional[str] = typer.Option(None, "--github-repo", help="Target GitHub repository"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN", help="GitHub Token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run"),
):
    """Run evaluation in batch mode for multiple models from a file."""
    setup_logger()
    if not batch_file.exists():
        logger.error(f"Batch file not found: {batch_file}")
        raise typer.BadParameter(f"File not found: {batch_file}")

    lines = [line.strip() for line in batch_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    logger.info(f"Loaded {len(lines)} models from {batch_file} for batch evaluation.")

    if output_dir == "data/results":
        if Path("website/data/results").exists():
            output_dir = "website/data/results"
        elif Path("Website/data/results").exists():
            output_dir = "Website/data/results"

    results = []
    for idx, model_id in enumerate(lines, 1):
        logger.info(f"\n[bold green]Processing model [{idx}/{len(lines)}]: {model_id}[/bold green]")
        res = evaluate(
            model_id=model_id,
            quantization=quantization,
            backend=backend,
            endpoint_url="http://localhost:11434/v1",
            api_key="EMPTY",
            model_ram_gb=None,
            output_dir=output_dir,
            profile=profile,
            benchmark=None,
            tasks=None,
            sample_limit=sample_limit,
            seed=42,
            submit=submit,
            github_repo=github_repo,
            token=token,
            dry_run=dry_run,
            notes="Batch evaluation",
            reasoning_effort=None,
        )
        results.append(res)

    logger.info(f"Batch run completed for {len(results)} models.")
    console.print(f"[bold green]Batch run completed for {len(results)} models.[/bold green]")


def main():
    if len(sys.argv) == 1:
        from .tui import launch_tui
        launch_tui()
    else:
        app()


if __name__ == "__main__":
    main()
