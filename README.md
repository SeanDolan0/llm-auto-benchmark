# Local AI Benchmark

A verifiable evaluation harness and leaderboard for local language models based strictly on **established, official benchmarks**.

[Live Leaderboard](https://seandolan0.github.io/llm-auto-benchmark/) &middot; [Official Benchmark Suite](#official-benchmark-suite) &middot; [Profiles](#evaluation-profiles) &middot; [Scoring & Integrity](#scoring-methodology--benchmark-integrity) &middot; [CLI Usage](#cli-usage) &middot; [Submit a Result](#submitting-a-result)

Most leaderboards evaluate language models on enterprise clusters with hundreds of gigabytes of VRAM. This benchmark evaluates models on hardware people actually own (RTX 3060/4080s, Apple Silicon Macs, consumer CPUs) while upholding the rigorous standards of established academic benchmarks.

Every result on the leaderboard records the host hardware, model footprint (RAM/VRAM), exact dataset revision, 95% Wilson confidence intervals, and full question-by-question provenance.

---

## Benchmark Integrity & Architecture

A core principle of this project is **strict benchmark integrity**:
* **Zero Handcrafted Questions**: We never create custom questions or synthetic mock evaluations and label them as official benchmarks.
* **Official Datasets & Evaluators**: Every evaluation runs the authentic upstream datasets (MMLU-Pro, GPQA Diamond, BBH, GSM8K, MATH-500, IFEval, EvalPlus, BFCL, RULER, XSTest).
* **Elimination of Scoring Shortcuts**: No credit is awarded for keyword presence in an explanation, semantic similarity approximations, or incidental numbers in response text.
* **Secure Isolated Execution**: Untrusted model-generated code is executed inside containerized Docker sandboxes with no network access, memory limits, and wall-clock timeouts. If Docker is unavailable, coding evaluations are reported as `UNAVAILABLE` rather than executing untrusted code on the host.
* **Strict Leaderboard Eligibility**: Only verified runs on complete official splits (`FULL` status) are eligible for official ranking. Seeded sample runs (`PARTIAL`) are transparently flagged and excluded from the leaderboard.

---

## Official Benchmark Suite

| Benchmark | Category | Official Source | Official Split & Count | Evaluation Method & Execution Requirements |
|:---|:---|:---|:---|:---|
| **MMLU-Pro** | Reasoning | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Test (12,032 questions) | 10-option multiple choice (A–J), deterministic extraction, subject breakdowns. |
| **GPQA Diamond** | Reasoning | [idavidrein/gpqa](https://github.com/idavidrein/gpqa) | Diamond (198 questions) | PhD-level Google-proof Q&A. Reproducible seeded choice shuffling and answer-key remapping. |
| **BIG-Bench Hard (BBH)** | Reasoning | [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | All 23 Subtasks (6,511 items) | Task-specific algorithmic and deduction extraction across all 23 official subtasks. |
| **GSM8K** | Mathematics | [openai/grade-school-math](https://github.com/openai/grade-school-math) | Test (1,319 problems) | Multi-step word math. Exact numerical normalization (`#### <num>`); no credit for explanation numbers. |
| **MATH-500** | Mathematics | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Test (500 problems) | Challenging competition math. LaTeX normalization and SymPy symbolic equivalence engine. |
| **IFEval** | Instruction | [google-research/ifeval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Train (541 prompts) | Deterministic verifiable instruction constraints (length, formatting, keywords). Prompt & instruction strict accuracy. |
| **EvalPlus** | Coding | [evalplus/evalplus](https://github.com/evalplus/evalplus) | HumanEval+ (164) & MBPP+ (378) | 80x expanded hidden test suites. Requires Docker sandbox container isolation. |
| **LiveCodeBench** | Coding | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | Release-window problems | Contest programming test cases. Requires Docker sandbox container isolation. |
| **BFCL v4** | Tool Calling | [gorilla-llm/berkeley-function-calling-leaderboard](https://gorilla.cs.berkeley.edu/leaderboard) | Standard & Multi-turn | AST parsing, strict parameter schema typing, native tool call execution, negative rejection. |
| **RULER** | Long Context | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | 4K, 8K, 16K, 32K context | Needle-in-a-haystack, multi-hop retrieval, variable tracking across sequence lengths. |
| **XSTest** | Safety | [paul-rottger/exaggerated-safety](https://github.com/paul-rottger/exaggerated-safety) | Test (450 prompts) | False Refusal Rate (FRR) vs Unsafe Compliance Rate (UCR). Segregated from intelligence score. |

---

## Evaluation Profiles

The evaluation suite supports five tiers, from a harness smoke test to the full specialist suite:

1. **`quick`** (Harness Smoke Test):
   - 5 samples per task across MMLU-Pro, GPQA Diamond, GSM8K, MATH-500, and IFEval (25 total).
   - Status: Always labeled **`PARTIAL`**. Ineligible for the official leaderboard.
2. **`small`** (Sanity Check):
   - 1,000 samples total, split evenly across the six standard tasks (≈167 each), including BBH.
   - Status: Always labeled **`PARTIAL`**. Ineligible for the official leaderboard.
3. **`medium`** (Smell Test):
   - 10,000 samples total: full splits for GPQA (198), MATH-500 (500), IFEval (541), and GSM8K (1,319), plus 3,721 each for MMLU-Pro and BBH.
   - Status: Always labeled **`PARTIAL`**. Ineligible for the official leaderboard.
4. **`standard`** (Core Official Leaderboard):
   - Full official test splits across core general intelligence pillars.
   - Tasks: Complete MMLU-Pro (12,032), GPQA Diamond (198), BBH (23 subtasks), GSM8K (1,319), MATH-500 (500), IFEval (541).
   - Status: Labeled **`FULL`** when all splits complete. Eligible for official leaderboard publication.
5. **`extended`** (Full Specialist Suite):
   - Standard suite + EvalPlus (HumanEval+/MBPP+ in Docker), BFCL v4, RULER, and XSTest.
   - Status: Labeled **`FULL`**.

---

## Scoring Methodology & Benchmark Integrity

### Composite SLM-Index
The composite **SLM-Index** reflects balanced performance across foundational capabilities:
* **Reasoning** (25%): MMLU-Pro, GPQA Diamond, BBH.
* **Mathematics** (25%): GSM8K, MATH-500.
* **Instruction Following** (25%): IFEval (strict prompt accuracy).
* **Coding** (15%): EvalPlus (HumanEval+ / MBPP+ pass@1).
* **Tool Calling** (10%): BFCL v4.

*(When coding or tool calling are unavailable due to sandbox or API constraints, remaining weights are re-normalized proportionally across active pillars).*

### Statistical Confidence
For all binary accuracy metrics, **95% Wilson score confidence intervals** are computed:
$$\hat{p} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$
This allows users to immediately distinguish meaningful model advantages from statistical sampling noise.

### Safety Metric Segregation
Safety evaluations (such as XSTest and HarmBench) report **False Refusal Rate (FRR)** and **Unsafe Compliance Rate (UCR)**. Safety scores are kept strictly segregated from the general intelligence composite score so that refusal behavior never artificially inflates or penalizes reasoning or coding marks.

---

## Supported Inference Backends

The benchmark adapter layer connects to any backend via a standardized inference gateway:
* **OpenAI-Compatible** (`--backend openai`): Connects to Ollama, LM Studio, vLLM server, LocalAI, or custom OpenAI-compatible endpoints (`/v1/chat/completions`). Supports native streaming, timing telemetry, and reasoning effort controls.
* **Transformers** (`--backend transformers`): Hugging Face native inference with bitsandbytes 4-bit / 8-bit quantization.
* **llama.cpp** (`--backend llamacpp`): High-efficiency GGUF inference with layer offloading to GPU.
* **vLLM** (`--backend vllm`): High-throughput PagedAttention engine.
* **Mock** (`--backend mock`): Zero-GPU testing backend that exercises dataset loading, extraction, and reporting pipelines.

---

## Installation

```bash
git clone https://github.com/SeanDolan0/llm-auto-benchmark.git
cd llm-auto-benchmark

# Core CLI, dataset manager, and TUI studio
pip install -e .

# Optional benchmark dependencies
pip install -e ".[reasoning]"      # datasets, huggingface_hub
pip install -e ".[math]"           # sympy for mathematical equivalence
pip install -e ".[coding]"         # isolated execution dependencies
pip install -e ".[all-benchmarks]" # full benchmark suite dependencies

# Inference backends
pip install -e ".[transformers]"   # Hugging Face + bitsandbytes
pip install -e ".[llamacpp]"       # llama.cpp Python bindings
pip install -e ".[vllm]"           # vLLM engine
```

### Sandbox Requirement
For coding benchmarks (`evalplus`, `livecodebench`), **Docker** must be installed and running on the host. If Docker is inactive, coding tasks will be reported as `UNAVAILABLE` to maintain system security.

---

## Dataset Management

Datasets are cached in `data/datasets/` and verified against pinned upstream revisions.

```bash
# List all registered official benchmarks, splits, and counts
slm-bench list-benchmarks

# Download official datasets
slm-bench download --all
# Or download a specific benchmark
slm-bench download --benchmark mmlu_pro

# Validate local cache integrity and sample counts
slm-bench validate-datasets
```

---

## CLI Usage

### 1. Host Hardware & Model Footprint Inspection
Inspect your hardware and auto-detect model VRAM/RAM requirements:
```bash
slm-bench info --model-id "Qwen/Qwen2.5-7B-Instruct"
```

### 2. Run an Evaluation
#### Smoke test with Mock backend:
```bash
slm-bench evaluate \
  --model-id "Qwen/Qwen2.5-7B-Instruct" \
  --backend mock \
  --profile quick \
  --sample-limit 2
```

#### Real evaluation via local Ollama endpoint (OpenAI API):
```bash
slm-bench evaluate \
  --model-id "qwen2.5:7b" \
  --backend openai \
  --endpoint-url "http://localhost:11434/v1" \
  --profile standard \
  --model-ram-gb 4.8 \
  --notes "RTX 4080 16GB, full GPU offload"
```

#### Evaluation with reasoning effort (for reasoning models):
```bash
slm-bench evaluate \
  --model-id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --backend openai \
  --endpoint-url "http://localhost:11434/v1" \
  --profile standard \
  --reasoning-effort high
```

#### Run a single official benchmark:
```bash
slm-bench evaluate \
  --model-id "meta-llama/Llama-3.2-3B-Instruct" \
  --backend openai \
  --benchmark gpqa_diamond
```

#### Reasoning models and token budgets:
Reasoning models may spend their entire output budget on a separate thinking channel and return an empty answer. The harness exposes generic controls rather than model-specific rules:

```bash
slm-bench evaluate ... \
  --thinking off \
  --max-new-tokens 2048 \
  --truncation-policy retry \
  --max-token-budget 16384 \
  --adaptive-budget
```

- `--thinking auto|on|off` passes `chat_template_kwargs` to servers that support it.
- `--max-new-tokens` overrides the per-question answer budget.
- `--truncation-policy retry` doubles the budget after a length-truncated answer until `--max-token-budget`; it never repeats at the cap. `invalid` or `accept` are also available.
- `--adaptive-budget` remembers a per-task starting cap in `--budget-state` (default `~/.cache/slm-bench/token_budgets.json`), halves it when a task finishes far below it, and raises it when a task needs more.
- Results report `truncated_samples`, `cap_exceeded_samples`, `truncation_rate`, `tokens_used`, `tokens_per_correct`, `budget_start_max`, and `completion_budget_p95` in the result JSON and per-task provenance.

#### Reuse prior answers (replay checkpoints):
Pass `--checkpoint PATH` to read an existing JSONL checkpoint and append every new model call to it. A later run reuses calls whose model, quantization, backend, reasoning effort, task, dataset revision, and generation parameters match; everything else runs fresh.

```bash
# Start with a smaller run; its answers are written to the checkpoint.
slm-bench evaluate --model-id "LiquidAI/LFM3-1.7B" --backend openai \
  --endpoint-url "http://127.0.0.1:8099/v1" --quantization GGUF_F16 \
  --profile small --checkpoint checkpoints/lfm3.jsonl

# A later run reuses the matching answers instead of re-running them.
slm-bench evaluate --model-id "LiquidAI/LFM3-1.7B" --backend openai \
  --endpoint-url "http://127.0.0.1:8099/v1" --quantization GGUF_F16 \
  --profile medium --checkpoint checkpoints/lfm3.jsonl
```

Reused questions are flagged (`"reused": true`) and counted in `provenance.reused_samples`; they are excluded from latency/throughput telemetry, and `local_efficiency` is omitted when an entire run is replayed. Checkpoint files are append-only and safe to keep across runs.

### 3. Inspect Question Provenance
Inspect exact prompts, raw model responses, extracted answers, and constraint failure reasons:
```bash
slm-bench inspect-run -f Website/data/results/Qwen_Qwen2.5-7B-Instruct-BF16.json --limit 5
```

### 4. Compare Two Models
Compare metric breakdowns and category tradeoffs between two model runs:
```bash
slm-bench compare \
  --model-a Website/data/results/Qwen_Qwen2.5-7B-Instruct-BF16.json \
  --model-b Website/data/results/meta-llama_Llama-3.2-3B-Instruct-BF16.json
```

### 5. Compare Quantizations
Analyze capability degradation vs RAM/VRAM savings between precision levels (e.g. BF16 vs Q4):
```bash
slm-bench compare-quantization \
  --baseline results/Qwen2.5-7B-BF16.json \
  --quantized results/Qwen2.5-7B-GGUF_Q4.json
```

### 6. Interactive Terminal Studio (TUI)
Launch the interactive terminal user interface:
```bash
slm-bench tui
```

---

## Submitting a Result

Results are submitted via Pull Request to `Website/data/results/<model-slug>.json`.

Automated submission directly from the CLI:
```bash
slm-bench evaluate \
  --model-id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --backend openai \
  --profile standard \
  --submit \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

Or submit an existing evaluation JSON:
```bash
slm-bench submit \
  --result-file "Website/data/results/Qwen-Qwen2.5-7B-Instruct-BF16.json" \
  --github-repo "SeanDolan0/llm-auto-benchmark" \
  --token "$GITHUB_TOKEN"
```

---

## Running the Website Leaderboard Locally

The leaderboard is static HTML/CSS/JS with zero build step:
```bash
# Aggregate result JSON files into leaderboard.json
python Website/scripts/aggregate.py

# Launch local preview server
python -m http.server -d Website 8000
```
Open <http://localhost:8000> in your browser.

---

## Testing

Run the full automated test suite:
```bash
pytest -v
```

All 53 unit and integration tests verify:
* Exact deterministic answer extraction across multiple-choice options.
* Anti-shortcut enforcement (zero credit for explanation keyword mentions).
* Choice shuffling and key remapping for GPQA Diamond.
* Mathematical LaTeX normalization and SymPy symbolic equivalence.
* Deterministic constraint evaluation for IFEval.
* Container sandbox security (rejection of untrusted host `exec()`).
* Wilson 95% confidence interval calculations.
* Leaderboard eligibility enforcement (FULL vs PARTIAL vs INVALID).
* Website aggregation and Schema 2.0.0 serialization.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
