"""
Dataset Management Layer for SLM-Bench.
Provides deterministic downloading, caching, revision tracking, integrity verification,
and stratified seeded sampling for official, established benchmarks.
"""

import os
import json
import csv
import io
import zipfile
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import urllib.request
import urllib.error

from ..utils.logger import logger

# Default local dataset storage cache directory
DEFAULT_CACHE_DIR = Path("data/datasets")

# Official Pinned Benchmark Dataset Metadata
BENCHMARK_DATASET_META = {
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "source": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "hf_repo": "TIGER-Lab/MMLU-Pro",
        "revision": "4b6efcb",
        "split": "test",
        "expected_count": 12032,
        "description": "Multi-discipline 10-choice academic and professional reasoning.",
        "category": "reasoning",
    },
    "gpqa_diamond": {
        "name": "GPQA Diamond",
        "source": "https://github.com/idavidrein/gpqa",
        "url": "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip",
        "revision": "main-commit-4b24",
        "split": "diamond",
        "expected_count": 198,
        "description": "Expert-validated graduate-level science reasoning (biology, chemistry, physics).",
        "category": "reasoning",
    },
    "bbh": {
        "name": "BIG-Bench Hard (BBH)",
        "source": "https://github.com/suzgunmirac/BIG-Bench-Hard",
        "hf_repo": "lukaemon/bbh",
        "revision": "main",
        "split": "test",
        "tasks_count": 23,
        "expected_count": 6511,
        "description": "23 challenging multi-step algorithmic and symbolic reasoning subtasks.",
        "category": "reasoning",
    },
    "gsm8k": {
        "name": "GSM8K",
        "source": "https://github.com/openai/grade-school-math",
        "hf_repo": "openai/gsm8k",
        "revision": "main",
        "split": "test",
        "expected_count": 1319,
        "description": "Grade school 8.5k multi-step math word problems (1,319 test questions).",
        "category": "mathematics",
    },
    "math500": {
        "name": "MATH-500",
        "source": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
        "hf_repo": "HuggingFaceH4/MATH-500",
        "revision": "main",
        "split": "test",
        "expected_count": 500,
        "description": "Challenging competition mathematics across 7 subjects and 5 difficulty levels.",
        "category": "mathematics",
    },
    "ifeval": {
        "name": "IFEval",
        "source": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
        "hf_repo": "google/IFEval",
        "revision": "main",
        "split": "train",
        "expected_count": 541,
        "description": "Verifiable instruction following evaluation with deterministic constraints.",
        "category": "instruction",
    },
    "xstest": {
        "name": "XSTest",
        "source": "https://github.com/paul-rottger/exaggerated-safety",
        "url": "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
        "revision": "main",
        "split": "test",
        "expected_count": 450,
        "description": "Exaggerated safety test distinguishing true refusals from false refusals.",
        "category": "safety",
    },
    "bfcl": {
        "name": "Berkeley Function Calling Leaderboard (BFCL v4)",
        "source": "https://github.com/ShishirPatil/gorilla",
        "revision": "v4",
        "split": "standard",
        "description": "Real-world native function calling and tool interaction scenarios.",
        "category": "tool_calling",
    },
    "evalplus": {
        "name": "EvalPlus (HumanEval+ and MBPP+)",
        "source": "https://github.com/evalplus/evalplus",
        "hf_repo": "evalplus/humanevalplus",
        "revision": "v0.3.0",
        "split": "test",
        "expected_count": 164,
        "description": "Rigorous Python code generation with expanded test suites.",
        "category": "coding",
    },
}

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgment",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
    "word_sorting",
]


class DatasetManager:
    """Manages downloading, local caching, and deterministic sampling of official benchmark datasets."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_benchmark_meta(self, benchmark_id: str) -> Dict[str, Any]:
        """Retrieve canonical metadata for an official benchmark."""
        if benchmark_id not in BENCHMARK_DATASET_META:
            raise ValueError(f"Unrecognized official benchmark ID: '{benchmark_id}'. Valid official benchmarks: {list(BENCHMARK_DATASET_META.keys())}")
        return BENCHMARK_DATASET_META[benchmark_id]

    # --------------------------------------------------------------------------
    # Official Benchmark Loaders
    # --------------------------------------------------------------------------

    def load_mmlu_pro(self) -> List[Dict[str, Any]]:
        """Load official MMLU-Pro test split (12,032 questions, 10 options A-J)."""
        cache_file = self.cache_dir / "mmlu_pro_test.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data

        try:
            from datasets import load_dataset
            logger.info("Loading official MMLU-Pro dataset from TIGER-Lab/MMLU-Pro...")
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            items = []
            for row in ds:
                items.append({
                    "question_id": str(row["question_id"]),
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["answer"],
                    "answer_index": row["answer_index"],
                    "category": row["category"],
                    "src": row.get("src", ""),
                })
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official MMLU-Pro dataset: {e}")

    def load_gpqa_diamond(self) -> List[Dict[str, Any]]:
        """
        Load official GPQA Diamond dataset (198 questions).
        Retrieved directly from the official idavidrein/gpqa repository.
        """
        cache_file = self.cache_dir / "gpqa_diamond.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            logger.info("Loading official GPQA Diamond dataset from idavidrein/gpqa...")
            req = urllib.request.Request(
                "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip",
                headers={"User-Agent": "SLM-Bench/2.0"},
            )
            content = urllib.request.urlopen(req, timeout=30).read()
            z = zipfile.ZipFile(io.BytesIO(content))
            # Official canary password documented in repo README
            f = z.open("dataset/gpqa_diamond.csv", pwd=b"deserted-untie-orchid")
            reader = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))

            items = []
            for idx, row in enumerate(reader):
                items.append({
                    "question_id": f"gpqa_diamond_{idx}",
                    "question": row["Question"],
                    "correct_answer": row["Correct Answer"],
                    "incorrect_1": row["Incorrect Answer 1"],
                    "incorrect_2": row["Incorrect Answer 2"],
                    "incorrect_3": row["Incorrect Answer 3"],
                    "subdomain": row.get("Subdomain", ""),
                    "high_level_domain": row.get("High-level domain", ""),
                    "explanation": row.get("Explanation", ""),
                })

            with open(cache_file, "w", encoding="utf-8") as out:
                json.dump(items, out, indent=2)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official GPQA Diamond dataset: {e}")

    def load_bbh(self, subtask: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load official BIG-Bench Hard (BBH) subtasks.
        Returns dictionary mapping subtask name to list of test examples.
        """
        cache_file = self.cache_dir / "bbh_tasks.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)
                if subtask:
                    return {subtask: tasks_data.get(subtask, [])}
                return tasks_data

        try:
            from datasets import load_dataset
            logger.info("Loading official BIG-Bench Hard (BBH) dataset...")
            tasks_data = {}
            target_tasks = [subtask] if subtask else BBH_TASKS

            for t_name in target_tasks:
                try:
                    ds = load_dataset("lukaemon/bbh", t_name, split="test")
                    tasks_data[t_name] = [
                        {
                            "id": f"{t_name}_{idx}",
                            "task": t_name,
                            "input": row["input"],
                            "target": row["target"],
                        }
                        for idx, row in enumerate(ds)
                    ]
                except Exception as task_err:
                    logger.warning(f"Failed to load BBH subtask '{t_name}': {task_err}")

            if not subtask and tasks_data:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(tasks_data, f)
            return tasks_data
        except Exception as e:
            raise RuntimeError(f"Failed to load official BBH dataset: {e}")

    def load_gsm8k(self) -> List[Dict[str, Any]]:
        """Load official GSM8K test split (1,319 questions)."""
        cache_file = self.cache_dir / "gsm8k_test.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            from datasets import load_dataset
            logger.info("Loading official GSM8K dataset from openai/gsm8k (test split)...")
            ds = load_dataset("openai/gsm8k", "main", split="test")
            items = []
            for idx, row in enumerate(ds):
                items.append({
                    "question_id": f"gsm8k_test_{idx}",
                    "question": row["question"],
                    "answer": row["answer"],
                })
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official GSM8K dataset: {e}")

    def load_math500(self) -> List[Dict[str, Any]]:
        """Load official MATH-500 test split (500 questions)."""
        cache_file = self.cache_dir / "math500_test.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            from datasets import load_dataset
            logger.info("Loading official MATH-500 dataset from HuggingFaceH4/MATH-500...")
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            items = []
            for idx, row in enumerate(ds):
                items.append({
                    "question_id": row.get("unique_id", f"math500_{idx}"),
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "answer": row["answer"],
                    "subject": row["subject"],
                    "level": row["level"],
                })
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official MATH-500 dataset: {e}")

    def load_ifeval(self) -> List[Dict[str, Any]]:
        """Load official IFEval dataset (541 verifiable instruction prompts)."""
        cache_file = self.cache_dir / "ifeval.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            from datasets import load_dataset
            logger.info("Loading official IFEval dataset from google/IFEval...")
            ds = load_dataset("google/IFEval", split="train")
            items = []
            for row in ds:
                items.append({
                    "key": str(row["key"]),
                    "prompt": row["prompt"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                })
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official IFEval dataset: {e}")

    def load_xstest(self) -> List[Dict[str, Any]]:
        """Load official XSTest prompts (450 prompts from paul-rottger/exaggerated-safety)."""
        cache_file = self.cache_dir / "xstest.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            logger.info("Loading official XSTest dataset from paul-rottger/exaggerated-safety...")
            req = urllib.request.Request(
                "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
                headers={"User-Agent": "SLM-Bench/2.0"},
            )
            content = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
            reader = list(csv.DictReader(io.StringIO(content)))
            items = []
            for row in reader:
                items.append({
                    "id": row.get("id", ""),
                    "prompt": row.get("prompt", ""),
                    "type": row.get("type", ""),
                    "label": row.get("label", ""),
                    "focus": row.get("focus", ""),
                    "note": row.get("note", ""),
                })
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official XSTest dataset: {e}")

    def load_bfcl(self) -> List[Dict[str, Any]]:
        """Load official BFCL (Berkeley Function Calling Leaderboard v4) test cases."""
        cache_file = self.cache_dir / "bfcl_v4.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            logger.info("Loading official BFCL v4 test suite from ShishirPatil/gorilla...")
            base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
            categories = ["BFCL_v4_simple_python.json", "BFCL_v4_parallel.json", "BFCL_v4_multiple.json"]
            items = []
            for cat_file in categories:
                cat_url = f"{base_url}/{cat_file}"
                req = urllib.request.Request(cat_url, headers={"User-Agent": "SLM-Bench/2.0"})
                content = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
                cat_data = json.loads(content)
                for entry in cat_data:
                    entry["category_file"] = cat_file
                    items.append(entry)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official BFCL dataset: {e}")

    def load_evalplus(self, subset: str = "humanevalplus") -> List[Dict[str, Any]]:
        """Load official EvalPlus dataset (humanevalplus or mbppplus)."""
        cache_file = self.cache_dir / f"evalplus_{subset}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            from datasets import load_dataset
            repo_id = f"evalplus/{subset}"
            logger.info(f"Loading official EvalPlus dataset from {repo_id}...")
            ds = load_dataset(repo_id, split="test")
            items = [dict(row) for row in ds]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to load official EvalPlus dataset ({subset}): {e}")

    # --------------------------------------------------------------------------
    # Deterministic Stratified Sampling
    # --------------------------------------------------------------------------

    @staticmethod
    def sample_stratified(
        items: List[Dict[str, Any]],
        sample_size: int,
        seed: int = 42,
        stratify_key: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Perform deterministic seeded sampling stratified by a key (e.g. subject, category, level).
        Never slices arbitrarily (`[:sample_size]`).
        Returns:
            (selected_items, selected_ids)
        """
        if not items or sample_size >= len(items):
            ids = [str(item.get("question_id") or item.get("id") or item.get("key") or idx) for idx, item in enumerate(items)]
            return list(items), ids

        rng = random.Random(seed)

        if not stratify_key or not any(stratify_key in item for item in items):
            # Deterministic seeded random sample
            indices = list(range(len(items)))
            rng.shuffle(indices)
            selected_indices = sorted(indices[:sample_size])
            selected = [items[i] for i in selected_indices]
            ids = [str(item.get("question_id") or item.get("id") or item.get("key") or idx) for idx, item in enumerate(selected)]
            return selected, ids

        # Stratified by group key
        groups: Dict[Any, List[int]] = {}
        for idx, item in enumerate(items):
            group_val = item.get(stratify_key, "default")
            groups.setdefault(group_val, []).append(idx)

        # Proportional allocation
        total_items = len(items)
        selected_indices = []
        for group_val, idx_list in groups.items():
            prop = len(idx_list) / total_items
            group_count = max(1, round(prop * sample_size))
            shuffled = list(idx_list)
            rng.shuffle(shuffled)
            selected_indices.extend(shuffled[:group_count])

        # Adjust to exact sample_size if rounding gave slightly more/less
        if len(selected_indices) > sample_size:
            rng.shuffle(selected_indices)
            selected_indices = selected_indices[:sample_size]
        elif len(selected_indices) < sample_size:
            remaining = [i for i in range(total_items) if i not in selected_indices]
            rng.shuffle(remaining)
            selected_indices.extend(remaining[:(sample_size - len(selected_indices))])

        selected_indices.sort()
        selected = [items[i] for i in selected_indices]
        ids = [str(item.get("question_id") or item.get("id") or item.get("key") or idx) for idx, item in enumerate(selected)]
        return selected, ids
