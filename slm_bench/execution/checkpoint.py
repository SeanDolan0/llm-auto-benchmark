"""
Replay checkpoints for benchmark inference calls.

A checkpoint is a JSONL file containing one record per model call. Records are
keyed by model identity, backend, reasoning effort, task, dataset revision,
call mode, exact messages or prompt, and generation parameters, so a later run
can serve identical calls without touching the model.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logger import logger

EVALUATOR_VERSION = "2.0.0"


class CheckpointStore:
    """Read and append replay records for one model/backend context."""

    def __init__(
        self,
        path: str,
        model_id: str,
        quantization: str,
        backend: str = "",
        backend_version: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        server: str = "",
        evaluator_version: str = EVALUATOR_VERSION,
    ):
        self.path = Path(path)
        self.model_id = model_id
        self.quantization = quantization
        self.backend = backend
        self.backend_version = backend_version
        self.reasoning_effort = reasoning_effort
        self.server = server
        self.evaluator_version = evaluator_version
        self.entries: Dict[str, dict] = {}
        self.loaded = 0
        self.hits = 0
        self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_trailing_newline()

    def _ensure_trailing_newline(self) -> None:
        """Repair a partially written final line before appending new records."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            with self.path.open("rb") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    with self.path.open("ab") as out:
                        out.write(b"\n")
        except OSError:
            pass

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError):
                        logger.debug(f"Skipping malformed checkpoint line {line_number}")
                        continue
                    if not isinstance(record, dict):
                        continue
                    if record.get("model_id") != self.model_id:
                        continue
                    if record.get("quantization") != self.quantization:
                        continue
                    if record.get("backend") != self.backend:
                        continue
                    if record.get("backend_version") != self.backend_version:
                        continue
                    if record.get("reasoning_effort") != self.reasoning_effort:
                        continue
                    if record.get("server") != self.server:
                        continue
                    if record.get("evaluator_version") != self.evaluator_version:
                        continue
                    key = record.get("key")
                    if not isinstance(key, str) or not key:
                        continue
                    if not isinstance(record.get("response"), str):
                        continue
                    try:
                        int(record.get("tokens_generated", 0))
                    except (TypeError, ValueError):
                        continue
                    if key not in self.entries:
                        self.loaded += 1
                    self.entries[key] = record
        except OSError as exc:
            logger.debug(f"Could not read checkpoint {self.path}: {exc}")

    @staticmethod
    def make_key(
        model_id: str,
        quantization: str,
        backend: str,
        backend_version: Optional[str],
        reasoning_effort: Optional[str],
        server: str,
        evaluator_version: str,
        task_id: str,
        dataset_revision: str,
        mode: str,
        payload: Dict[str, Any],
    ) -> str:
        material = json.dumps(
            {
                "model_id": model_id,
                "quantization": quantization,
                "backend": backend,
                "backend_version": backend_version,
                "reasoning_effort": reasoning_effort,
                "server": server,
                "evaluator_version": evaluator_version,
                "task_id": task_id,
                "dataset_revision": dataset_revision,
                "mode": mode,
                "payload": payload,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        return self.entries.get(key)

    def put(self, record: Dict[str, Any]) -> None:
        self.entries[record["key"]] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Compatibility no-op: records are flushed on every append."""
        return
