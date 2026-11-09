"""
Adaptive per-task token budgets.

The manager remembers the cap that a task needed last time, starts there on the
next run, and halves the start when a task finishes far below it. State is
persisted per model / quantization / backend / thinking mode / task, so it is
learned rather than hardcoded.
"""

import json
import os
from pathlib import Path
from typing import Optional

from ..utils.logger import logger


class BudgetManager:
    """Persist and update per-task starting token caps."""

    def __init__(
        self,
        path: str,
        model_id: str,
        quantization: str,
        backend: str = "",
        thinking: str = "auto",
        min_start: int = 128,
    ):
        self.path = Path(path).expanduser()
        self.model_id = model_id
        self.quantization = quantization
        self.backend = backend
        self.thinking = thinking
        self.min_start = max(1, int(min_start))

    def _key(self, task_id: str) -> str:
        return f"{self.model_id}|{self.quantization}|{self.backend}|{self.thinking}|{task_id}"

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable budget state {self.path}: {exc}")
            return {}

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def start_for(self, task_id: str, base: int, cap: int) -> int:
        """Return the learned starting cap, free to move between min_start and cap."""
        state = self._read()
        entry = state.get(self._key(task_id), {})
        learned = entry.get("start") if isinstance(entry, dict) else entry
        try:
            learned = int(learned)
        except (TypeError, ValueError):
            learned = int(base)
        return max(self.min_start, min(learned, cap))

    def record(
        self,
        task_id: str,
        base: int,
        start: int,
        cap: int,
        final_tokens: int,
        truncated: bool,
        final_budget: Optional[int] = None,
    ) -> int:
        """Update the learned cap after a call and return the next start."""
        observed = max(1, int(final_tokens))
        if truncated:
            next_start = min(cap, max(start, int(final_budget or start)))
            required_cap = None
        elif observed <= max(1, start // 2):
            next_start = max(self.min_start, start // 2)
            required_cap = int(final_budget or self._next_pow2(observed))
        else:
            next_start = min(cap, max(start, self._next_pow2(observed)))
            required_cap = int(final_budget or next_start)
        next_start = max(self.min_start, min(next_start, cap))
        if required_cap is not None:
            required_cap = max(self.min_start, min(required_cap, cap))

        state = self._read()
        state[self._key(task_id)] = {
            "start": next_start,
            "last_tokens": observed,
            "required_cap": required_cap,
            "truncated": bool(truncated),
            "cap_exceeded": bool(truncated and final_budget is not None and final_budget >= cap),
        }
        self._write(state)
        return next_start

    @staticmethod
    def _next_pow2(value: int) -> int:
        value = max(1, int(value))
        return 1 << (value - 1).bit_length()
