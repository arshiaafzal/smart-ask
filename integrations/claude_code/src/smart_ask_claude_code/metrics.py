"""Prompt-free persistence for metrics emitted by SmartAsk."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlMetricsSink:
    """Append complete SmartAsk metrics envelopes with process-local locking."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = Lock()
        self._closed = False

    def write(self, value: dict[str, Any]) -> None:
        line = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("metrics sink is closed")
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._file.close()
