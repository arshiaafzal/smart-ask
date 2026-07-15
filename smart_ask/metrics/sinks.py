"""Prompt-free persistence for metrics emitted by SmartAsk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlMetricsSink:
    """Append JSON objects with process-local locking and immediate flushing."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Metrics can still carry operational identifiers. Do not depend on
        # the caller's umask to keep a newly-created file private.
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        os.chmod(self.path, 0o600)
        self._file = os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)
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
