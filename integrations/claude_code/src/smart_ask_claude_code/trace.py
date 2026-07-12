"""Compact persistence for content-bearing conversation trace events."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from .metrics import JsonlSink


class JsonlTraceSink:
    """Compact canonical runtime events for one launcher trace file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._sink = JsonlSink(path)
        self._lock = Lock()
        self._runs: dict[str, int] = {}
        self._next_run = 1
        self._sink.write({
            "event": "trace_start",
            "schema": "smart-ask.conversation-trace-log/v1",
        })

    def write(self, value: dict[str, Any]) -> None:
        if value.get("schema") != "smart-ask.conversation-trace-event/v1":
            raise ValueError("trace event has an unsupported schema")
        run_id = value.get("run_id")
        event = value.get("event")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("trace event requires a run_id")
        if not isinstance(event, str) or not event:
            raise ValueError("trace event requires an event name")
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                if event != "run_start":
                    raise ValueError("first event for a run must be run_start")
                run = self._next_run
                self._next_run += 1
                self._runs[run_id] = run
            compact = {
                "event": event,
                "run": run,
                **{
                    key: item
                    for key, item in value.items()
                    if key not in ("schema", "run_id", "sequence", "event")
                },
            }
            if event == "run_start":
                compact["run_id"] = run_id
            self._sink.write(compact)

    def close(self) -> None:
        self._sink.close()
