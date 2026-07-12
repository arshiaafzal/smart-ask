"""Compact persistence for content-bearing conversation trace events."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .metrics import JsonlSink


class JsonlTraceSink:
    """Remove file-local redundancy while preserving interleaved run evidence."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._sink = JsonlSink(path)
        self._lock = Lock()
        self._runs: dict[str, int] = {}
        self._next_run = 1
        self._header_context: str | None = None
        self._contexts: dict[str, int] = {}
        self._next_context = 1
        self._attempt_counts: dict[int, int] = {}
        self._current_attempt: dict[int, int] = {}
        self._attempt_models: dict[tuple[int, int], str | None] = {}
        self._interned: dict[str, dict[str, int]] = {
            "content": {},
            "metadata": {},
            "change": {},
        }

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _without_defaults(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if item is not None and item is not False and item != {}
        }

    def _context_for(
        self,
        value: dict[str, Any],
    ) -> tuple[int, dict[str, Any], bool]:
        context = self._without_defaults({
            "session_id": value.get("session_id"),
            "strategy": value.get("strategy"),
        })
        canonical = self._canonical(context)
        reference = self._contexts.get(canonical)
        created = reference is None
        if reference is None:
            reference = self._next_context
            self._next_context += 1
            self._contexts[canonical] = reference
        return reference, context, created

    def _intern(
        self,
        value: dict[str, Any],
        *,
        kind: str,
        payload_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        payload = {
            key: value[key]
            for key in payload_keys
            if key in value
        }
        if not payload:
            return value
        canonical = self._canonical(payload)
        reference = self._interned[kind].get(canonical)
        if reference is None:
            reference = len(self._interned[kind]) + 1
            self._interned[kind][canonical] = reference
            return {**value, f"{kind}_id": reference}
        return {
            **{
                key: item
                for key, item in value.items()
                if key not in payload_keys
            },
            f"{kind}_ref": reference,
        }

    def _run_start(
        self,
        value: dict[str, Any],
        *,
        run: int,
        run_id: str,
    ) -> list[dict[str, Any]]:
        context_ref, context, context_created = self._context_for(value)
        context_key = self._canonical(context)
        rows: list[dict[str, Any]] = []
        if self._header_context is None:
            self._header_context = context_key
            rows.append({
                "event": "trace_start",
                "schema": "smart-ask.conversation-trace-log/v2",
                **context,
            })
        elif context_key != self._header_context and context_created:
            rows.append({"event": "context_start", "context": context_ref, **context})
        compact = {
            "event": "run_start",
            "run": run,
            "run_id": run_id,
            **self._without_defaults({
                "agent_id": value.get("agent_id"),
                "parent_agent_id": value.get("parent_agent_id"),
            }),
        }
        if context_key != self._header_context:
            compact["context"] = context_ref
        rows.append(compact)
        return rows

    @staticmethod
    def _route(value: dict[str, Any], run: int) -> dict[str, Any]:
        route = value.get("route")
        if not isinstance(route, dict):
            raise ValueError("route trace event requires a route object")
        evidence = route.get("events")
        compact = {"event": "route", "run": run}
        if isinstance(evidence, list) and len(evidence) == 1:
            item = evidence[0]
            compact.update({
                "outcome": item.get("outcome"),
                "reason": item.get("reason"),
                "source": item.get("source"),
            })
        elif evidence:
            compact["evidence"] = evidence
        compact.update({
            "model": route.get("model"),
            "phase": route.get("phase"),
        })
        if route.get("action") != "execute":
            compact["action"] = route.get("action")
        return JsonlTraceSink._without_defaults(compact)

    def _compact(self, value: dict[str, Any], run: int) -> dict[str, Any]:
        event = value["event"]
        compact = {
            "event": event,
            "run": run,
            **{
                key: item
                for key, item in value.items()
                if key not in ("schema", "run_id", "sequence", "event")
            },
        }
        if event == "route":
            return self._route(value, run)
        if event == "message_start":
            compact = self._without_defaults(compact)
        elif event == "request_metadata":
            compact = self._without_defaults(compact)
            compact = self._intern(
                compact,
                kind="metadata",
                payload_keys=("parameters", "extensions"),
            )
        elif event == "context_block":
            compact = self._intern(
                compact,
                kind="content",
                payload_keys=(
                    "block",
                    "text_field",
                    "chunk_index",
                    "chunk_count",
                    "text",
                ),
            )
        elif event == "attempt_start":
            attempt = self._attempt_counts.get(run, 0) + 1
            self._attempt_counts[run] = attempt
            self._current_attempt[run] = attempt
            self._attempt_models[(run, attempt)] = compact.get("selected_model")
            compact = {"event": event, "run": run, "attempt": attempt}
        elif event == "context_change":
            compact.pop("phase", None)
            compact.pop("index", None)
            compact["attempt"] = self._current_attempt[run]
            change = compact.pop("change", None)
            if isinstance(change, dict):
                compact.update(change)
            compact = self._intern(
                compact,
                kind="change",
                payload_keys=(
                    "operation",
                    "values",
                    "text",
                    "chunk_index",
                    "chunk_count",
                ),
            )
        elif event == "attempt_end":
            attempt = self._current_attempt[run]
            selected = self._attempt_models[(run, attempt)]
            actual = compact.get("actual_model")
            compact = self._without_defaults({
                "event": event,
                "run": run,
                "attempt": attempt,
                "actual_model": actual if actual != selected else None,
                "stop_reason": compact.get("stop_reason"),
                "error": compact.get("error"),
                "cancelled": compact.get("cancelled"),
            })
        elif event == "attempt_output":
            compact.pop("phase", None)
            compact["attempt"] = self._current_attempt[run]
            if compact.get("chunk_count") == 1:
                compact.pop("chunk_count", None)
                compact.pop("chunk_index", None)
        elif event == "run_end":
            compact = self._without_defaults(compact)
        return compact

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
                rows = self._run_start(value, run=run, run_id=run_id)
            else:
                rows = [self._compact(value, run)]
            for row in rows:
                self._sink.write(row)

    def close(self) -> None:
        self._sink.close()
