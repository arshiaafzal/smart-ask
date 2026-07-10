"""Authoritative prompt-free metrics for SmartAsk conversation runs."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any
from uuid import uuid4

from ..metrics import DEFAULT_PRICE_CATALOG, RunStats, TokenUsage, price_usage
from .domain import ConversationEvent


@dataclass
class ConversationAttemptMeasurement:
    """Runtime-owned evidence for one selected executor attempt."""

    phase: str | None
    role: str
    selected_model: str

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()
        self.first_output_at: float | None = None
        self.finished_at: float | None = None
        self.actual_model: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        self.cache_read_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self.provider_cost_usd: float | None = None
        self.stop_reason: str | None = None
        self.tool_call_count = 0
        self.text_parts: list[str] = []
        self.complete = False
        self.error: str | None = None
        self.cancelled = False

    def observe(self, event: ConversationEvent) -> None:
        data = event.data
        if event.kind == "message_start":
            model = data.get("model")
            if isinstance(model, str) and model:
                self.actual_model = model
        elif event.kind == "content_start":
            block = data.get("block")
            if isinstance(block, dict):
                block_type = block.get("type")
            else:
                block_type = block.get("type") if hasattr(block, "get") else None
            if block_type in ("text", "thinking", "tool_call"):
                if self.first_output_at is None:
                    self.first_output_at = time.perf_counter()
            if block_type == "tool_call":
                self.tool_call_count += 1
        elif event.kind == "content_delta":
            delta = data.get("delta")
            if hasattr(delta, "get") and delta.get("type") == "text":
                text = delta.get("text")
                if isinstance(text, str):
                    self.text_parts.append(text)
        elif event.kind == "usage":
            for source, target in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("reasoning_tokens", "reasoning_tokens"),
                ("cache_read_tokens", "cache_read_tokens"),
                ("cache_write_tokens", "cache_write_tokens"),
            ):
                value = data.get(source)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    setattr(self, target, value)
            cost = data.get("provider_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
                self.provider_cost_usd = float(cost)
        elif event.kind == "message_delta":
            reason = data.get("stop_reason")
            if isinstance(reason, str) and reason:
                self.stop_reason = reason
        elif event.kind == "message_stop":
            self.complete = True
        elif event.kind == "error":
            message = data.get("message")
            self.error = message if isinstance(message, str) else "executor error"

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def finish(self, *, error: str | None = None, cancelled: bool = False) -> None:
        if error is not None:
            self.error = error
        self.cancelled = cancelled
        self.finished_at = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        finished = self.finished_at or time.perf_counter()
        total_tokens = (
            self.input_tokens + self.output_tokens
            if self.input_tokens is not None and self.output_tokens is not None
            else None
        )
        usage = TokenUsage(
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cached_input_tokens=self.cache_read_tokens,
            cache_write_input_tokens=self.cache_write_tokens,
        )
        quote = price_usage(
            self.actual_model or self.selected_model,
            usage,
            DEFAULT_PRICE_CATALOG,
        )
        return {
            "phase": self.phase,
            "role": self.role,
            "selected_model": self.selected_model,
            "actual_model": self.actual_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": quote.cost_usd,
            "estimated_cost_status": quote.status,
            "provider_cost_usd": self.provider_cost_usd,
            "time_to_first_token_ms": (
                None
                if self.first_output_at is None
                else (self.first_output_at - self.started_at) * 1000
            ),
            "duration_ms": (finished - self.started_at) * 1000,
            "tool_call_count": self.tool_call_count,
            "stop_reason": self.stop_reason,
            "stream_complete": self.complete,
            "error": self.error,
            "cancelled": self.cancelled,
        }


class ConversationRunMeasurement:
    """One SmartAsk conversation request with classifier and attempt evidence."""

    def __init__(
        self,
        *,
        strategy_name: str,
        strategy_digest: str,
        session_id: str | None,
        agent_id: str | None,
        parent_agent_id: str | None,
    ):
        self.run_id = uuid4().hex
        self.strategy_name = strategy_name
        self.strategy_digest = strategy_digest
        self.session_id = session_id
        self.agent_id = agent_id
        self.parent_agent_id = parent_agent_id
        self.started_at = time.perf_counter()
        self.classifier_stats: RunStats | None = None
        self.routing_events: list[dict[str, Any]] = []
        self.attempts: list[ConversationAttemptMeasurement] = []
        self.error: str | None = None
        self.cancelled = False

    def start_attempt(
        self,
        *,
        phase: str | None,
        role: str,
        model: str,
    ) -> ConversationAttemptMeasurement:
        attempt = ConversationAttemptMeasurement(phase, role, model)
        self.attempts.append(attempt)
        return attempt

    def to_dict(self) -> dict[str, Any]:
        attempts = [attempt.to_dict() for attempt in self.attempts]
        classifier = self.classifier_stats
        known_classifier_tokens = (
            classifier.known_total_tokens if classifier is not None else 0
        )
        generation_totals = [attempt["total_tokens"] for attempt in attempts]
        total_tokens = (
            None
            if any(value is None for value in generation_totals)
            else known_classifier_tokens + sum(generation_totals)
        )
        estimated_costs = [attempt["estimated_cost_usd"] for attempt in attempts]
        classifier_cost = classifier.total_cost_usd if classifier is not None else 0.0
        estimated_cost = (
            None
            if classifier_cost is None or any(value is None for value in estimated_costs)
            else classifier_cost + sum(estimated_costs)
        )
        return {
            "schema": "smart-ask.conversation-run/v1",
            "run_id": self.run_id,
            "strategy": {
                "name": self.strategy_name,
                "digest": self.strategy_digest,
            },
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "classifier": (
                None
                if classifier is None
                else {
                    "calls": classifier.interaction_count,
                    "duration_ms": classifier.duration_ms,
                    "known_total_tokens": classifier.known_total_tokens,
                    "total_tokens": classifier.total_tokens,
                    "estimated_cost_usd": classifier.total_cost_usd,
                    "errors": classifier.failed_interactions,
                }
            ),
            "routing_events": deepcopy(self.routing_events),
            "route_path": [attempt["phase"] for attempt in attempts],
            "attempts": attempts,
            "totals": {
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
                "provider_cost_usd": (
                    None
                    if any(attempt["provider_cost_usd"] is None for attempt in attempts)
                    else sum(attempt["provider_cost_usd"] for attempt in attempts)
                ),
            },
            "duration_ms": (time.perf_counter() - self.started_at) * 1000,
            "error": self.error,
            "cancelled": self.cancelled,
        }


class ConversationMetricsStore:
    """Runtime-owned request records and complete session/model aggregates."""

    def __init__(
        self,
        *,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable or None")
        self._records: list[dict[str, Any]] = []
        self._sessions: dict[str, dict[str, Any]] = {}
        self._sink = sink
        self._sink_errors: list[str] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._records))

    @property
    def sessions(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._sessions)

    @property
    def sink_errors(self) -> tuple[str, ...]:
        return tuple(self._sink_errors)

    def record(self, measurement: ConversationRunMeasurement) -> dict[str, Any]:
        record = measurement.to_dict()
        self._records.append(record)
        session_key = record["session_id"] or f"run:{record['run_id']}"
        session = self._sessions.setdefault(session_key, {
            "session_id": record["session_id"],
            "runs": 0,
            "errors": 0,
            "cancelled": 0,
            "known_total_tokens": 0,
            "token_total_complete": True,
            "by_model": {},
        })
        session["runs"] += 1
        session["errors"] += int(record["error"] is not None)
        session["cancelled"] += int(record["cancelled"])
        total_tokens = record["totals"]["total_tokens"]
        if total_tokens is None:
            session["token_total_complete"] = False
        else:
            session["known_total_tokens"] += total_tokens
        for attempt in record["attempts"]:
            model = attempt["actual_model"] or attempt["selected_model"]
            bucket = session["by_model"].setdefault(model, {
                "attempts": 0,
                "errors": 0,
                "known_total_tokens": 0,
                "missing_token_attempts": 0,
            })
            bucket["attempts"] += 1
            bucket["errors"] += int(attempt["error"] is not None)
            if attempt["total_tokens"] is None:
                bucket["missing_token_attempts"] += 1
            else:
                bucket["known_total_tokens"] += attempt["total_tokens"]
        envelope = {"run": deepcopy(record), "session": deepcopy(session)}
        if self._sink is not None:
            try:
                self._sink(deepcopy(envelope))
            except Exception as exc:
                self._sink_errors.append(f"{type(exc).__name__}: {exc}")
        return envelope
