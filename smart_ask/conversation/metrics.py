"""Content-free persistence and aggregation for canonical engine run records."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ..metrics import DEFAULT_PRICE_CATALOG, PriceCatalog, aggregate_resources
from .domain import thaw_value
from .model import RunRecord


def run_record_dict(record: RunRecord) -> dict[str, Any]:
    """Serialize a run record without introducing duplicate route projections."""

    if not isinstance(record, RunRecord):
        raise TypeError("record must be a RunRecord")
    return {
        "schema": "smart-ask.run/v2",
        "run_id": record.run_id,
        "metadata": {
            "strategy_name": record.metadata.strategy_name,
            "strategy_digest": record.metadata.strategy_digest,
            "session_id": record.metadata.session_id,
            "agent_id": record.metadata.agent_id,
            "parent_agent_id": record.metadata.parent_agent_id,
            "request_id": record.metadata.request_id,
            "extensions": thaw_value(record.metadata.extensions),
        },
        "status": record.status,
        "started_at": record.started_at,
        "duration_ms": record.duration_ms,
        "decisions": [
            {
                "decision_id": decision.decision_id,
                "sequence": decision.sequence,
                "gate": decision.gate,
                "outcome": decision.outcome,
                "reason_code": decision.reason_code,
                "selected_profile_id": decision.selected_profile_id,
                "evidence_call_ids": list(decision.evidence_call_ids),
            }
            for decision in record.decisions
        ],
        "model_calls": [
            {
                "call_id": call.call_id,
                "sequence": call.sequence,
                "profile_id": call.profile_id,
                "target_id": call.target_id,
                "selected_model": call.selected_model,
                "role": call.role,
                "phase": call.phase,
                "caused_by_decision_id": call.caused_by_decision_id,
                "provider_request_ids": list(call.provider_request_ids),
                "status": call.status,
                "error": call.error,
            }
            for call in record.model_calls
        ],
        "provider_requests": [
            {
                "provider_request_id": request.provider_request_id,
                "call_id": request.call_id,
                "sequence": request.sequence,
                "status": request.status,
                "target_id": request.target_id,
                "requested_max_output_tokens": request.requested_max_output_tokens,
                "selected_model": request.selected_model,
                "actual_model": request.actual_model,
                "input_tokens": request.input_tokens,
                "output_tokens": request.output_tokens,
                "visible_output_tokens": request.visible_output_tokens,
                "reasoning_tokens": request.reasoning_tokens,
                "cache_read_tokens": request.cache_read_tokens,
                "cache_write_tokens": request.cache_write_tokens,
                "provider_cost_usd": request.provider_cost_usd,
                "stop_reason": request.stop_reason,
                "stream_complete": request.stream_complete,
                "tool_call_count": request.tool_call_count,
                "visible_text_chars": request.visible_text_chars,
                "output_status": request.output_status,
                "time_to_first_output_ms": request.time_to_first_output_ms,
                "duration_ms": request.duration_ms,
                "error": request.error,
            }
            for request in record.provider_requests
        ],
        "final_call_id": record.final_call_id,
        "final_decision_id": record.final_decision_id,
        "error": record.error,
    }


class RunMetricsStore:
    """Store one canonical record per method invocation and session rollups."""

    def __init__(
        self,
        *,
        sink: Callable[[dict[str, Any]], None] | None = None,
        price_catalog: PriceCatalog = DEFAULT_PRICE_CATALOG,
        max_records: int = 10000,
        max_sessions: int = 1000,
        max_requests_per_session: int = 10000,
    ) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable or None")
        if not isinstance(price_catalog, PriceCatalog):
            raise TypeError("price_catalog must be a PriceCatalog")
        for name, value in (
            ("max_records", max_records),
            ("max_sessions", max_sessions),
            ("max_requests_per_session", max_requests_per_session),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._sink = sink
        self._price_catalog = price_catalog
        self._max_records = max_records
        self._max_sessions = max_sessions
        self._max_requests_per_session = max_requests_per_session
        self._records: list[dict[str, Any]] = []
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._sink_errors: list[str] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._records))

    @property
    def sessions(self) -> dict[str, dict[str, Any]]:
        return {
            key: deepcopy(self._session_public(value))
            for key, value in self._sessions.items()
        }

    @property
    def sink_errors(self) -> tuple[str, ...]:
        return tuple(self._sink_errors)

    def record(self, record: RunRecord) -> dict[str, Any]:
        serialized = run_record_dict(record)
        if self._max_records:
            self._records.append(serialized)
            if len(self._records) > self._max_records:
                del self._records[: len(self._records) - self._max_records]
        session_key = record.metadata.session_id or f"run:{record.run_id}"
        session = self._sessions.setdefault(session_key, {
            "session_id": record.metadata.session_id,
            "runs": 0,
            "errors": 0,
            "cancelled": 0,
            "completed": 0,
            "model_calls": 0,
            "routing": {"gates": {}, "transitions": {}, "paths": {}},
            "_calls": [],
            "_requests": [],
        })
        self._sessions.move_to_end(session_key)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        session["runs"] += 1
        session["errors"] += int(record.status == "error")
        session["cancelled"] += int(record.status == "cancelled")
        session["completed"] += int(record.status == "completed")
        session["model_calls"] += len(record.model_calls)
        prefix = record.run_id + ":"
        session["_calls"].extend({
            **call,
            "call_id": prefix + call["call_id"],
        } for call in serialized["model_calls"])
        session["_requests"].extend({
            **request,
            "call_id": prefix + request["call_id"],
        } for request in serialized["provider_requests"])
        overflow = len(session["_requests"]) - self._max_requests_per_session
        if overflow > 0:
            del session["_requests"][:overflow]
            retained = {request["call_id"] for request in session["_requests"]}
            session["_calls"] = [
                call for call in session["_calls"] if call["call_id"] in retained
            ]
        self._add_routing(session["routing"], serialized["decisions"])
        envelope = {
            "run": serialized,
            "session": deepcopy(self._session_public(session)),
        }
        if self._sink is not None:
            try:
                self._sink(deepcopy(envelope))
            except Exception as exc:
                self._sink_errors.append(f"{type(exc).__name__}: {exc}")
        return envelope

    def _session_public(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in session.items()
            if not key.startswith("_")
        } | {
            "resources": aggregate_resources(
                session["_requests"],
                session["_calls"],
                price_catalog=self._price_catalog,
            )
        }

    @staticmethod
    def _add_routing(
        routing: dict[str, dict[str, int]],
        decisions: list[dict[str, Any]],
    ) -> None:
        outcomes = []
        for decision in decisions:
            gate = str(decision.get("gate") or "unknown")
            outcome = str(decision.get("outcome") or "unknown")
            gate_counts = routing["gates"].setdefault(gate, {})
            gate_counts[outcome] = gate_counts.get(outcome, 0) + 1
            outcomes.append(outcome)
        path = outcomes or ["none"]
        path_key = " → ".join(path)
        routing["paths"][path_key] = routing["paths"].get(path_key, 0) + 1
        previous = "start"
        for outcome in path:
            transition = f"{previous} → {outcome}"
            routing["transitions"][transition] = (
                routing["transitions"].get(transition, 0) + 1
            )
            previous = outcome
        if previous not in ("accept", "none"):
            transition = f"{previous} → response"
            routing["transitions"][transition] = (
                routing["transitions"].get(transition, 0) + 1
            )
