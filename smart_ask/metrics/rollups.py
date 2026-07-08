"""Deterministic resource rollups derived from canonical call evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .._numeric import (
    checked_difference,
    checked_fsum,
    checked_mean,
    checked_ratio,
)
from .models import CallStats, RunStats


_UNASSIGNED_STRATEGY = "<unassigned>"
_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "visible_output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
)


@dataclass(frozen=True)
class _CallEvidence:
    run_id: str
    call_id: str
    ordinal: int
    channel: str
    role: str
    requested_model: str
    actual_model: str | None
    priced_model: str | None
    status: str
    latency_ms: float
    usage: tuple[int | None, ...]
    usage_status: str
    cost_usd: float | None
    provider_cost_usd: float | None
    cost_status: str
    finish_reason: str
    output_status: str | None
    output_empty: bool | None
    error_category: str | None
    max_tokens_reached: bool | None

    def token(self, field: str) -> int | None:
        return self.usage[_TOKEN_FIELDS.index(field)]


@dataclass(frozen=True)
class _ResourceRollup:
    calls: tuple[_CallEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        calls = self.calls
        latencies = [call.latency_ms for call in calls]
        known_tokens = {
            field: sum(
                int(value)
                for call in calls
                if (value := call.token(field)) is not None
            )
            for field in _TOKEN_FIELDS
        }
        missing_tokens = {
            field: sum(call.token(field) is None for call in calls)
            for field in _TOKEN_FIELDS
        }

        actual_model_calls = sum(call.actual_model is not None for call in calls)
        requested_model_fallback_calls = len(calls) - actual_model_calls
        priced_model_calls: dict[str, list[_CallEvidence]] = defaultdict(list)
        for call in calls:
            if call.cost_usd is not None and call.priced_model is None:
                raise ValueError(
                    "a call with known cost must identify its priced_model"
                )
            if call.priced_model is not None:
                priced_model_calls[call.priced_model].append(call)

        comparable_cost_calls = [
            call
            for call in calls
            if call.cost_usd is not None and call.provider_cost_usd is not None
        ]
        known_cost_difference = checked_fsum(
            (
                checked_difference(
                    float(call.cost_usd),
                    float(call.provider_cost_usd),
                    name="catalog/provider cost difference",
                )
                for call in comparable_cost_calls
            ),
            name="aggregate catalog/provider cost difference",
        )

        throughput_calls = [
            call
            for call in calls
            if call.token("visible_output_tokens") is not None
            and call.latency_ms > 0
        ]
        throughput_latency_ms = checked_fsum(
            (call.latency_ms for call in throughput_calls),
            name="aggregate throughput latency",
        )
        visible_tokens = sum(
            int(call.token("visible_output_tokens") or 0)
            for call in throughput_calls
        )

        return {
            "calls": len(calls),
            "call_errors": sum(call.status == "error" for call in calls),
            "model_attribution": {
                "actual_model_calls": actual_model_calls,
                "requested_model_fallback_calls": (
                    requested_model_fallback_calls
                ),
            },
            "tokens": {
                "known": known_tokens,
                "missing_calls": missing_tokens,
                "usage_error_calls": sum(
                    call.usage_status == "error" for call in calls
                ),
            },
            "cost": {
                **_cost_summary(calls),
                "pricing_error_calls": sum(
                    call.cost_status == "error" for call in calls
                ),
                "unattributed_calls": sum(
                    call.priced_model is None for call in calls
                ),
                "by_priced_model": {
                    model: _cost_summary(model_calls)
                    for model, model_calls in sorted(priced_model_calls.items())
                },
                "provider_reported": _provider_cost_summary(calls),
                "catalog_estimate_minus_provider": {
                    "known_usd": known_cost_difference,
                    "total_usd": (
                        known_cost_difference
                        if len(comparable_cost_calls) == len(calls)
                        else None
                    ),
                    "comparable_calls": len(comparable_cost_calls),
                    "missing_calls": len(calls) - len(comparable_cost_calls),
                },
            },
            "latency_ms": {
                "mean": (
                    checked_mean(latencies, name="mean call latency")
                ),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "observed_output_throughput": {
                "tokens_per_second": (
                    checked_ratio(
                        visible_tokens * 1_000,
                        throughput_latency_ms,
                        name="observed output throughput",
                    )
                    if throughput_latency_ms > 0
                    else None
                ),
                "eligible_calls": len(throughput_calls),
                "missing_calls": len(calls) - len(throughput_calls),
            },
            "responses": {
                "finish_reasons": _counts(
                    call.finish_reason for call in calls
                ),
                "output_statuses": _counts(
                    call.output_status or "unavailable" for call in calls
                ),
                "output_emptiness": _counts(
                    "unknown"
                    if call.output_empty is None
                    else "empty"
                    if call.output_empty
                    else "nonempty"
                    for call in calls
                ),
                "error_categories": _counts(
                    call.error_category
                    for call in calls
                    if call.error_category is not None
                ),
                "max_tokens_reached_calls": sum(
                    call.max_tokens_reached is True for call in calls
                ),
            },
        }


@dataclass(frozen=True)
class ResourceReport:
    """Immutable resource aggregation with a JSON-compatible projection."""

    _total: _ResourceRollup
    _by_actual_model: tuple[tuple[str, _ResourceRollup], ...]
    _by_requested_model_fallback: tuple[tuple[str, _ResourceRollup], ...]
    _by_channel: tuple[tuple[str, _ResourceRollup], ...]
    _by_role: tuple[tuple[str, _ResourceRollup], ...]
    _by_strategy: tuple[tuple[str, _ResourceRollup], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh, deterministic JSON-compatible report."""

        return {
            "total": self._total.to_dict(),
            "by_model": {
                "actual": _groups_to_dict(self._by_actual_model),
                "requested_fallback": _groups_to_dict(
                    self._by_requested_model_fallback
                ),
            },
            "by_channel": _groups_to_dict(self._by_channel),
            "by_role": _groups_to_dict(self._by_role),
            "by_strategy": _groups_to_dict(self._by_strategy),
        }


def aggregate_resources(runs: Iterable[RunStats]) -> ResourceReport:
    """Aggregate immutable run snapshots across resource dimensions."""

    try:
        snapshots = tuple(runs)
    except TypeError as exc:
        raise TypeError("runs must be an iterable of RunStats values") from exc

    for index, run in enumerate(snapshots):
        if not isinstance(run, RunStats):
            raise TypeError(f"runs[{index}] must be a RunStats value")

    _require_unique_run_ids(run.run_id for run in snapshots)
    rows = [
        (run.strategy_id or _UNASSIGNED_STRATEGY, _from_call_stats(call))
        for run in sorted(snapshots, key=lambda item: item.run_id)
        for call in run.calls
    ]
    return _aggregate_rows(rows)


def aggregate_record_resources(
    records: Iterable[Mapping[str, Any]],
) -> ResourceReport:
    """Aggregate already-validated benchmark record call ledgers.

    Only resource fields are projected from each canonical call mapping. Price
    catalogs are deliberately not reconstructed: cost attribution uses the
    call's serialized ``models.priced`` and ``cost.usd`` evidence directly.
    """

    try:
        snapshots = tuple(records)
    except TypeError as exc:
        raise TypeError("records must be an iterable of mappings") from exc

    normalized: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = []
    for index, record in enumerate(snapshots):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        try:
            strategy_id = record["strategy_id"]
            metrics = record["metrics"]
            calls = record["calls"]
            run_id = metrics["identity"]["run_id"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"records[{index}] lacks canonical benchmark identity or calls"
            ) from exc
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise TypeError(f"records[{index}].strategy_id must be non-empty text")
        if not isinstance(run_id, str) or not run_id.strip():
            raise TypeError(f"records[{index}] run_id must be non-empty text")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise TypeError(f"records[{index}].calls must be a sequence")
        for call_index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise TypeError(
                    f"records[{index}].calls[{call_index}] must be a mapping"
                )
        normalized.append((run_id, strategy_id, calls))

    _require_unique_run_ids(run_id for run_id, _, _ in normalized)
    rows: list[tuple[str, _CallEvidence]] = []
    for run_id, strategy_id, calls in sorted(normalized):
        for call_index, call in enumerate(calls):
            evidence = _from_record_call(call, call_index=call_index)
            if evidence.run_id != run_id:
                raise ValueError("record call run_id contradicts record identity")
            rows.append((strategy_id, evidence))
    return _aggregate_rows(rows)


def _from_call_stats(call: CallStats) -> _CallEvidence:
    return _CallEvidence(
        run_id=call.run_id,
        call_id=call.call_id,
        ordinal=call.ordinal,
        channel=call.channel,
        role=call.role,
        requested_model=call.requested_model,
        actual_model=call.actual_model,
        priced_model=call.priced_model,
        status=call.status,
        latency_ms=call.latency_ms,
        usage=tuple(getattr(call.usage, field) for field in _TOKEN_FIELDS),
        usage_status=call.usage_status,
        cost_usd=call.price_quote.cost_usd,
        provider_cost_usd=call.provider_cost_usd,
        cost_status=call.price_quote.status,
        finish_reason=call.finish_reason,
        output_status=call.output_status,
        output_empty=call.output_empty,
        error_category=call.error_category,
        max_tokens_reached=call.max_tokens_reached,
    )


def _from_record_call(
    call: Mapping[str, Any],
    *,
    call_index: int,
) -> _CallEvidence:
    try:
        models = call["models"]
        timing = call["timing"]
        usage = call["usage"]
        cost = call["cost"]
        response = call["response"]
        error = call["error"]
        return _CallEvidence(
            run_id=str(call["run_id"]),
            call_id=str(call["call_id"]),
            ordinal=int(call["ordinal"]),
            channel=str(call["channel"]),
            role=str(call["role"]),
            requested_model=str(models["requested"]),
            actual_model=models["actual"],
            priced_model=models["priced"],
            status=str(call["status"]),
            latency_ms=float(timing["latency_ms"]),
            usage=tuple(usage[field] for field in _TOKEN_FIELDS),
            usage_status=str(usage["status"]),
            cost_usd=cost["usd"],
            provider_cost_usd=cost["provider_reported_usd"],
            cost_status=str(cost["status"]),
            finish_reason=str(response["finish_reason"]),
            output_status=response["output_status"],
            output_empty=response["output_empty"],
            error_category=None if error is None else error["category"],
            max_tokens_reached=response["max_tokens_reached"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"record call {call_index} lacks canonical resource evidence"
        ) from exc


def _aggregate_rows(
    rows: Sequence[tuple[str, _CallEvidence]],
) -> ResourceReport:
    all_calls: list[_CallEvidence] = []
    by_actual_model: dict[str, list[_CallEvidence]] = defaultdict(list)
    by_requested_model_fallback: dict[
        str, list[_CallEvidence]
    ] = defaultdict(list)
    by_channel: dict[str, list[_CallEvidence]] = defaultdict(list)
    by_role: dict[str, list[_CallEvidence]] = defaultdict(list)
    by_strategy: dict[str, list[_CallEvidence]] = defaultdict(list)
    seen_calls: set[tuple[str, str]] = set()

    for strategy, call in rows:
        call_key = (call.run_id, call.call_id)
        if call_key in seen_calls:
            raise ValueError("call evidence must be unique across resource rollups")
        seen_calls.add(call_key)
        all_calls.append(call)
        if call.actual_model is not None:
            by_actual_model[call.actual_model].append(call)
        else:
            by_requested_model_fallback[call.requested_model].append(call)
        by_channel[call.channel].append(call)
        by_role[call.role].append(call)
        by_strategy[strategy].append(call)

    return ResourceReport(
        _total=_rollup(all_calls),
        _by_actual_model=_group_rollups(by_actual_model),
        _by_requested_model_fallback=_group_rollups(
            by_requested_model_fallback
        ),
        _by_channel=_group_rollups(by_channel),
        _by_role=_group_rollups(by_role),
        _by_strategy=_group_rollups(by_strategy),
    )


def _require_unique_run_ids(run_ids: Iterable[str]) -> None:
    values = tuple(run_ids)
    if len(set(values)) != len(values):
        raise ValueError("run_id values must be unique across resource rollups")


def _rollup(calls: Iterable[_CallEvidence]) -> _ResourceRollup:
    return _ResourceRollup(tuple(sorted(
        calls,
        key=lambda call: (call.run_id, call.ordinal, call.call_id),
    )))


def _group_rollups(
    groups: dict[str, list[_CallEvidence]],
) -> tuple[tuple[str, _ResourceRollup], ...]:
    return tuple(
        (key, _rollup(groups[key]))
        for key in sorted(groups)
    )


def _groups_to_dict(
    groups: tuple[tuple[str, _ResourceRollup], ...],
) -> dict[str, dict[str, Any]]:
    return {key: rollup.to_dict() for key, rollup in groups}


def _cost_summary(calls: Sequence[_CallEvidence]) -> dict[str, Any]:
    known_costs = sorted(
        float(call.cost_usd)
        for call in calls
        if call.cost_usd is not None
    )
    missing_calls = len(calls) - len(known_costs)
    known_usd = checked_fsum(known_costs, name="aggregate catalog cost")
    return {
        "calls": len(calls),
        "known_usd": known_usd,
        "total_usd": known_usd if missing_calls == 0 else None,
        "complete": missing_calls == 0,
        "missing_calls": missing_calls,
    }


def _provider_cost_summary(calls: Sequence[_CallEvidence]) -> dict[str, Any]:
    known_costs = sorted(
        float(call.provider_cost_usd)
        for call in calls
        if call.provider_cost_usd is not None
    )
    missing_calls = len(calls) - len(known_costs)
    known_usd = checked_fsum(
        known_costs,
        name="aggregate provider-reported cost",
    )
    return {
        "known_usd": known_usd,
        "total_usd": known_usd if missing_calls == 0 else None,
        "complete": missing_calls == 0,
        "missing_calls": missing_calls,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return {key: counts[key] for key in sorted(counts)}


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


__all__ = [
    "ResourceReport",
    "aggregate_record_resources",
    "aggregate_resources",
]
