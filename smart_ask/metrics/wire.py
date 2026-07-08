"""Serialization and strict validation for the public metrics wire schema."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any, Literal

from ..domain import FINISH_REASONS, OUTPUT_STATUSES
from .cost import PriceCatalog, PriceQuote
from .._numeric import checked_fsum, is_finite_real
from .models import (
    ERROR_CATEGORIES,
    OUTPUT_EMPTINESS,
    TASK_OUTCOMES,
    CallStats,
    RunStats,
    StatsSummary,
    TokenUsage,
)


METRICS_WIRE_SCHEMA = "smart-ask.metrics/v2"

__all__ = ["METRICS_WIRE_SCHEMA", "aggregate_metric_payloads"]


def _token_usage_to_dict(usage: TokenUsage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "visible_output_tokens": usage.visible_output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_write_input_tokens": usage.cache_write_input_tokens,
        "completeness": {
            "total": usage.total_complete,
            "breakdown": usage.breakdown_complete,
        },
    }


def _call_stats_to_dict(stats: CallStats) -> dict[str, Any]:
    usage = _token_usage_to_dict(stats.usage)
    usage.update({
        "status": stats.usage_status,
        "diagnostic": stats.usage_diagnostic,
    })
    cost = stats.price_quote.to_dict()
    cost["provider_reported_usd"] = stats.provider_cost_usd
    return {
        "run_id": stats.run_id,
        "call_id": stats.call_id,
        "ordinal": stats.ordinal,
        "channel": stats.channel,
        "role": stats.role,
        "status": stats.status,
        "telemetry_status": stats.telemetry_status,
        "models": {
            "requested": stats.requested_model,
            "actual": stats.actual_model,
            "priced": stats.priced_model,
        },
        "timing": {
            "latency_ms": stats.latency_ms,
            "started_offset_ms": stats.started_offset_ms,
        },
        "usage": usage,
        "cost": cost,
        "response": {
            "finish_reason": stats.finish_reason,
            "native_finish_reason": stats.native_finish_reason,
            "output_status": stats.output_status,
            "output_empty": stats.output_empty,
            "refusal": stats.refusal,
            "requested_max_tokens": stats.requested_max_tokens,
            "applied_max_tokens": stats.applied_max_tokens,
            "max_tokens_reached": stats.max_tokens_reached,
        },
        "error": None if stats.error_type is None else {
            "category": stats.error_category,
            "type": stats.error_type,
            "message": stats.error_message,
        },
    }


def _run_stats_to_dict(
    stats: RunStats,
    *,
    include_calls: bool = True,
) -> dict[str, Any]:
    payload = _metrics_payload(
        scope="run",
        runs=1,
        run_duration_ms=stats.duration_ms,
        cumulative_run_duration_ms=stats.duration_ms,
        interactions=stats.interaction_count,
        failed_interactions=stats.failed_interactions,
        interactions_by_channel=stats.interactions_by_channel,
        interactions_by_role=stats.interactions_by_role,
        interactions_by_requested_model=stats.interactions_by_requested_model,
        interactions_by_actual_model=stats.interactions_by_actual_model,
        interactions_by_priced_model=stats.interactions_by_priced_model,
        error_categories=stats.error_categories,
        known_prompt_tokens=stats.known_prompt_tokens,
        known_completion_tokens=stats.known_completion_tokens,
        known_total_tokens=stats.known_total_tokens,
        known_visible_output_tokens=stats.known_visible_output_tokens,
        known_reasoning_tokens=stats.known_reasoning_tokens,
        known_cached_input_tokens=stats.known_cached_input_tokens,
        known_cache_write_input_tokens=stats.known_cache_write_input_tokens,
        total_tokens=stats.total_tokens,
        missing_total_usage_calls=stats.missing_total_usage_calls,
        missing_usage_breakdown_calls=stats.missing_usage_breakdown_calls,
        missing_visible_output_usage_calls=(
            stats.missing_visible_output_usage_calls
        ),
        missing_reasoning_usage_calls=stats.missing_reasoning_usage_calls,
        missing_cached_input_usage_calls=stats.missing_cached_input_usage_calls,
        missing_cache_write_input_usage_calls=(
            stats.missing_cache_write_input_usage_calls
        ),
        usage_error_calls=stats.usage_error_calls,
        known_cost_usd=stats.known_cost_usd,
        total_cost_usd=stats.total_cost_usd,
        missing_cost_calls=stats.missing_cost_calls,
        pricing_error_calls=stats.pricing_error_calls,
        known_provider_cost_usd=stats.known_provider_cost_usd,
        total_provider_cost_usd=stats.total_provider_cost_usd,
        missing_provider_cost_calls=stats.missing_provider_cost_calls,
        priced_calls_by_source=stats.priced_calls_by_source,
        price_catalogs=stats.price_catalogs,
        generation_attempts=stats.generation_attempts,
        routing_events=stats.routing_events,
        finish_reasons=stats.finish_reasons,
        output_statuses=stats.output_statuses,
        output_emptiness=stats.output_emptiness,
        max_tokens_reached_calls=stats.max_tokens_reached_calls,
        outcome_counts={outcome: int(stats.outcome == outcome) for outcome in TASK_OUTCOMES},
    )
    payload["identity"] = {
        "run_id": stats.run_id,
        "task_id": stats.task_id,
        "strategy_id": stats.strategy_id,
    }
    if include_calls:
        payload["calls"] = [_call_stats_to_dict(call) for call in stats.calls]
    return payload


def _stats_summary_to_dict(stats: StatsSummary) -> dict[str, Any]:
    return _metrics_payload(
        scope="summary",
        runs=stats.runs,
        run_duration_ms=None,
        cumulative_run_duration_ms=stats.cumulative_run_duration_ms,
        interactions=stats.interactions,
        failed_interactions=stats.failed_interactions,
        interactions_by_channel=stats.interactions_by_channel,
        interactions_by_role=stats.interactions_by_role,
        interactions_by_requested_model=stats.interactions_by_requested_model,
        interactions_by_actual_model=stats.interactions_by_actual_model,
        interactions_by_priced_model=stats.interactions_by_priced_model,
        error_categories=stats.error_categories,
        known_prompt_tokens=stats.known_prompt_tokens,
        known_completion_tokens=stats.known_completion_tokens,
        known_total_tokens=stats.known_total_tokens,
        known_visible_output_tokens=stats.known_visible_output_tokens,
        known_reasoning_tokens=stats.known_reasoning_tokens,
        known_cached_input_tokens=stats.known_cached_input_tokens,
        known_cache_write_input_tokens=stats.known_cache_write_input_tokens,
        total_tokens=stats.total_tokens,
        missing_total_usage_calls=stats.missing_total_usage_calls,
        missing_usage_breakdown_calls=stats.missing_usage_breakdown_calls,
        missing_visible_output_usage_calls=(
            stats.missing_visible_output_usage_calls
        ),
        missing_reasoning_usage_calls=stats.missing_reasoning_usage_calls,
        missing_cached_input_usage_calls=stats.missing_cached_input_usage_calls,
        missing_cache_write_input_usage_calls=(
            stats.missing_cache_write_input_usage_calls
        ),
        usage_error_calls=stats.usage_error_calls,
        known_cost_usd=stats.known_cost_usd,
        total_cost_usd=stats.total_cost_usd,
        missing_cost_calls=stats.missing_cost_calls,
        pricing_error_calls=stats.pricing_error_calls,
        known_provider_cost_usd=stats.known_provider_cost_usd,
        total_provider_cost_usd=stats.total_provider_cost_usd,
        missing_provider_cost_calls=stats.missing_provider_cost_calls,
        priced_calls_by_source=stats.priced_calls_by_source,
        price_catalogs=stats.price_catalogs,
        generation_attempts=stats.generation_attempts,
        routing_events=stats.routing_events,
        finish_reasons=stats.finish_reasons,
        output_statuses=stats.output_statuses,
        output_emptiness=stats.output_emptiness,
        max_tokens_reached_calls=stats.max_tokens_reached_calls,
        outcome_counts=stats.outcome_counts,
    )


def _metrics_payload(
    *,
    scope: Literal["run", "summary"],
    runs: int,
    run_duration_ms: float | None,
    cumulative_run_duration_ms: float,
    interactions: int,
    failed_interactions: int,
    interactions_by_channel: dict[str, int],
    interactions_by_role: dict[str, int],
    interactions_by_requested_model: dict[str, int],
    interactions_by_actual_model: dict[str, int],
    interactions_by_priced_model: dict[str, int],
    error_categories: dict[str, int],
    known_prompt_tokens: int,
    known_completion_tokens: int,
    known_total_tokens: int,
    known_visible_output_tokens: int,
    known_reasoning_tokens: int,
    known_cached_input_tokens: int,
    known_cache_write_input_tokens: int,
    total_tokens: int | None,
    missing_total_usage_calls: int,
    missing_usage_breakdown_calls: int,
    missing_visible_output_usage_calls: int,
    missing_reasoning_usage_calls: int,
    missing_cached_input_usage_calls: int,
    missing_cache_write_input_usage_calls: int,
    usage_error_calls: int,
    known_cost_usd: float,
    total_cost_usd: float | None,
    missing_cost_calls: int,
    pricing_error_calls: int,
    known_provider_cost_usd: float,
    total_provider_cost_usd: float | None,
    missing_provider_cost_calls: int,
    priced_calls_by_source: dict[str, int],
    price_catalogs: tuple[PriceCatalog, ...],
    generation_attempts: int,
    routing_events: int,
    finish_reasons: dict[str, int],
    output_statuses: dict[str, int],
    output_emptiness: dict[str, int],
    max_tokens_reached_calls: int,
    outcome_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": METRICS_WIRE_SCHEMA,
        "scope": scope,
        "runs": runs,
        "timing": {
            "run_duration_ms": run_duration_ms,
            "cumulative_run_duration_ms": cumulative_run_duration_ms,
        },
        "interactions": {
            "total": interactions,
            "failed": failed_interactions,
            "by_channel": interactions_by_channel,
            "by_role": interactions_by_role,
            "by_requested_model": interactions_by_requested_model,
            "by_actual_model": interactions_by_actual_model,
            "by_priced_model": interactions_by_priced_model,
            "errors_by_category": error_categories,
        },
        "usage": {
            "known": {
                "prompt_tokens": known_prompt_tokens,
                "completion_tokens": known_completion_tokens,
                "total_tokens": known_total_tokens,
                "visible_output_tokens": known_visible_output_tokens,
                "reasoning_tokens": known_reasoning_tokens,
                "cached_input_tokens": known_cached_input_tokens,
                "cache_write_input_tokens": known_cache_write_input_tokens,
            },
            "total_tokens": total_tokens,
            "completeness": {
                "total": missing_total_usage_calls == 0,
                "breakdown": missing_usage_breakdown_calls == 0,
                "missing_total_calls": missing_total_usage_calls,
                "missing_breakdown_calls": missing_usage_breakdown_calls,
                "error_calls": usage_error_calls,
                "details": {
                    "visible_output_tokens": {
                        "complete": missing_visible_output_usage_calls == 0,
                        "missing_calls": missing_visible_output_usage_calls,
                    },
                    "reasoning_tokens": {
                        "complete": missing_reasoning_usage_calls == 0,
                        "missing_calls": missing_reasoning_usage_calls,
                    },
                    "cached_input_tokens": {
                        "complete": missing_cached_input_usage_calls == 0,
                        "missing_calls": missing_cached_input_usage_calls,
                    },
                    "cache_write_input_tokens": {
                        "complete": missing_cache_write_input_usage_calls == 0,
                        "missing_calls": missing_cache_write_input_usage_calls,
                    },
                },
            },
        },
        "cost": {
            "known_usd": known_cost_usd,
            "total_usd": total_cost_usd,
            "completeness": {
                "complete": missing_cost_calls == 0,
                "missing_calls": missing_cost_calls,
                "error_calls": pricing_error_calls,
            },
            "priced_calls_by_source": priced_calls_by_source,
            "catalogs": [catalog.to_dict() for catalog in price_catalogs],
            "provider_reported": {
                "known_usd": known_provider_cost_usd,
                "total_usd": total_provider_cost_usd,
                "complete": missing_provider_cost_calls == 0,
                "missing_calls": missing_provider_cost_calls,
            },
        },
        "routing": {
            "generation_attempts": generation_attempts,
            "events": routing_events,
        },
        "responses": {
            "finish_reasons": finish_reasons,
            "output_statuses": output_statuses,
            "output_emptiness": output_emptiness,
            "max_tokens_reached_calls": max_tokens_reached_calls,
        },
        "outcomes": outcome_counts,
    }


def aggregate_metric_payloads(
    payloads: Iterable[Mapping[str, Any]],
) -> StatsSummary:
    """Validate and aggregate serialized ``scope='run'`` metric envelopes."""

    if isinstance(payloads, (str, bytes, Mapping)):
        raise TypeError("payloads must be an iterable of metric mappings")
    try:
        snapshots = tuple(payloads)
    except TypeError as exc:
        raise TypeError("payloads must be an iterable of metric mappings") from exc
    runs = tuple(
        _validate_run_metric_payload(payload, index)
        for index, payload in enumerate(snapshots, start=1)
    )
    duplicate_run_ids = sorted(
        run_id
        for run_id, count in Counter(run["run_id"] for run in runs).items()
        if count > 1
    )
    if duplicate_run_ids:
        raise ValueError(
            "run_id values must be unique; duplicates: "
            + ", ".join(duplicate_run_ids)
        )
    runs = tuple(sorted(runs, key=lambda run: run["run_id"]))

    def merge_counts(key: str) -> tuple[tuple[str, int], ...]:
        counts: Counter[str] = Counter()
        for run in runs:
            counts.update(run[key])
        return tuple(sorted(counts.items()))

    catalogs: dict[str, PriceCatalog] = {}
    for run in runs:
        for catalog in run["price_catalogs"]:
            previous = catalogs.setdefault(catalog.catalog_id, catalog)
            if previous != catalog:
                raise ValueError(
                    f"conflicting price catalogs share id {catalog.catalog_id!r}"
                )

    return StatsSummary(
        runs=len(runs),
        interactions=sum(run["interactions"] for run in runs),
        failed_interactions=sum(run["failed_interactions"] for run in runs),
        interactions_by_channel_items=merge_counts("interactions_by_channel"),
        interactions_by_role_items=merge_counts("interactions_by_role"),
        interactions_by_requested_model_items=merge_counts(
            "interactions_by_requested_model"
        ),
        interactions_by_actual_model_items=merge_counts(
            "interactions_by_actual_model"
        ),
        interactions_by_priced_model_items=merge_counts(
            "interactions_by_priced_model"
        ),
        generation_attempts=sum(run["generation_attempts"] for run in runs),
        routing_events=sum(run["routing_events"] for run in runs),
        known_prompt_tokens=sum(run["known_prompt_tokens"] for run in runs),
        known_completion_tokens=sum(
            run["known_completion_tokens"] for run in runs
        ),
        known_total_tokens=sum(run["known_total_tokens"] for run in runs),
        known_visible_output_tokens=sum(
            run["known_visible_output_tokens"] for run in runs
        ),
        known_reasoning_tokens=sum(
            run["known_reasoning_tokens"] for run in runs
        ),
        known_cached_input_tokens=sum(
            run["known_cached_input_tokens"] for run in runs
        ),
        known_cache_write_input_tokens=sum(
            run["known_cache_write_input_tokens"] for run in runs
        ),
        missing_total_usage_calls=sum(
            run["missing_total_usage_calls"] for run in runs
        ),
        missing_usage_breakdown_calls=sum(
            run["missing_usage_breakdown_calls"] for run in runs
        ),
        missing_visible_output_usage_calls=sum(
            run["missing_visible_output_usage_calls"] for run in runs
        ),
        missing_reasoning_usage_calls=sum(
            run["missing_reasoning_usage_calls"] for run in runs
        ),
        missing_cached_input_usage_calls=sum(
            run["missing_cached_input_usage_calls"] for run in runs
        ),
        missing_cache_write_input_usage_calls=sum(
            run["missing_cache_write_input_usage_calls"] for run in runs
        ),
        usage_error_calls=sum(run["usage_error_calls"] for run in runs),
        known_cost_usd=checked_fsum(
            (run["known_cost_usd"] for run in runs),
            name="wire aggregate catalog cost",
        ),
        missing_cost_calls=sum(run["missing_cost_calls"] for run in runs),
        pricing_error_calls=sum(run["pricing_error_calls"] for run in runs),
        known_provider_cost_usd=checked_fsum(
            (run["known_provider_cost_usd"] for run in runs),
            name="wire aggregate provider-reported cost",
        ),
        missing_provider_cost_calls=sum(
            run["missing_provider_cost_calls"] for run in runs
        ),
        priced_calls_by_source_items=merge_counts("priced_calls_by_source"),
        finish_reasons_items=merge_counts("finish_reasons"),
        output_statuses_items=merge_counts("output_statuses"),
        output_emptiness_items=merge_counts("output_emptiness"),
        error_categories_items=merge_counts("error_categories"),
        outcome_counts_items=merge_counts("outcome_counts"),
        max_tokens_reached_calls=sum(
            run["max_tokens_reached_calls"] for run in runs
        ),
        price_catalogs=tuple(catalogs[key] for key in sorted(catalogs)),
        cumulative_run_duration_ms=checked_fsum(
            (run["run_duration_ms"] for run in runs),
            name="wire cumulative run duration",
        ),
    )


def _validate_run_metric_payload(
    payload: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    path = f"metrics payload {index}"
    root = _object(payload, path)
    _keys(
        root,
        path,
        required={
            "schema",
            "scope",
            "runs",
            "timing",
            "interactions",
            "usage",
            "cost",
            "routing",
            "responses",
            "outcomes",
            "identity",
        },
        optional={"calls"},
    )
    if root["schema"] != METRICS_WIRE_SCHEMA:
        raise ValueError(f"{path}.schema must be {METRICS_WIRE_SCHEMA!r}")
    if root["scope"] != "run":
        raise ValueError(f"{path}.scope must be 'run'")
    if _integer(root["runs"], f"{path}.runs") != 1:
        raise ValueError(f"{path}.runs must be 1")
    timing = _object(root["timing"], f"{path}.timing")
    _keys(
        timing,
        f"{path}.timing",
        required={"run_duration_ms", "cumulative_run_duration_ms"},
    )
    run_duration_ms = _number(
        timing["run_duration_ms"],
        f"{path}.timing.run_duration_ms",
    )
    cumulative_run_duration_ms = _number(
        timing["cumulative_run_duration_ms"],
        f"{path}.timing.cumulative_run_duration_ms",
    )
    if cumulative_run_duration_ms != run_duration_ms:
        raise ValueError(
            f"{path}.timing cumulative duration must equal its one run"
        )

    identity = _object(root["identity"], f"{path}.identity")
    _keys(
        identity,
        f"{path}.identity",
        required={"run_id", "task_id", "strategy_id"},
    )
    run_id = _string(identity["run_id"], f"{path}.identity.run_id")
    task_id = _optional_string(identity["task_id"], f"{path}.identity.task_id")
    strategy_id = _optional_string(
        identity["strategy_id"],
        f"{path}.identity.strategy_id",
    )

    interactions = _object(root["interactions"], f"{path}.interactions")
    _keys(
        interactions,
        f"{path}.interactions",
        required={
            "total",
            "failed",
            "by_channel",
            "by_role",
            "by_requested_model",
            "by_actual_model",
            "by_priced_model",
            "errors_by_category",
        },
    )
    interaction_count = _integer(
        interactions["total"],
        f"{path}.interactions.total",
    )
    failed_interactions = _integer(
        interactions["failed"],
        f"{path}.interactions.failed",
    )
    if failed_interactions > interaction_count:
        raise ValueError(f"{path}.interactions.failed exceeds total")
    interactions_by_channel = _counter(
        interactions["by_channel"],
        f"{path}.interactions.by_channel",
    )
    interactions_by_role = _counter(
        interactions["by_role"],
        f"{path}.interactions.by_role",
    )
    interactions_by_requested_model = _counter(
        interactions["by_requested_model"],
        f"{path}.interactions.by_requested_model",
    )
    interactions_by_actual_model = _counter(
        interactions["by_actual_model"],
        f"{path}.interactions.by_actual_model",
    )
    interactions_by_priced_model = _counter(
        interactions["by_priced_model"],
        f"{path}.interactions.by_priced_model",
    )
    error_categories = _enum_counter(
        interactions["errors_by_category"],
        f"{path}.interactions.errors_by_category",
        ERROR_CATEGORIES,
    )
    for name, counts in (
        ("by_channel", interactions_by_channel),
        ("by_role", interactions_by_role),
        ("by_requested_model", interactions_by_requested_model),
    ):
        if sum(counts.values()) != interaction_count:
            raise ValueError(f"{path}.interactions.{name} does not sum to total")
    if sum(interactions_by_actual_model.values()) > (
        interaction_count - failed_interactions
    ):
        raise ValueError(
            f"{path}.interactions.by_actual_model exceeds successful calls"
        )
    if sum(interactions_by_priced_model.values()) != (
        interaction_count - failed_interactions
    ):
        raise ValueError(
            f"{path}.interactions.by_priced_model must count successful calls"
        )
    if sum(error_categories.values()) != failed_interactions:
        raise ValueError(
            f"{path}.interactions.errors_by_category must count failed calls"
        )

    usage = _object(root["usage"], f"{path}.usage")
    _keys(
        usage,
        f"{path}.usage",
        required={"known", "total_tokens", "completeness"},
    )
    known = _object(usage["known"], f"{path}.usage.known")
    _keys(
        known,
        f"{path}.usage.known",
        required={
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "visible_output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
        },
    )
    known_prompt_tokens = _integer(
        known["prompt_tokens"],
        f"{path}.usage.known.prompt_tokens",
    )
    known_completion_tokens = _integer(
        known["completion_tokens"],
        f"{path}.usage.known.completion_tokens",
    )
    known_total_tokens = _integer(
        known["total_tokens"],
        f"{path}.usage.known.total_tokens",
    )
    known_visible_output_tokens = _integer(
        known["visible_output_tokens"],
        f"{path}.usage.known.visible_output_tokens",
    )
    known_reasoning_tokens = _integer(
        known["reasoning_tokens"],
        f"{path}.usage.known.reasoning_tokens",
    )
    known_cached_input_tokens = _integer(
        known["cached_input_tokens"],
        f"{path}.usage.known.cached_input_tokens",
    )
    known_cache_write_input_tokens = _integer(
        known["cache_write_input_tokens"],
        f"{path}.usage.known.cache_write_input_tokens",
    )
    usage_completeness = _object(
        usage["completeness"],
        f"{path}.usage.completeness",
    )
    _keys(
        usage_completeness,
        f"{path}.usage.completeness",
        required={
            "total",
            "breakdown",
            "missing_total_calls",
            "missing_breakdown_calls",
            "error_calls",
            "details",
        },
    )
    missing_total_usage_calls = _integer(
        usage_completeness["missing_total_calls"],
        f"{path}.usage.completeness.missing_total_calls",
    )
    missing_usage_breakdown_calls = _integer(
        usage_completeness["missing_breakdown_calls"],
        f"{path}.usage.completeness.missing_breakdown_calls",
    )
    usage_error_calls = _integer(
        usage_completeness["error_calls"],
        f"{path}.usage.completeness.error_calls",
    )
    details = _object(
        usage_completeness["details"],
        f"{path}.usage.completeness.details",
    )
    detail_fields = (
        "visible_output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
    )
    _keys(
        details,
        f"{path}.usage.completeness.details",
        required=set(detail_fields),
    )
    detail_missing: dict[str, int] = {}
    for field in detail_fields:
        detail_path = f"{path}.usage.completeness.details.{field}"
        detail = _object(details[field], detail_path)
        _keys(detail, detail_path, required={"complete", "missing_calls"})
        missing = _integer(detail["missing_calls"], f"{detail_path}.missing_calls")
        complete = _boolean(detail["complete"], f"{detail_path}.complete")
        if missing > interaction_count:
            raise ValueError(f"{detail_path}.missing_calls exceeds interactions")
        if complete != (missing == 0):
            raise ValueError(f"{detail_path}.complete is contradictory")
        detail_missing[field] = missing
    if max(
        missing_total_usage_calls,
        missing_usage_breakdown_calls,
        usage_error_calls,
    ) > interaction_count:
        raise ValueError(f"{path}.usage completeness count exceeds interactions")
    if missing_total_usage_calls > missing_usage_breakdown_calls:
        raise ValueError(
            f"{path}.usage missing total calls must also lack a breakdown"
        )
    if usage_error_calls > missing_total_usage_calls:
        raise ValueError(
            f"{path}.usage error calls must lack total-token evidence"
        )
    total_complete = _boolean(
        usage_completeness["total"],
        f"{path}.usage.completeness.total",
    )
    breakdown_complete = _boolean(
        usage_completeness["breakdown"],
        f"{path}.usage.completeness.breakdown",
    )
    if total_complete != (missing_total_usage_calls == 0):
        raise ValueError(f"{path}.usage total completeness is contradictory")
    if breakdown_complete != (missing_usage_breakdown_calls == 0):
        raise ValueError(f"{path}.usage breakdown completeness is contradictory")
    if total_complete:
        if _integer(usage["total_tokens"], f"{path}.usage.total_tokens") != (
            known_total_tokens
        ):
            raise ValueError(f"{path}.usage.total_tokens contradicts known total")
    elif usage["total_tokens"] is not None:
        raise ValueError(f"{path}.usage.total_tokens must be null when incomplete")
    if breakdown_complete and known_total_tokens != (
        known_prompt_tokens + known_completion_tokens
    ):
        raise ValueError(f"{path}.usage known token breakdown is contradictory")

    cost = _object(root["cost"], f"{path}.cost")
    _keys(
        cost,
        f"{path}.cost",
        required={
            "known_usd",
            "total_usd",
            "completeness",
            "priced_calls_by_source",
            "catalogs",
            "provider_reported",
        },
    )
    known_cost_usd = _number(cost["known_usd"], f"{path}.cost.known_usd")
    cost_completeness = _object(
        cost["completeness"],
        f"{path}.cost.completeness",
    )
    _keys(
        cost_completeness,
        f"{path}.cost.completeness",
        required={"complete", "missing_calls", "error_calls"},
    )
    missing_cost_calls = _integer(
        cost_completeness["missing_calls"],
        f"{path}.cost.completeness.missing_calls",
    )
    pricing_error_calls = _integer(
        cost_completeness["error_calls"],
        f"{path}.cost.completeness.error_calls",
    )
    if max(missing_cost_calls, pricing_error_calls) > interaction_count:
        raise ValueError(f"{path}.cost completeness count exceeds interactions")
    if pricing_error_calls > missing_cost_calls:
        raise ValueError(f"{path}.cost error calls must have missing cost")
    cost_complete = _boolean(
        cost_completeness["complete"],
        f"{path}.cost.completeness.complete",
    )
    if cost_complete != (missing_cost_calls == 0):
        raise ValueError(f"{path}.cost completeness is contradictory")
    if cost_complete:
        total_cost_usd = _number(cost["total_usd"], f"{path}.cost.total_usd")
        if total_cost_usd != known_cost_usd:
            raise ValueError(f"{path}.cost.total_usd contradicts known_usd")
    elif cost["total_usd"] is not None:
        raise ValueError(f"{path}.cost.total_usd must be null when incomplete")
    priced_calls_by_source = _counter(
        cost["priced_calls_by_source"],
        f"{path}.cost.priced_calls_by_source",
    )
    if sum(priced_calls_by_source.values()) != (
        interaction_count - missing_cost_calls
    ):
        raise ValueError(
            f"{path}.cost.priced_calls_by_source must count priced calls"
        )
    provider_cost = _object(
        cost["provider_reported"],
        f"{path}.cost.provider_reported",
    )
    _keys(
        provider_cost,
        f"{path}.cost.provider_reported",
        required={"known_usd", "total_usd", "complete", "missing_calls"},
    )
    known_provider_cost_usd = _number(
        provider_cost["known_usd"],
        f"{path}.cost.provider_reported.known_usd",
    )
    missing_provider_cost_calls = _integer(
        provider_cost["missing_calls"],
        f"{path}.cost.provider_reported.missing_calls",
    )
    if missing_provider_cost_calls > interaction_count:
        raise ValueError(
            f"{path}.cost.provider_reported.missing_calls exceeds interactions"
        )
    provider_complete = _boolean(
        provider_cost["complete"],
        f"{path}.cost.provider_reported.complete",
    )
    if provider_complete != (missing_provider_cost_calls == 0):
        raise ValueError(
            f"{path}.cost.provider_reported completeness is contradictory"
        )
    if provider_complete:
        total_provider_cost_usd = _number(
            provider_cost["total_usd"],
            f"{path}.cost.provider_reported.total_usd",
        )
        if total_provider_cost_usd != known_provider_cost_usd:
            raise ValueError(
                f"{path}.cost.provider_reported total contradicts known cost"
            )
    elif provider_cost["total_usd"] is not None:
        raise ValueError(
            f"{path}.cost.provider_reported.total_usd must be null when incomplete"
        )
    raw_catalogs = cost["catalogs"]
    if not isinstance(raw_catalogs, list):
        raise TypeError(f"{path}.cost.catalogs must be a list")
    price_catalogs: list[PriceCatalog] = []
    catalog_ids: set[str] = set()
    for catalog_index, raw_catalog in enumerate(raw_catalogs, start=1):
        catalog_path = f"{path}.cost.catalogs[{catalog_index}]"
        catalog_data = _object(raw_catalog, catalog_path)
        _keys(
            catalog_data,
            catalog_path,
            required={"catalog_id", "effective_date", "source", "prices"},
        )
        catalog = PriceCatalog(
            catalog_id=_string(
                catalog_data["catalog_id"],
                f"{catalog_path}.catalog_id",
            ),
            effective_date=_string(
                catalog_data["effective_date"],
                f"{catalog_path}.effective_date",
            ),
            source=_string(catalog_data["source"], f"{catalog_path}.source"),
            prices=_object(catalog_data["prices"], f"{catalog_path}.prices"),
        )
        if catalog.catalog_id in catalog_ids:
            raise ValueError(f"{path} contains duplicate price catalog ids")
        catalog_ids.add(catalog.catalog_id)
        price_catalogs.append(catalog)

    routing = _object(root["routing"], f"{path}.routing")
    _keys(
        routing,
        f"{path}.routing",
        required={"generation_attempts", "events"},
    )
    generation_attempts = _integer(
        routing["generation_attempts"],
        f"{path}.routing.generation_attempts",
    )
    routing_events = _integer(routing["events"], f"{path}.routing.events")
    if generation_attempts > interaction_count:
        raise ValueError(
            f"{path}.routing.generation_attempts exceeds interactions"
        )

    responses = _object(root["responses"], f"{path}.responses")
    _keys(
        responses,
        f"{path}.responses",
        required={
            "finish_reasons",
            "output_statuses",
            "output_emptiness",
            "max_tokens_reached_calls",
        },
    )
    finish_reasons = _enum_counter(
        responses["finish_reasons"],
        f"{path}.responses.finish_reasons",
        FINISH_REASONS,
    )
    output_statuses = _enum_counter(
        responses["output_statuses"],
        f"{path}.responses.output_statuses",
        OUTPUT_STATUSES,
    )
    output_emptiness = _enum_counter(
        responses["output_emptiness"],
        f"{path}.responses.output_emptiness",
        OUTPUT_EMPTINESS,
    )
    max_tokens_reached_calls = _integer(
        responses["max_tokens_reached_calls"],
        f"{path}.responses.max_tokens_reached_calls",
    )
    if sum(finish_reasons.values()) != interaction_count:
        raise ValueError(f"{path}.responses.finish_reasons must count calls")
    if sum(output_statuses.values()) != interaction_count:
        raise ValueError(f"{path}.responses.output_statuses must count calls")
    if sum(output_emptiness.values()) != interaction_count:
        raise ValueError(f"{path}.responses.output_emptiness must count calls")
    if output_statuses.get("unavailable", 0) != output_emptiness.get(
        "unknown",
        0,
    ):
        raise ValueError(
            f"{path}.responses unavailable status contradicts output emptiness"
        )
    if output_statuses.get("empty", 0) > output_emptiness.get("empty", 0):
        raise ValueError(
            f"{path}.responses empty status contradicts output emptiness"
        )
    if output_statuses.get("usable", 0) > output_emptiness.get(
        "nonempty",
        0,
    ):
        raise ValueError(
            f"{path}.responses usable status contradicts output emptiness"
        )
    if max_tokens_reached_calls > interaction_count:
        raise ValueError(
            f"{path}.responses.max_tokens_reached_calls exceeds interactions"
        )
    if max_tokens_reached_calls != finish_reasons.get("length", 0):
        raise ValueError(
            f"{path}.responses.max_tokens_reached_calls must match length "
            "finish reasons"
        )

    outcome_counts = _outcome_counter(root["outcomes"], f"{path}.outcomes")
    if sum(outcome_counts.values()) != 1:
        raise ValueError(f"{path}.outcomes must identify exactly one outcome")
    outcome = next(key for key, count in outcome_counts.items() if count == 1)

    if "calls" in root:
        calls = root["calls"]
        if not isinstance(calls, list):
            raise TypeError(f"{path}.calls must be a list")
        if len(calls) != interaction_count:
            raise ValueError(f"{path}.calls length must match interactions.total")
        catalogs_by_id = {
            catalog.catalog_id: catalog for catalog in price_catalogs
        }
        call_stats = tuple(
            _validate_call_metric_payload(
                call,
                call_index,
                path,
                run_id,
                catalogs_by_id,
            )
            for call_index, call in enumerate(calls, start=1)
        )
        reconstructed = RunStats(
            run_id=run_id,
            task_id=task_id,
            strategy_id=strategy_id,
            duration_ms=run_duration_ms,
            calls=call_stats,
            generation_attempts=generation_attempts,
            routing_events=routing_events,
            outcome=outcome,
        )
        expected_envelope = _run_stats_to_dict(
            reconstructed,
            include_calls=False,
        )
        observed_envelope = dict(root)
        observed_envelope.pop("calls")
        if observed_envelope != expected_envelope:
            raise ValueError(f"{path}.calls contradict the metric envelope")

    return {
        "run_id": run_id,
        "run_duration_ms": run_duration_ms,
        "interactions": interaction_count,
        "failed_interactions": failed_interactions,
        "interactions_by_channel": interactions_by_channel,
        "interactions_by_role": interactions_by_role,
        "interactions_by_requested_model": interactions_by_requested_model,
        "interactions_by_actual_model": interactions_by_actual_model,
        "interactions_by_priced_model": interactions_by_priced_model,
        "error_categories": error_categories,
        "known_prompt_tokens": known_prompt_tokens,
        "known_completion_tokens": known_completion_tokens,
        "known_total_tokens": known_total_tokens,
        "known_visible_output_tokens": known_visible_output_tokens,
        "known_reasoning_tokens": known_reasoning_tokens,
        "known_cached_input_tokens": known_cached_input_tokens,
        "known_cache_write_input_tokens": known_cache_write_input_tokens,
        "missing_total_usage_calls": missing_total_usage_calls,
        "missing_usage_breakdown_calls": missing_usage_breakdown_calls,
        "missing_visible_output_usage_calls": detail_missing[
            "visible_output_tokens"
        ],
        "missing_reasoning_usage_calls": detail_missing["reasoning_tokens"],
        "missing_cached_input_usage_calls": detail_missing[
            "cached_input_tokens"
        ],
        "missing_cache_write_input_usage_calls": detail_missing[
            "cache_write_input_tokens"
        ],
        "usage_error_calls": usage_error_calls,
        "known_cost_usd": known_cost_usd,
        "missing_cost_calls": missing_cost_calls,
        "pricing_error_calls": pricing_error_calls,
        "known_provider_cost_usd": known_provider_cost_usd,
        "missing_provider_cost_calls": missing_provider_cost_calls,
        "priced_calls_by_source": priced_calls_by_source,
        "finish_reasons": finish_reasons,
        "output_statuses": output_statuses,
        "output_emptiness": output_emptiness,
        "max_tokens_reached_calls": max_tokens_reached_calls,
        "outcome_counts": {
            key: count for key, count in outcome_counts.items() if count
        },
        "price_catalogs": tuple(price_catalogs),
        "generation_attempts": generation_attempts,
        "routing_events": routing_events,
    }


def _validate_call_metric_payload(
    value: Any,
    index: int,
    run_path: str,
    expected_run_id: str,
    catalogs_by_id: Mapping[str, PriceCatalog],
) -> CallStats:
    path = f"{run_path}.calls[{index}]"
    call = _object(value, path)
    _keys(
        call,
        path,
        required={
            "run_id",
            "call_id",
            "ordinal",
            "channel",
            "role",
            "status",
            "telemetry_status",
            "models",
            "timing",
            "usage",
            "cost",
            "response",
            "error",
        },
    )

    models = _object(call["models"], f"{path}.models")
    _keys(
        models,
        f"{path}.models",
        required={"requested", "actual", "priced"},
    )
    timing = _object(call["timing"], f"{path}.timing")
    _keys(
        timing,
        f"{path}.timing",
        required={"latency_ms", "started_offset_ms"},
    )

    usage_data = _object(call["usage"], f"{path}.usage")
    _keys(
        usage_data,
        f"{path}.usage",
        required={
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "visible_output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "completeness",
            "status",
            "diagnostic",
        },
    )
    usage = TokenUsage(
        prompt_tokens=_optional_integer(
            usage_data["prompt_tokens"],
            f"{path}.usage.prompt_tokens",
        ),
        completion_tokens=_optional_integer(
            usage_data["completion_tokens"],
            f"{path}.usage.completion_tokens",
        ),
        total_tokens=_optional_integer(
            usage_data["total_tokens"],
            f"{path}.usage.total_tokens",
        ),
        visible_output_tokens=_optional_integer(
            usage_data["visible_output_tokens"],
            f"{path}.usage.visible_output_tokens",
        ),
        reasoning_tokens=_optional_integer(
            usage_data["reasoning_tokens"],
            f"{path}.usage.reasoning_tokens",
        ),
        cached_input_tokens=_optional_integer(
            usage_data["cached_input_tokens"],
            f"{path}.usage.cached_input_tokens",
        ),
        cache_write_input_tokens=_optional_integer(
            usage_data["cache_write_input_tokens"],
            f"{path}.usage.cache_write_input_tokens",
        ),
    )
    completeness = _object(
        usage_data["completeness"],
        f"{path}.usage.completeness",
    )
    _keys(
        completeness,
        f"{path}.usage.completeness",
        required={"total", "breakdown"},
    )
    observed_completeness = {
        "total": _boolean(
            completeness["total"],
            f"{path}.usage.completeness.total",
        ),
        "breakdown": _boolean(
            completeness["breakdown"],
            f"{path}.usage.completeness.breakdown",
        ),
    }
    expected_completeness = {
        "total": usage.total_complete,
        "breakdown": usage.breakdown_complete,
    }
    if observed_completeness != expected_completeness:
        raise ValueError(f"{path}.usage.completeness contradicts token evidence")

    cost_data = _object(call["cost"], f"{path}.cost")
    _keys(
        cost_data,
        f"{path}.cost",
        required={
            "usd",
            "status",
            "source",
            "catalog_id",
            "diagnostic",
            "provider_reported_usd",
        },
    )
    catalog_id = _optional_string(
        cost_data["catalog_id"],
        f"{path}.cost.catalog_id",
    )
    catalog = None
    if catalog_id is not None:
        try:
            catalog = catalogs_by_id[catalog_id]
        except KeyError as exc:
            raise ValueError(
                f"{path}.cost.catalog_id references an absent catalog"
            ) from exc
    quote = PriceQuote(
        cost_usd=_optional_number(cost_data["usd"], f"{path}.cost.usd"),
        status=_string(cost_data["status"], f"{path}.cost.status"),
        source=_optional_string(cost_data["source"], f"{path}.cost.source"),
        catalog=catalog,
        diagnostic=_optional_text(
            cost_data["diagnostic"],
            f"{path}.cost.diagnostic",
        ),
    )

    response_data = _object(call["response"], f"{path}.response")
    _keys(
        response_data,
        f"{path}.response",
        required={
            "finish_reason",
            "native_finish_reason",
            "output_status",
            "output_empty",
            "refusal",
            "requested_max_tokens",
            "applied_max_tokens",
            "max_tokens_reached",
        },
    )

    error_data = call["error"]
    if error_data is None:
        error_category = None
        error_type = None
        error_message = None
    else:
        error = _object(error_data, f"{path}.error")
        _keys(
            error,
            f"{path}.error",
            required={"category", "type", "message"},
        )
        error_category = _string(
            error["category"],
            f"{path}.error.category",
        )
        error_type = _string(error["type"], f"{path}.error.type")
        error_message = _text(error["message"], f"{path}.error.message")

    stats = CallStats(
        run_id=_string(call["run_id"], f"{path}.run_id"),
        call_id=_string(call["call_id"], f"{path}.call_id"),
        ordinal=_integer(call["ordinal"], f"{path}.ordinal"),
        channel=_string(call["channel"], f"{path}.channel"),
        role=_string(call["role"], f"{path}.role"),
        requested_model=_string(
            models["requested"],
            f"{path}.models.requested",
        ),
        actual_model=_optional_string(
            models["actual"],
            f"{path}.models.actual",
        ),
        priced_model=_optional_string(
            models["priced"],
            f"{path}.models.priced",
        ),
        status=_string(call["status"], f"{path}.status"),
        latency_ms=_number(timing["latency_ms"], f"{path}.timing.latency_ms"),
        started_offset_ms=_number(
            timing["started_offset_ms"],
            f"{path}.timing.started_offset_ms",
        ),
        usage=usage,
        usage_status=_string(usage_data["status"], f"{path}.usage.status"),
        usage_diagnostic=_optional_text(
            usage_data["diagnostic"],
            f"{path}.usage.diagnostic",
        ),
        price_quote=quote,
        finish_reason=_string(
            response_data["finish_reason"],
            f"{path}.response.finish_reason",
        ),
        output_status=_optional_string(
            response_data["output_status"],
            f"{path}.response.output_status",
        ),
        output_empty=_optional_boolean(
            response_data["output_empty"],
            f"{path}.response.output_empty",
        ),
        requested_max_tokens=_optional_positive_integer(
            response_data["requested_max_tokens"],
            f"{path}.response.requested_max_tokens",
        ),
        applied_max_tokens=_optional_positive_integer(
            response_data["applied_max_tokens"],
            f"{path}.response.applied_max_tokens",
        ),
        max_tokens_reached=_optional_boolean(
            response_data["max_tokens_reached"],
            f"{path}.response.max_tokens_reached",
        ),
        provider_cost_usd=_optional_number(
            cost_data["provider_reported_usd"],
            f"{path}.cost.provider_reported_usd",
        ),
        native_finish_reason=_optional_text(
            response_data["native_finish_reason"],
            f"{path}.response.native_finish_reason",
        ),
        refusal=_optional_text(
            response_data["refusal"],
            f"{path}.response.refusal",
        ),
        error_category=error_category,
        error_type=error_type,
        error_message=error_message,
    )
    if stats.run_id != expected_run_id:
        raise ValueError(f"{path}.run_id must match the containing run")
    telemetry_status = _string(
        call["telemetry_status"],
        f"{path}.telemetry_status",
    )
    if telemetry_status != stats.telemetry_status:
        raise ValueError(f"{path}.telemetry_status contradicts telemetry evidence")
    if dict(call) != _call_stats_to_dict(stats):
        raise ValueError(f"{path} is not a canonical call metrics object")
    return stats


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(extra))}")


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise TypeError(f"{path} must be a non-negative integer")
    return int(value)


def _number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not is_finite_real(value)
        or value < 0
    ):
        raise TypeError(f"{path} must be a finite non-negative number")
    return float(value)


def _optional_integer(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _optional_positive_integer(value: Any, path: str) -> int | None:
    if value is None:
        return None
    normalized = _integer(value, path)
    if normalized < 1:
        raise TypeError(f"{path} must be a positive integer")
    return normalized


def _optional_number(value: Any, path: str) -> float | None:
    return None if value is None else _number(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _optional_boolean(value: Any, path: str) -> bool | None:
    return None if value is None else _boolean(value, path)


def _string(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise TypeError(f"{path} must be a non-empty trimmed string")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _optional_text(value: Any, path: str) -> str | None:
    return None if value is None else _text(value, path)


def _counter(value: Any, path: str) -> dict[str, int]:
    raw = _object(value, path)
    counts: dict[str, int] = {}
    for key, count in raw.items():
        normalized_key = _string(key, f"{path} key")
        normalized_count = _integer(count, f"{path}.{normalized_key}")
        if normalized_count == 0:
            raise ValueError(f"{path}.{normalized_key} must be positive")
        counts[normalized_key] = normalized_count
    return dict(sorted(counts.items()))


def _enum_counter(
    value: Any,
    path: str,
    allowed: Iterable[str],
) -> dict[str, int]:
    counts = _counter(value, path)
    unknown = sorted(set(counts) - set(allowed))
    if unknown:
        raise ValueError(
            f"{path} contains unknown values: {', '.join(unknown)}"
        )
    return counts


def _outcome_counter(value: Any, path: str) -> dict[str, int]:
    raw = _object(value, path)
    if set(raw) != set(TASK_OUTCOMES):
        raise ValueError(f"{path} must contain every canonical task outcome")
    return {
        outcome: _integer(raw[outcome], f"{path}.{outcome}")
        for outcome in TASK_OUTCOMES
    }
