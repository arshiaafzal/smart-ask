"""Canonical resource aggregation over provider-request evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .cost import PriceCatalog, price_usage
from .models import TokenUsage


_TOKEN_FIELDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "visible_output": "visible_output_tokens",
    "reasoning": "reasoning_tokens",
    "cache_read": "cache_read_tokens",
    "cache_write": "cache_write_tokens",
}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _empty_bucket() -> dict[str, Any]:
    value: dict[str, Any] = {
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "cancelled_requests": 0,
        "tool_calls": 0,
        "known_total_tokens": 0,
        "missing_total_token_requests": 0,
        "known_cost_usd": 0.0,
        "provider_reported_cost_usd": 0.0,
        "catalog_estimated_cost_usd": 0.0,
        "missing_cost_requests": 0,
        "output_statuses": {
            "usable": 0,
            "empty": 0,
            "truncated": 0,
            "refused": 0,
        },
        "duration_ms_sum": 0.0,
        "duration_ms_p50": None,
        "duration_ms_p95": None,
        "time_to_first_output_ms_p50": None,
        "time_to_first_output_ms_p95": None,
        "output_tokens_per_second": None,
        "cost_sources": {},
    }
    for label in _TOKEN_FIELDS:
        value[f"known_{label}_tokens"] = 0
        value[f"missing_{label}_token_requests"] = 0
    return value


def _model_identity(request: Mapping[str, Any]) -> tuple[str, str]:
    actual = request.get("actual_model")
    if isinstance(actual, str) and actual:
        return actual, actual
    selected = request.get("selected_model")
    if isinstance(selected, str) and selected:
        return f"requested:{selected}", selected
    return "unknown", "unknown"


def _add(
    bucket: dict[str, Any],
    request: Mapping[str, Any],
    *,
    model: str,
    price_catalog: PriceCatalog,
    durations: list[float],
    first_outputs: list[float],
    output_rates: list[float],
) -> None:
    bucket["requests"] += 1
    status = request.get("status")
    bucket["successful_requests"] += int(status == "completed")
    bucket["failed_requests"] += int(status == "error")
    bucket["cancelled_requests"] += int(status == "cancelled")
    tool_calls = request.get("tool_call_count")
    if isinstance(tool_calls, int) and not isinstance(tool_calls, bool):
        bucket["tool_calls"] += tool_calls

    token_values: dict[str, int | None] = {}
    for label, field in _TOKEN_FIELDS.items():
        value = request.get(field)
        normalized = (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )
        token_values[label] = normalized
        if normalized is None:
            bucket[f"missing_{label}_token_requests"] += 1
        else:
            bucket[f"known_{label}_tokens"] += normalized
    total = (
        token_values["input"] + token_values["output"]
        if token_values["input"] is not None
        and token_values["output"] is not None
        else None
    )
    if total is None:
        bucket["missing_total_token_requests"] += 1
    else:
        bucket["known_total_tokens"] += total

    provider_cost = request.get("provider_cost_usd")
    if isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool):
        cost = float(provider_cost)
        source = "provider"
        bucket["provider_reported_cost_usd"] += cost
    else:
        quote = price_usage(model, TokenUsage(
            prompt_tokens=token_values["input"],
            completion_tokens=token_values["output"],
            total_tokens=total,
            reasoning_tokens=token_values["reasoning"],
            cached_input_tokens=token_values["cache_read"],
            cache_write_input_tokens=token_values["cache_write"],
        ), price_catalog)
        cost = quote.cost_usd
        source = (
            f"catalog:{price_catalog.catalog_id}"
            if cost is not None
            else "missing"
        )
        if cost is not None:
            bucket["catalog_estimated_cost_usd"] += cost
    sources = bucket["cost_sources"]
    sources[source] = sources.get(source, 0) + 1
    if cost is None:
        bucket["missing_cost_requests"] += 1
    else:
        bucket["known_cost_usd"] += cost

    output_status = request.get("output_status")
    if output_status in bucket["output_statuses"]:
        bucket["output_statuses"][output_status] += 1
    duration = request.get("duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        duration = float(duration)
        durations.append(duration)
        bucket["duration_ms_sum"] += duration
        first = request.get("time_to_first_output_ms")
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            first = float(first)
            first_outputs.append(first)
            if token_values["output"] is not None and duration > first:
                output_rates.append(
                    token_values["output"] * 1000 / (duration - first)
                )


def _finish(
    bucket: dict[str, Any],
    durations: Sequence[float],
    first_outputs: Sequence[float],
    output_rates: Sequence[float],
) -> None:
    bucket["duration_ms_p50"] = _percentile(durations, 0.50)
    bucket["duration_ms_p95"] = _percentile(durations, 0.95)
    bucket["time_to_first_output_ms_p50"] = _percentile(first_outputs, 0.50)
    bucket["time_to_first_output_ms_p95"] = _percentile(first_outputs, 0.95)
    if output_rates:
        bucket["output_tokens_per_second"] = sum(output_rates) / len(output_rates)
    bucket["cost_sources"] = dict(sorted(bucket["cost_sources"].items()))


def aggregate_resources(
    requests: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    *,
    price_catalog: PriceCatalog,
) -> dict[str, Any]:
    """Aggregate resource evidence overall and by useful attribution axes."""

    if not isinstance(price_catalog, PriceCatalog):
        raise TypeError("price_catalog must be a PriceCatalog")
    call_by_id = {call.get("call_id"): call for call in calls}
    dimensions: dict[str, dict[str, dict[str, Any]]] = {
        "by_model": {},
        "by_target": {},
        "by_profile": {},
        "by_role": {},
    }
    samples: dict[int, tuple[list[float], list[float], list[float]]] = {}

    def bucket(container: dict[str, dict[str, Any]], key: str):
        value = container.setdefault(key, _empty_bucket())
        return value, samples.setdefault(id(value), ([], [], []))

    overall = _empty_bucket()
    samples[id(overall)] = ([], [], [])
    for request in requests:
        if not isinstance(request, Mapping):
            raise TypeError("provider requests must be mappings")
        model, priced_model = _model_identity(request)
        call = call_by_id.get(request.get("call_id"), {})
        keys = {
            "by_model": model,
            "by_target": str(request.get("target_id") or "unknown"),
            "by_profile": str(call.get("profile_id") or "unknown"),
            "by_role": str(call.get("role") or "unknown"),
        }
        targets = [(overall, samples[id(overall)])]
        targets.extend(bucket(dimensions[name], key) for name, key in keys.items())
        for value, (durations, first_outputs, output_rates) in targets:
            _add(
                value,
                request,
                model=priced_model,
                price_catalog=price_catalog,
                durations=durations,
                first_outputs=first_outputs,
                output_rates=output_rates,
            )
    _finish(overall, *samples[id(overall)])
    for values in dimensions.values():
        for value in values.values():
            _finish(value, *samples[id(value)])
    return {
        "pricing": {
            "catalog_id": price_catalog.catalog_id,
            "effective_date": price_catalog.effective_date,
        },
        "overall": overall,
        **{
            name: dict(sorted(values.items()))
            for name, values in dimensions.items()
        },
    }
