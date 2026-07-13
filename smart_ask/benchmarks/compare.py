"""Derived benchmark summaries over the canonical call ledger."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Mapping, Sequence

from ..metrics import PriceCatalog, aggregate_resources


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    price_catalog: PriceCatalog,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["strategy_id"]), []).append(record)
    summaries = {}
    for strategy_id, values in grouped.items():
        evaluations = [value.get("evaluation") for value in values]
        outcomes = Counter()
        for value, evaluation in zip(values, evaluations, strict=True):
            error = value.get("error")
            stage = error.get("stage") if isinstance(error, Mapping) else None
            if stage == "execution":
                outcomes["execution_error"] += 1
            elif stage == "evaluation":
                outcomes["evaluation_error"] += 1
            elif not isinstance(evaluation, Mapping):
                outcomes["unrated"] += 1
            elif evaluation.get("passed") is True:
                outcomes["passed"] += 1
            else:
                outcomes["incorrect"] += 1
        passed = outcomes["passed"]
        scores = [
            float(evaluation["score"])
            for evaluation in evaluations
            if isinstance(evaluation, Mapping)
            and isinstance(evaluation.get("score"), (int, float))
        ]
        model_calls = [
            call for value in values for call in value.get("model_calls", ())
        ]
        requests = [
            request
            for value in values
            for request in value.get("provider_requests", ())
        ]
        resources = aggregate_resources(
            requests,
            model_calls,
            price_catalog=price_catalog,
        )
        paths = Counter(
            " → ".join(
                str(decision.get("outcome"))
                for decision in value.get("decisions", ())
            ) or "none"
            for value in values
        )
        summaries[strategy_id] = {
            "tasks": len(values),
            "outcomes": {
                name: outcomes[name]
                for name in (
                    "passed",
                    "incorrect",
                    "execution_error",
                    "evaluation_error",
                    "unrated",
                )
            },
            "pass_rate": passed / len(values) if values else 0.0,
            "mean_score": mean(scores) if scores else None,
            "model_calls": len(model_calls),
            "resources": resources,
            "route_paths": dict(sorted(paths.items())),
        }
    return summaries


def compare(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    strategy_order: Sequence[str],
) -> dict[str, Any]:
    if not strategy_order:
        return {}
    baseline_id = strategy_order[0]
    baseline = summaries.get(baseline_id)
    if baseline is None:
        return {}
    values = {}
    for strategy_id in strategy_order[1:]:
        current = summaries.get(strategy_id)
        if current is None:
            continue
        values[strategy_id] = {
            "pass_rate_delta": current["pass_rate"] - baseline["pass_rate"],
            "known_cost_usd_delta": (
                current["resources"]["overall"]["known_cost_usd"]
                - baseline["resources"]["overall"]["known_cost_usd"]
            ),
            "known_total_tokens_delta": (
                current["resources"]["overall"]["known_total_tokens"]
                - baseline["resources"]["overall"]["known_total_tokens"]
            ),
        }
    return {"baseline": baseline_id, "strategies": values}


def format_report(
    summaries: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
) -> str:
    if not summaries:
        return "No completed benchmark records."
    lines = [
        "strategy | passed/tasks | pass rate | tokens | cost | calls/requests",
        "--- | --- | --- | --- | --- | ---",
    ]
    for strategy, value in summaries.items():
        resources = value["resources"]["overall"]
        lines.append(
            f"{strategy} | {value['outcomes']['passed']}/{value['tasks']} | "
            f"{value['pass_rate']:.1%} | {resources['known_total_tokens']} | "
            f"${resources['known_cost_usd']:.6f} | "
            f"{value['model_calls']}/{resources['requests']}"
        )
    baseline = comparison.get("baseline")
    if baseline:
        lines.append(f"\nBaseline: {baseline}")
    return "\n".join(lines)
