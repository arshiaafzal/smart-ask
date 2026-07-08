"""Aggregate and compare strict benchmark records for the current schema."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import math
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from ..metrics import aggregate_metric_payloads
from .._numeric import checked_fsum
from ..metrics.rollups import aggregate_record_resources

from .artifact_schema import validate_records
from .counterfactual import evaluate_counterfactual_routing
from .routing_analysis import derive_routing_flow


def summarize(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize evaluation results and canonical per-task metrics."""

    items = validate_records(records, manifest)
    routing_flow = derive_routing_flow(items, manifest=manifest)["by_strategy"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in items:
        grouped[record["strategy_id"]].append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for strategy_id, strategy_records in sorted(grouped.items()):
        strategy_records.sort(key=lambda item: str(item["task_id"]))
        rated_records = [
            item for item in strategy_records if _is_rated(item)
        ]
        passed = sum(_outcome(item) == "passed" for item in rated_records)
        scores = [_score(item) for item in rated_records]
        durations = [
            float(item["metrics"]["timing"]["run_duration_ms"])
            for item in strategy_records
        ]
        routes = Counter(
            str(item.get("route") or "unknown") for item in strategy_records
        )
        metric_summary = aggregate_metric_payloads(
            item["metrics"] for item in strategy_records
        ).to_dict()
        summaries[strategy_id] = {
            "tasks": len(strategy_records),
            "evaluation": {
                "rated_tasks": len(rated_records),
                "excluded_tasks": len(strategy_records) - len(rated_records),
                "all_task_success_rate": passed / len(strategy_records),
                "pass_rate": (
                    passed / len(rated_records) if rated_records else None
                ),
                "mean_score": _finite_mean(scores),
            },
            "routes": dict(sorted(routes.items())),
            "metrics": metric_summary,
            "resources": aggregate_record_resources(
                strategy_records
            ).to_dict(),
            "routing_flow": routing_flow[strategy_id],
            "timing": {
                "wall_clock_record_span_ms": _wall_clock_record_span_ms(
                    strategy_records
                ),
                "cumulative_run_duration_ms": metric_summary["timing"][
                    "cumulative_run_duration_ms"
                ],
                "run_duration_distribution_ms": {
                    "mean": mean(durations),
                    "p50": median(durations),
                    "p95": _percentile(durations, 0.95),
                },
            },
        }
    return summaries


def compare(
    records: Iterable[Mapping[str, Any]],
    *,
    strategy_order: Sequence[str] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return strict paired comparisons without dropping missing tasks."""

    items = validate_records(records, manifest)
    by_strategy: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in items:
        by_strategy[record["strategy_id"]][record["task_id"]] = record

    observed = set(by_strategy)
    if strategy_order is None:
        strategies = sorted(observed)
    else:
        strategies = [str(strategy) for strategy in strategy_order]
        if len(set(strategies)) != len(strategies):
            raise ValueError("strategy_order must not contain duplicates")
        omitted = sorted(observed - set(strategies))
        if omitted:
            raise ValueError(
                "strategy_order omits observed strategy/strategies: "
                + ", ".join(omitted)
            )

    pairs = []
    for reference, candidate in combinations(strategies, 2):
        ref_records = by_strategy.get(reference, {})
        cand_records = by_strategy.get(candidate, {})
        task_ids = sorted(set(ref_records) | set(cand_records))
        paired_records = [
            (ref_records[task_id], cand_records[task_id])
            for task_id in task_ids
            if task_id in ref_records and task_id in cand_records
        ]
        cost_source = (
            "provider_reported"
            if paired_records and all(
                _cost_value(record, "provider_reported") is not None
                for pair in paired_records
                for record in pair
            )
            else "catalog_estimate"
        )
        both_pass = only_reference = only_candidate = neither = missing = 0
        rated_pairs = excluded_pairs = 0
        paired_rows = []
        cost_deltas: list[float] = []
        duration_deltas: list[float] = []
        score_deltas: list[float] = []
        missing_cost_pairs = 0

        for task_id in task_ids:
            ref = ref_records.get(task_id)
            cand = cand_records.get(task_id)
            if ref is None or cand is None:
                missing += 1
                ref_outcome = None if ref is None else _outcome(ref)
                cand_outcome = None if cand is None else _outcome(cand)
                paired_rows.append({
                    "task_id": task_id,
                    "reference_missing": ref is None,
                    "candidate_missing": cand is None,
                    "reference_outcome": ref_outcome,
                    "candidate_outcome": cand_outcome,
                    "reference_passed": _passed_value(ref_outcome),
                    "candidate_passed": _passed_value(cand_outcome),
                    "quality_rated": False,
                })
                continue

            ref_outcome = _outcome(ref)
            cand_outcome = _outcome(cand)
            quality_rated = (
                ref_outcome in _RATED_OUTCOMES
                and cand_outcome in _RATED_OUTCOMES
            )
            ref_passed = _passed_value(ref_outcome)
            cand_passed = _passed_value(cand_outcome)
            if quality_rated:
                rated_pairs += 1
                if ref_passed and cand_passed:
                    both_pass += 1
                elif ref_passed:
                    only_reference += 1
                elif cand_passed:
                    only_candidate += 1
                else:
                    neither += 1
                score_delta = _optional_delta(_score(cand), _score(ref))
                if score_delta is not None:
                    score_deltas.append(score_delta)
            else:
                excluded_pairs += 1
                score_delta = None
            cost_delta = _optional_delta(
                _cost_value(cand, cost_source),
                _cost_value(ref, cost_source),
            )
            duration_delta = (
                float(cand["metrics"]["timing"]["run_duration_ms"])
                - float(ref["metrics"]["timing"]["run_duration_ms"])
            )
            if cost_delta is None:
                missing_cost_pairs += 1
            else:
                cost_deltas.append(cost_delta)
            duration_deltas.append(duration_delta)
            paired_rows.append({
                "task_id": task_id,
                "reference_outcome": ref_outcome,
                "candidate_outcome": cand_outcome,
                "reference_passed": ref_passed,
                "candidate_passed": cand_passed,
                "quality_rated": quality_rated,
                "reference_route": ref["route"],
                "candidate_route": cand["route"],
                "score_delta": score_delta,
                "cost_delta_usd": cost_delta,
                "duration_delta_ms": duration_delta,
            })

        pairs.append({
            "reference": reference,
            "candidate": candidate,
            "tasks": len(task_ids),
            "paired_tasks": len(task_ids) - missing,
            "missing_tasks": missing,
            "rated_pairs": rated_pairs,
            "excluded_pairs": excluded_pairs,
            "both_pass": both_pass,
            "only_reference_passes": only_reference,
            "only_candidate_passes": only_candidate,
            "neither_passes": neither,
            "mean_score_delta": _finite_mean(score_deltas),
            "total_cost_delta_usd": (
                _finite_sum(cost_deltas)
                if task_ids and not missing and not missing_cost_pairs
                else None
            ),
            "missing_cost_pairs": missing_cost_pairs,
            "cost_source": cost_source,
            "mean_duration_delta_ms": (
                _finite_mean(duration_deltas)
            ),
            "per_task": paired_rows,
        })

    report = {
        "strategy_order": strategies,
        "pairs": pairs,
        "resources": aggregate_record_resources(items).to_dict(),
        "timing": {
            "wall_clock_record_span_ms": _wall_clock_record_span_ms(items),
            "cumulative_run_duration_ms": checked_fsum(
                (
                    float(item["metrics"]["timing"]["run_duration_ms"])
                    for item in items
                ),
                name="comparison cumulative run duration",
            ),
        },
    }
    if manifest is not None:
        report["counterfactual_routing"] = evaluate_counterfactual_routing(
            manifest,
            items,
        )
    return report


def format_report(
    summaries: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
) -> str:
    """Render a compact terminal report for one or several strategies."""

    lines = [
        "strategy                  success/all   rated pass   excl       cost       p50 ms   attempts",
        "─" * 100,
    ]
    for strategy_id, summary in summaries.items():
        metrics = summary["metrics"]
        cost_metrics = metrics["cost"]
        evaluation = summary["evaluation"]
        pass_rate = evaluation["pass_rate"]
        rated_pass = (
            f"{metrics['outcomes']['passed']}/{evaluation['rated_tasks']} "
            + (
                f"({pass_rate * 100:.1f}%)"
                if pass_rate is not None
                else "(n/a)"
            )
        )
        provider_cost = cost_metrics["provider_reported"]
        if provider_cost["complete"]:
            cost = f"${provider_cost['known_usd']:.6f} billed"
        else:
            suffix = "" if cost_metrics["completeness"]["complete"] else "+?"
            cost = f"${cost_metrics['known_usd']:.6f}{suffix} est."
        all_task_success = (
            f"{metrics['outcomes']['passed']}/{summary['tasks']} "
            f"({evaluation['all_task_success_rate'] * 100:.1f}%)"
        )
        lines.append(
            f"{strategy_id:<25} {all_task_success:<13} {rated_pass:<12} "
            f"{evaluation['excluded_tasks']:>4} {cost:>11} "
            f"{summary['timing']['run_duration_distribution_ms']['p50']:>10.1f} "
            f"{metrics['routing']['generation_attempts']:>10}"
        )

    for pair in comparison.get("pairs", []):
        lines.extend([
            "",
            f"{pair['reference']} vs {pair['candidate']}: "
            f"only reference {pair['only_reference_passes']}, "
            f"only candidate {pair['only_candidate_passes']}, "
            f"both {pair['both_pass']}, neither {pair['neither_passes']}, "
            f"excluded {pair['excluded_pairs']}, missing {pair['missing_tasks']}",
        ])
    return "\n".join(lines)


_RATED_OUTCOMES = frozenset({"passed", "incorrect"})


def _outcome(record: Mapping[str, Any]) -> str:
    outcomes = record["metrics"]["outcomes"]
    observed = [name for name, count in outcomes.items() if count == 1]
    if len(observed) != 1:
        raise ValueError("record must contain exactly one task outcome")
    return observed[0]


def _is_rated(record: Mapping[str, Any]) -> bool:
    return _outcome(record) in _RATED_OUTCOMES


def _passed_value(outcome: str | None) -> bool | None:
    if outcome not in _RATED_OUTCOMES:
        return None
    return outcome == "passed"


def _score(record: Mapping[str, Any]) -> float:
    return float(record["evaluation"]["score"])


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    value = float(left) - float(right)
    return value if math.isfinite(value) else None


def _finite_sum(values: Sequence[float]) -> float | None:
    if not values:
        return None
    try:
        value = checked_fsum(values, name="comparison aggregate")
    except ValueError:
        return None
    return value


def _finite_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    try:
        value = mean(values)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _cost_value(record: Mapping[str, Any], source: str) -> Any:
    cost = record["metrics"]["cost"]
    if source == "provider_reported":
        return cost["provider_reported"]["total_usd"]
    return cost["total_usd"]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _wall_clock_record_span_ms(
    records: Sequence[Mapping[str, Any]],
) -> float | None:
    if not records:
        return None
    starts = [_datetime(item["started_at"]) for item in records]
    finishes = [_datetime(item["finished_at"]) for item in records]
    return (max(finishes) - min(starts)).total_seconds() * 1_000


def _datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
