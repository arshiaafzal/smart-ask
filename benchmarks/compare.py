"""Aggregate and paired comparison logic for benchmark-owned records."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize accuracy, routing, cost, usage, attempts, and latency."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["strategy_id"])].append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for strategy_id, items in sorted(grouped.items()):
        passed = sum(_passed(item) for item in items)
        scores = [_score(item) for item in items]
        known_costs = [
            float(item["cost_usd"])
            for item in items
            if item.get("cost_usd") is not None
        ]
        latencies = [
            float(item["total_latency_ms"])
            for item in items
            if item.get("total_latency_ms") is not None
        ]
        routes = Counter(str(item.get("route") or "unknown") for item in items)
        summaries[strategy_id] = {
            "tasks": len(items),
            "passed": passed,
            "pass_rate": passed / len(items) if items else 0.0,
            "mean_score": mean(scores) if scores else 0.0,
            "errors": sum(item.get("error") is not None for item in items),
            "routes": dict(sorted(routes.items())),
            "calls": sum(len(item.get("calls", [])) for item in items),
            "attempts": sum(len(item.get("attempts", [])) for item in items),
            "prompt_tokens": sum(
                int(item.get("usage", {}).get("prompt_tokens", 0)) for item in items
            ),
            "completion_tokens": sum(
                int(item.get("usage", {}).get("completion_tokens", 0)) for item in items
            ),
            "total_cost_usd": sum(known_costs),
            "missing_cost_tasks": len(items) - len(known_costs),
            "mean_latency_ms": mean(latencies) if latencies else None,
            "p50_latency_ms": median(latencies) if latencies else None,
            "p95_latency_ms": _percentile(latencies, 0.95) if latencies else None,
            "missing_latency_tasks": len(items) - len(latencies),
        }
    return summaries


def compare(
    records: Iterable[Mapping[str, Any]],
    *,
    strategy_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return all paired comparisons without dropping missing or failed tasks."""

    items = list(records)
    by_strategy: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in items:
        by_strategy[str(record["strategy_id"])][str(record["task_id"])] = record

    strategies = list(strategy_order or sorted(by_strategy))
    pairs = []
    for reference, candidate in combinations(strategies, 2):
        ref_records = by_strategy.get(reference, {})
        cand_records = by_strategy.get(candidate, {})
        task_ids = sorted(set(ref_records) | set(cand_records))
        both_pass = only_reference = only_candidate = neither = missing = 0
        paired_rows = []
        cost_deltas = []
        latency_deltas = []
        score_deltas = []
        missing_cost_pairs = 0
        missing_latency_pairs = 0

        for task_id in task_ids:
            ref = ref_records.get(task_id)
            cand = cand_records.get(task_id)
            if ref is None or cand is None:
                missing += 1
                paired_rows.append({
                    "task_id": task_id,
                    "reference_missing": ref is None,
                    "candidate_missing": cand is None,
                })
                continue

            ref_passed = _passed(ref)
            cand_passed = _passed(cand)
            if ref_passed and cand_passed:
                both_pass += 1
            elif ref_passed:
                only_reference += 1
            elif cand_passed:
                only_candidate += 1
            else:
                neither += 1

            score_delta = _score(cand) - _score(ref)
            score_deltas.append(score_delta)
            cost_delta = _optional_delta(cand.get("cost_usd"), ref.get("cost_usd"))
            latency_delta = _optional_delta(
                cand.get("total_latency_ms"), ref.get("total_latency_ms")
            )
            if cost_delta is not None:
                cost_deltas.append(cost_delta)
            else:
                missing_cost_pairs += 1
            if latency_delta is not None:
                latency_deltas.append(latency_delta)
            else:
                missing_latency_pairs += 1
            paired_rows.append({
                "task_id": task_id,
                "reference_passed": ref_passed,
                "candidate_passed": cand_passed,
                "reference_route": ref.get("route"),
                "candidate_route": cand.get("route"),
                "score_delta": score_delta,
                "cost_delta_usd": cost_delta,
                "latency_delta_ms": latency_delta,
            })

        pairs.append({
            "reference": reference,
            "candidate": candidate,
            "tasks": len(task_ids),
            "paired_tasks": len(task_ids) - missing,
            "missing_tasks": missing,
            "both_pass": both_pass,
            "only_reference_passes": only_reference,
            "only_candidate_passes": only_candidate,
            "neither_passes": neither,
            "mean_score_delta": mean(score_deltas) if score_deltas else None,
            "total_cost_delta_usd": (
                sum(cost_deltas) if not missing_cost_pairs else None
            ),
            "missing_cost_pairs": missing_cost_pairs,
            "mean_latency_delta_ms": mean(latency_deltas) if latency_deltas else None,
            "missing_latency_pairs": missing_latency_pairs,
            "per_task": paired_rows,
        })

    return {"strategy_order": strategies, "pairs": pairs}


def format_report(
    summaries: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
) -> str:
    """Render a compact terminal report for one or several strategies."""

    lines = [
        "strategy                  pass@1        cost       p50 ms   attempts",
        "─" * 72,
    ]
    for strategy_id, summary in summaries.items():
        passed = f"{summary['passed']}/{summary['tasks']} ({summary['pass_rate'] * 100:.1f}%)"
        cost = (
            f"${summary['total_cost_usd']:.6f}"
            if not summary.get("missing_cost_tasks")
            else f"${summary['total_cost_usd']:.6f}+?"
        )
        p50 = summary.get("p50_latency_ms")
        latency = f"{p50:.1f}" if p50 is not None else "?"
        lines.append(
            f"{strategy_id:<25} {passed:<14} {cost:>11} {latency:>10} "
            f"{summary['attempts']:>10}"
        )

    for pair in comparison.get("pairs", []):
        lines.extend([
            "",
            f"{pair['reference']} vs {pair['candidate']}: "
            f"only reference {pair['only_reference_passes']}, "
            f"only candidate {pair['only_candidate_passes']}, "
            f"both {pair['both_pass']}, neither {pair['neither_passes']}, "
            f"missing {pair['missing_tasks']}",
        ])
    return "\n".join(lines)


def _passed(record: Mapping[str, Any]) -> bool:
    return bool(record.get("evaluation", {}).get("passed", False))


def _score(record: Mapping[str, Any]) -> float:
    return float(record.get("evaluation", {}).get("score", 1.0 if _passed(record) else 0.0))


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]
