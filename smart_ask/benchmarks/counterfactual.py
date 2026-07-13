"""Counterfactual routing quality from routed, cheap, and hard baselines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _cost(record: Mapping[str, Any]) -> float | None:
    requests = record.get("provider_requests", ())
    if not requests:
        return None
    values = [
        request.get("provider_cost_usd")
        for request in requests
    ]
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def _score(record: Mapping[str, Any]) -> float | None:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def _passed(record: Mapping[str, Any]) -> bool | None:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("passed")
    return value if isinstance(value, bool) else None


def evaluate_counterfactual_routing(
    records: Sequence[Mapping[str, Any]],
    *,
    routed_strategy: str,
    cheap_strategy: str,
    hard_strategy: str,
) -> dict[str, Any]:
    by_key = {}
    for record in records:
        key = (str(record["strategy_id"]), str(record["task_id"]))
        if key in by_key:
            raise ValueError(f"duplicate counterfactual record: {key}")
        by_key[key] = record
    task_ids = sorted({
        task for strategy, task in by_key
        if strategy == routed_strategy
    })
    unnecessary_expensive = 0
    unsafe_cheap = 0
    opportunities = 0
    captured = 0
    known_cost_regret = 0.0
    known_quality_regret = 0.0
    known_cost_regret_tasks = 0
    known_quality_regret_tasks = 0
    hard_routes = 0
    justified_hard_routes = 0
    route_identity_unknown = 0
    matched = 0
    for task_id in task_ids:
        routed = by_key.get((routed_strategy, task_id))
        cheap = by_key.get((cheap_strategy, task_id))
        hard = by_key.get((hard_strategy, task_id))
        if routed is None or cheap is None or hard is None:
            continue
        matched += 1
        cheap_pass = _passed(cheap)
        hard_pass = _passed(hard)
        routed_pass = _passed(routed)
        routed_call = _final_call(routed)
        cheap_call = _final_call(cheap)
        hard_call = _final_call(hard)
        used_hard = _same_route(routed_call, hard_call)
        used_cheap = _same_route(routed_call, cheap_call)
        if not used_hard and not used_cheap:
            route_identity_unknown += 1
        if used_hard:
            hard_routes += 1
            justified_hard_routes += int(cheap_pass is False and hard_pass is True)
        if cheap_pass is True:
            opportunities += 1
            captured += int(not used_hard)
            unnecessary_expensive += int(used_hard)
        if cheap_pass is False and hard_pass is True and routed_pass is False:
            unsafe_cheap += int(used_cheap)
        cheap_cost, hard_cost, routed_cost = map(_cost, (cheap, hard, routed))
        if None not in (cheap_cost, hard_cost, routed_cost):
            oracle_cost = (
                cheap_cost if cheap_pass is True
                else hard_cost if hard_pass is True
                else None
            )
            if oracle_cost is not None:
                known_cost_regret += max(0.0, routed_cost - oracle_cost)
                known_cost_regret_tasks += 1
        cheap_score, hard_score, routed_score = map(_score, (cheap, hard, routed))
        if None not in (cheap_score, hard_score, routed_score):
            known_quality_regret += max(
                0.0,
                max(cheap_score, hard_score) - routed_score,
            )
            known_quality_regret_tasks += 1
    return {
        "matched_tasks": matched,
        "unnecessary_expensive": unnecessary_expensive,
        "unsafe_cheap": unsafe_cheap,
        "cheap_opportunities": opportunities,
        "cheap_opportunity_capture": (
            captured / opportunities if opportunities else None
        ),
        "hard_routes": hard_routes,
        "escalation_precision": (
            justified_hard_routes / hard_routes if hard_routes else None
        ),
        "route_identity_unknown": route_identity_unknown,
        "cost_regret_usd": (
            known_cost_regret if known_cost_regret_tasks else None
        ),
        "cost_regret_known_tasks": known_cost_regret_tasks,
        "quality_regret": (
            known_quality_regret if known_quality_regret_tasks else None
        ),
        "quality_regret_known_tasks": known_quality_regret_tasks,
    }


def _final_call(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    final = record.get("final_call")
    matches = [
        call
        for call in record.get("model_calls", ())
        if call.get("call_id") == final
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _same_route(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    if left is None or right is None:
        return False
    left_target = left.get("target_id")
    right_target = right.get("target_id")
    left_model = left.get("selected_model")
    right_model = right.get("selected_model")
    return (
        isinstance(left_target, str)
        and left_target
        and left_target == right_target
        and (
            left_model == right_model
            or left_model is None
            or right_model is None
        )
    )
