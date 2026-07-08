"""Paired counterfactual diagnostics for benchmark routing decisions.

The diagnostics in this module compare observed outputs from a routed strategy
with observed outputs from fixed strategies over the same benchmark cases.  A
paired fixed run is evidence about that run, not proof of what would have
happened inside the routed run; the report deliberately preserves that caveat.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable, Mapping

from .._numeric import checked_fsum, is_finite_real
from ..strategy.schema import (
    CascadeMethodConfig,
    DifficultyMethodConfig,
    FilePromptConfig,
    FixedMethodConfig,
    InlinePromptConfig,
    ModelProfileConfig,
    StrategyConfig,
)
from .artifact_schema import (
    SCHEMA_VERSION,
    validate_manifest,
    validate_records,
)


COUNTERFACTUAL_REPORT_SCHEMA_VERSION = 1

_EVIDENCE_CAVEAT = (
    "Diagnostics are associations across paired observed benchmark outputs. "
    "They do not establish that a different route would have caused the same "
    "output in the routed execution."
)
_BASELINE_SEMANTICS = (
    "Cheap and expensive mean the routed strategy's configured easy and hard "
    "profiles. The labels are not inferred from observed per-task cost."
)

_METRIC_DEFINITIONS = {
    "cheap_opportunity_capture": (
        "Among tasks whose fixed cheap-profile output passed, the share whose "
        "routed output also passed using only the cheap profile."
    ),
    "unnecessary_expensive_rate": (
        "Among routed tasks that used the expensive profile, the share whose "
        "paired fixed cheap-profile output passed; this is a paired-run proxy, "
        "not a causal necessity claim."
    ),
    "unsafe_cheap_rate": (
        "Among routed tasks that used only the cheap profile, the share where "
        "the routed output failed while the paired fixed expensive-profile "
        "output passed."
    ),
    "escalation_precision": (
        "Among routed tasks that escalated, the share where the paired fixed "
        "cheap-profile output failed and the fixed expensive-profile output "
        "passed."
    ),
    "cost_regret_usd": (
        "Routed total cost minus the paired oracle-baseline cost. The oracle is "
        "the least-cost passing baseline. Provider-reported cost is used when "
        "the routed run and all passing baselines expose it; otherwise complete "
        "catalog estimates are preferred."
    ),
    "quality_regret": (
        "Paired oracle-baseline score minus routed score, using the same "
        "passing-baseline oracle as cost_regret_usd."
    ),
}


def evaluate_counterfactual_routing(
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive deterministic paired routing diagnostics from current artifacts.

    Difficulty and cascade strategies are matched to fixed baselines by the
    complete generation profile: model, resolved system prompt, tuning
    parameters, and generation executor configuration.  A report is still
    emitted when matching or per-task evidence is unavailable, with explicit
    reasons and null values.
    """

    validate_manifest(manifest)
    items = validate_records(records, manifest)
    strategy_snapshots = {
        str(strategy["name"]): strategy
        for strategy in manifest["strategies"]
    }
    configs = {
        strategy_id: StrategyConfig.model_validate(snapshot["config"])
        for strategy_id, snapshot in strategy_snapshots.items()
    }
    by_strategy: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in items:
        by_strategy[str(record["strategy_id"])][str(record["task_id"])] = record

    fixed_by_profile: dict[str, list[str]] = defaultdict(list)
    for strategy_id, config in configs.items():
        if not isinstance(config.method, FixedMethodConfig):
            continue
        fixed_by_profile[_profile_key(
            config.method.model,
            config,
            strategy_snapshots[strategy_id],
            role=config.method.role,
            user_prompt_transform=_fixed_prompt_transform(
                config.method,
                strategy_snapshots[strategy_id],
            ),
        )].append(strategy_id)
    for candidates in fixed_by_profile.values():
        candidates.sort()

    reports = []
    for strategy_id in sorted(configs):
        config = configs[strategy_id]
        if not isinstance(
            config.method,
            (DifficultyMethodConfig, CascadeMethodConfig),
        ):
            continue
        reports.append(_strategy_report(
            strategy_id=strategy_id,
            config=config,
            snapshot=strategy_snapshots[strategy_id],
            fixed_by_profile=fixed_by_profile,
            records_by_strategy=by_strategy,
            task_ids=[str(task_id) for task_id in manifest["case_ids"]],
        ))

    return {
        "schema_version": COUNTERFACTUAL_REPORT_SCHEMA_VERSION,
        "benchmark_schema_version": SCHEMA_VERSION,
        "evidence_caveat": _EVIDENCE_CAVEAT,
        "baseline_semantics": _BASELINE_SEMANTICS,
        "metric_definitions": dict(_METRIC_DEFINITIONS),
        "strategies": reports,
    }


def _strategy_report(
    *,
    strategy_id: str,
    config: StrategyConfig,
    snapshot: Mapping[str, Any],
    fixed_by_profile: Mapping[str, list[str]],
    records_by_strategy: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_ids: list[str],
) -> dict[str, Any]:
    method = config.method
    if not isinstance(method, (DifficultyMethodConfig, CascadeMethodConfig)):
        raise TypeError("counterfactual reports require a routed strategy")

    cheap_transform = (
        {
            "suffix": _resolved_prompt(
                method.escalation.self_check_suffix,
                snapshot,
            ),
        }
        if isinstance(method, CascadeMethodConfig)
        else None
    )
    cheap_key = _profile_key(
        method.easy,
        config,
        snapshot,
        role="generator",
        user_prompt_transform=cheap_transform,
    )
    expensive_key = _profile_key(
        method.hard,
        config,
        snapshot,
        role="writer",
        user_prompt_transform=None,
    )
    cheap_match = _baseline_match(
        model=method.easy.model,
        candidates=fixed_by_profile.get(cheap_key, []),
        label="cheap",
        user_prompt_transform=cheap_transform,
    )
    expensive_match = _baseline_match(
        model=method.hard.model,
        candidates=fixed_by_profile.get(expensive_key, []),
        label="expensive",
        user_prompt_transform=None,
    )
    common_mapping_reasons = (
        ["profiles_not_distinct"] if cheap_key == expensive_key else []
    )
    cheap_mapping_reasons = list(common_mapping_reasons)
    expensive_mapping_reasons = list(common_mapping_reasons)
    if cheap_match["status"] != "matched":
        cheap_mapping_reasons.append(
            f"{cheap_match['status']}_cheap_baseline"
        )
    if expensive_match["status"] != "matched":
        expensive_mapping_reasons.append(
            f"{expensive_match['status']}_expensive_baseline"
        )
    mapping_reasons = cheap_mapping_reasons + expensive_mapping_reasons
    mapping_reasons = sorted(set(mapping_reasons))
    baselines_available = not mapping_reasons

    cheap_id = (
        cheap_match["strategy_id"]
        if not cheap_mapping_reasons
        else None
    )
    expensive_id = (
        expensive_match["strategy_id"]
        if not expensive_mapping_reasons
        else None
    )
    per_task = [
        _task_report(
            task_id=task_id,
            routed_record=records_by_strategy.get(strategy_id, {}).get(task_id),
            cheap_record=(
                records_by_strategy.get(str(cheap_id), {}).get(task_id)
                if cheap_id is not None else None
            ),
            expensive_record=(
                records_by_strategy.get(str(expensive_id), {}).get(task_id)
                if expensive_id is not None else None
            ),
            cheap_mapping_reasons=cheap_mapping_reasons,
            expensive_mapping_reasons=expensive_mapping_reasons,
        )
        for task_id in task_ids
    ]

    evidence_reasons = Counter(
        reason
        for task in per_task
        for reason in task["full_pair_evidence"]["reasons"]
    )
    evidence_tasks = sum(
        task["full_pair_evidence"]["status"] == "available"
        for task in per_task
    )
    return {
        "strategy_id": strategy_id,
        "method": method.type,
        "baselines": {
            "status": "available" if baselines_available else "unavailable",
            "reasons": mapping_reasons,
            "cheap": cheap_match,
            "expensive": expensive_match,
        },
        "tasks": len(per_task),
        "fully_paired_tasks": evidence_tasks,
        "incomplete_pair_tasks": len(per_task) - evidence_tasks,
        "incomplete_pair_reasons": dict(sorted(evidence_reasons.items())),
        "metrics": {
            name: _aggregate_ratio(per_task, name)
            for name in (
                "cheap_opportunity_capture",
                "unnecessary_expensive_rate",
                "unsafe_cheap_rate",
                "escalation_precision",
            )
        } | {
            "cost_regret_usd": _aggregate_delta(
                per_task,
                "cost_regret_usd",
            ),
            "quality_regret": _aggregate_delta(per_task, "quality_regret"),
        },
        "per_task": per_task,
    }


def _baseline_match(
    *,
    model: str,
    candidates: Iterable[str],
    label: str,
    user_prompt_transform: Mapping[str, str] | None,
) -> dict[str, Any]:
    candidate_ids = sorted(str(candidate) for candidate in candidates)
    status = (
        "missing" if not candidate_ids
        else "matched" if len(candidate_ids) == 1
        else "ambiguous"
    )
    return {
        "profile": label,
        "model": model,
        "required_user_prompt_transform": user_prompt_transform,
        "status": status,
        "strategy_id": candidate_ids[0] if status == "matched" else None,
        "candidate_strategy_ids": candidate_ids,
    }


def _profile_key(
    profile: ModelProfileConfig,
    config: StrategyConfig,
    snapshot: Mapping[str, Any],
    *,
    role: str,
    user_prompt_transform: Mapping[str, str] | None,
) -> str:
    prompt_texts = {
        str(prompt["declared_path"]): str(prompt["text"])
        for prompt in snapshot["prompts"]
    }
    prompt = profile.system_prompt
    if prompt is None:
        resolved_prompt = None
    elif isinstance(prompt, InlinePromptConfig):
        resolved_prompt = prompt.text
    elif isinstance(prompt, FilePromptConfig):
        resolved_prompt = prompt_texts[prompt.path]
    else:  # pragma: no cover - the closed pydantic union prevents this
        raise TypeError("unknown model-profile prompt type")
    identity = {
        "model": profile.model,
        "role": role,
        "system_prompt": resolved_prompt,
        "user_prompt_transform": user_prompt_transform,
        "parameters": profile.parameters.model_dump(mode="json"),
        "generation": config.generation.model_dump(mode="json"),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _resolved_prompt(
    prompt: InlinePromptConfig | FilePromptConfig,
    snapshot: Mapping[str, Any],
) -> str:
    if isinstance(prompt, InlinePromptConfig):
        return prompt.text
    prompt_texts = {
        str(item["declared_path"]): str(item["text"])
        for item in snapshot["prompts"]
    }
    return prompt_texts[prompt.path]


def _fixed_prompt_transform(
    method: FixedMethodConfig,
    snapshot: Mapping[str, Any],
) -> dict[str, str] | None:
    transform = {}
    if method.prompt_prefix is not None:
        transform["prefix"] = _resolved_prompt(method.prompt_prefix, snapshot)
    if method.prompt_suffix is not None:
        transform["suffix"] = _resolved_prompt(method.prompt_suffix, snapshot)
    return transform or None


def _task_report(
    *,
    task_id: str,
    routed_record: Mapping[str, Any] | None,
    cheap_record: Mapping[str, Any] | None,
    expensive_record: Mapping[str, Any] | None,
    cheap_mapping_reasons: list[str],
    expensive_mapping_reasons: list[str],
) -> dict[str, Any]:
    selection = _route_selection(routed_record)
    routed_evaluation = _evaluation_snapshot(routed_record)
    cheap_evaluation = _evaluation_snapshot(cheap_record)
    expensive_evaluation = _evaluation_snapshot(expensive_record)
    routed_reasons = _evaluation_reasons("routed", routed_record)
    cheap_reasons = _evaluation_reasons(
        "cheap",
        cheap_record,
        mapping_reasons=cheap_mapping_reasons,
    )
    expensive_reasons = _evaluation_reasons(
        "expensive",
        expensive_record,
        mapping_reasons=expensive_mapping_reasons,
    )
    reasons = sorted(set(
        routed_reasons + cheap_reasons + expensive_reasons
    ))
    evidence_available = not reasons

    metrics = {
        "cheap_opportunity_capture": _indicator(None, None, cheap_reasons),
        "unnecessary_expensive_rate": _indicator(None, None, routed_reasons),
        "unsafe_cheap_rate": _indicator(None, None, routed_reasons),
        "escalation_precision": _indicator(None, None, routed_reasons),
        "cost_regret_usd": _unavailable_delta(reasons),
        "quality_regret": _unavailable_delta(reasons),
    }
    oracle = None
    oracle_cost_source = None

    routed_passed = routed_evaluation["passed"] is True
    cheap_passed = cheap_evaluation["passed"] is True
    expensive_passed = expensive_evaluation["passed"] is True
    cheap_only = selection == "cheap_only"
    used_expensive = selection in ("expensive_direct", "escalated")
    escalated = selection == "escalated"

    if not cheap_reasons:
        if not cheap_passed:
            metrics["cheap_opportunity_capture"] = _indicator(False, None)
        elif routed_reasons:
            metrics["cheap_opportunity_capture"] = _indicator(
                None, None, routed_reasons
            )
        else:
            metrics["cheap_opportunity_capture"] = _indicator(
                True,
                routed_passed and cheap_only,
            )
    if not routed_reasons:
        if not used_expensive:
            metrics["unnecessary_expensive_rate"] = _indicator(False, None)
        elif cheap_reasons:
            metrics["unnecessary_expensive_rate"] = _indicator(
                None, None, cheap_reasons
            )
        else:
            metrics["unnecessary_expensive_rate"] = _indicator(
                True,
                cheap_passed,
            )

        if not cheap_only:
            metrics["unsafe_cheap_rate"] = _indicator(False, None)
        elif routed_passed:
            metrics["unsafe_cheap_rate"] = _indicator(True, False)
        elif expensive_reasons:
            metrics["unsafe_cheap_rate"] = _indicator(
                None, None, expensive_reasons
            )
        else:
            metrics["unsafe_cheap_rate"] = _indicator(
                True,
                not routed_passed and expensive_passed,
            )

        if not escalated:
            metrics["escalation_precision"] = _indicator(False, None)
        elif not cheap_reasons and cheap_passed:
            metrics["escalation_precision"] = _indicator(True, False)
        elif not expensive_reasons and not expensive_passed:
            metrics["escalation_precision"] = _indicator(True, False)
        elif cheap_reasons or expensive_reasons:
            metrics["escalation_precision"] = _indicator(
                None,
                None,
                cheap_reasons + expensive_reasons,
            )
        else:
            metrics["escalation_precision"] = _indicator(
                True,
                not cheap_passed and expensive_passed,
            )

    if evidence_available:
        (
            oracle,
            oracle_record,
            oracle_evaluation,
            oracle_cost_source,
            oracle_reasons,
        ) = _select_oracle(
            routed_record=routed_record,
            cheap_record=cheap_record,
            cheap_evaluation=cheap_evaluation,
            expensive_record=expensive_record,
            expensive_evaluation=expensive_evaluation,
        )

        if oracle_record is None or oracle_evaluation is None:
            metrics["cost_regret_usd"] = _unavailable_delta(oracle_reasons)
            metrics["quality_regret"] = _unavailable_delta(oracle_reasons)
        else:
            metrics["quality_regret"] = _available_delta(
                float(oracle_evaluation["score"])
                - float(routed_evaluation["score"])
            )
            cost_reasons = []
            if oracle_cost_source is None:
                oracle_cost_source = _paired_cost_source(
                    routed_record,
                    oracle_record,
                )
            routed_cost = (
                _total_cost(routed_record, oracle_cost_source)
                if oracle_cost_source is not None
                else None
            )
            oracle_cost = (
                _total_cost(oracle_record, oracle_cost_source)
                if oracle_cost_source is not None
                else None
            )
            if routed_cost is not None and oracle_cost is not None:
                metrics["cost_regret_usd"] = _available_delta(
                    routed_cost - oracle_cost,
                    source=oracle_cost_source,
                )
            else:
                if routed_cost is None:
                    cost_reasons.append("routed_cost_unavailable")
                if oracle_cost is None:
                    cost_reasons.append(f"{oracle}_baseline_cost_unavailable")
                metrics["cost_regret_usd"] = _unavailable_delta(cost_reasons)

    return {
        "task_id": task_id,
        "full_pair_evidence": {
            "status": "available" if evidence_available else "unavailable",
            "reasons": reasons,
        },
        "route_path": _route_path(routed_record),
        "selection": selection,
        "routed": routed_evaluation,
        "cheap_baseline": cheap_evaluation,
        "expensive_baseline": expensive_evaluation,
        "oracle_baseline": oracle,
        "oracle_cost_source": oracle_cost_source,
        "metrics": metrics,
    }


def _evaluation_reasons(
    label: str,
    record: Mapping[str, Any] | None,
    *,
    mapping_reasons: Iterable[str] = (),
) -> list[str]:
    reasons = sorted(set(mapping_reasons))
    if reasons:
        return reasons
    if record is None:
        return [f"missing_{label}_record"]
    if record["error"] is not None:
        return [f"{label}_{record['error']['stage']}_error"]
    return []


def _route_selection(record: Mapping[str, Any] | None) -> str | None:
    if record is None:
        return None
    phases = [attempt["route"]["phase"] for attempt in record["attempts"]]
    if phases == ["initial-easy"]:
        return "cheap_only"
    if phases == ["initial-hard"]:
        return "expensive_direct"
    if phases == ["initial-easy", "escalation"]:
        return "escalated"
    return None


def _route_path(record: Mapping[str, Any] | None) -> str | None:
    if record is None:
        return None
    selection = _route_selection(record)
    if selection == "cheap_only":
        middle = ["cheap"]
    elif selection == "expensive_direct":
        middle = ["expensive"]
    elif selection == "escalated":
        middle = ["cheap", "expensive"]
    else:
        middle = []
    error = record["error"]
    terminal = (
        "error"
        if error is not None and error["stage"] in {"routing", "execution"}
        else "accept"
    )
    return " -> ".join(["start", *middle, terminal])


def _evaluation_snapshot(
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if record is None or record["error"] is not None:
        return {"passed": None, "score": None}
    return {
        "passed": bool(record["evaluation"]["passed"]),
        "score": float(record["evaluation"]["score"]),
    }


def _total_cost(
    record: Mapping[str, Any] | None,
    source: str,
) -> float | None:
    if record is None:
        return None
    cost = record["metrics"]["cost"]
    value = (
        cost.get("provider_reported", {}).get("total_usd")
        if source == "provider_reported"
        else cost["total_usd"]
    )
    return None if value is None else float(value)


def _select_oracle(
    *,
    routed_record: Mapping[str, Any],
    cheap_record: Mapping[str, Any] | None,
    cheap_evaluation: Mapping[str, Any],
    expensive_record: Mapping[str, Any] | None,
    expensive_evaluation: Mapping[str, Any],
) -> tuple[
    str | None,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
    str | None,
    list[str],
]:
    candidates = [
        ("cheap", cheap_record, cheap_evaluation),
        ("expensive", expensive_record, expensive_evaluation),
    ]
    passing = [
        candidate
        for candidate in candidates
        if candidate[1] is not None and candidate[2]["passed"] is True
    ]
    if not passing:
        return None, None, None, None, ["no_passing_baseline"]
    if len(passing) == 1:
        label, record, evaluation = passing[0]
        return (
            label,
            record,
            evaluation,
            _paired_cost_source(routed_record, record),
            [],
        )

    # Rank the oracle with one source that can also price the routed execution.
    # Fall back from provider charges to catalog estimates as a whole; never
    # rank with one source and subtract with another.
    for source in ("provider_reported", "catalog_estimate"):
        priced = [
            (label, record, evaluation, _total_cost(record, source))
            for label, record, evaluation in passing
        ]
        if (
            _total_cost(routed_record, source) is not None
            and all(item[3] is not None for item in priced)
        ):
            label, record, evaluation, _cost = min(
                priced,
                key=lambda item: (
                    float(item[3]),
                    0 if item[0] == "cheap" else 1,
                ),
            )
            return label, record, evaluation, source, []
    # Quality regret can still use a cost-minimal oracle even when routed cost
    # is unavailable, but cost regret will remain explicitly unavailable.
    for source in ("provider_reported", "catalog_estimate"):
        priced = [
            (label, record, evaluation, _total_cost(record, source))
            for label, record, evaluation in passing
        ]
        if all(item[3] is not None for item in priced):
            label, record, evaluation, _cost = min(
                priced,
                key=lambda item: (
                    float(item[3]),
                    0 if item[0] == "cheap" else 1,
                ),
            )
            return label, record, evaluation, source, []
    return (
        None,
        None,
        None,
        None,
        ["passing_baseline_costs_incomparable"],
    )


def _paired_cost_source(
    routed_record: Mapping[str, Any],
    oracle_record: Mapping[str, Any],
) -> str | None:
    for source in ("provider_reported", "catalog_estimate"):
        if (
            _total_cost(routed_record, source) is not None
            and _total_cost(oracle_record, source) is not None
        ):
            return source
    # There is no comparable pair. Keep the strongest source observed on
    # either side so the diagnostic identifies only the side actually missing.
    for source in ("provider_reported", "catalog_estimate"):
        if (
            _total_cost(routed_record, source) is not None
            or _total_cost(oracle_record, source) is not None
        ):
            return source
    return None


def _indicator(
    eligible: bool | None,
    value: bool | None,
    reasons: Iterable[str] = (),
) -> dict[str, Any]:
    unavailable_reasons = sorted(set(reasons))
    if unavailable_reasons:
        return {
            "status": "unavailable",
            "eligible": None,
            "value": None,
            "reasons": unavailable_reasons,
        }
    if eligible is not True:
        value = None
    return {
        "status": "available",
        "eligible": eligible,
        "value": value,
        "reasons": [],
    }


def _available_delta(
    value: float,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    if not is_finite_real(value):
        return _unavailable_delta(["non_finite_delta"])
    return {
        "status": "available",
        "value": value,
        "source": source,
        "reasons": [],
    }


def _unavailable_delta(reasons: Iterable[str]) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value": None,
        "source": None,
        "reasons": sorted(set(reasons)),
    }


def _aggregate_ratio(
    tasks: Iterable[Mapping[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    snapshots = list(tasks)
    indicators = [task["metrics"][metric_name] for task in snapshots]
    evidence_tasks = sum(
        indicator["status"] == "available" for indicator in indicators
    )
    denominator = sum(indicator["eligible"] is True for indicator in indicators)
    numerator = sum(
        indicator["eligible"] is True and indicator["value"] is True
        for indicator in indicators
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "evidence_tasks": evidence_tasks,
        "unavailable_tasks": len(snapshots) - evidence_tasks,
        "unavailable_reasons": dict(sorted(Counter(
            reason
            for indicator in indicators
            if indicator["status"] == "unavailable"
            for reason in indicator["reasons"]
        ).items())),
    }


def _aggregate_delta(
    tasks: Iterable[Mapping[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    snapshots = [task["metrics"][metric_name] for task in tasks]
    values = [
        float(snapshot["value"])
        for snapshot in snapshots
        if snapshot["status"] == "available"
    ]
    unavailable_reasons = Counter(
        reason
        for snapshot in snapshots
        if snapshot["status"] == "unavailable"
        for reason in snapshot["reasons"]
    )
    sources = Counter(
        snapshot["source"]
        for snapshot in snapshots
        if snapshot["status"] == "available"
        and snapshot["source"] is not None
    )
    by_source = {}
    for source in sorted(sources):
        source_values = [
            float(snapshot["value"])
            for snapshot in snapshots
            if snapshot["status"] == "available"
            and snapshot["source"] == source
        ]
        by_source[source] = _delta_group(source_values)
    mixed_sources = len(sources) > 1
    total = _finite_sum(values) if values and not mixed_sources else None
    aggregate_reasons = (
        ["mixed_cost_sources"]
        if mixed_sources
        else ["non_finite_aggregate"]
        if values and total is None
        else ["no_available_values"]
        if not values
        else []
    )
    return {
        "available_tasks": len(values),
        "unavailable_tasks": len(snapshots) - len(values),
        "total": total,
        "mean": total / len(values) if total is not None else None,
        "aggregate_status": "available" if total is not None else "unavailable",
        "aggregate_reasons": aggregate_reasons,
        "mixed_sources": mixed_sources,
        "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
        "sources": dict(sorted(sources.items())),
        "by_source": by_source,
    }


def _finite_sum(values: Iterable[float]) -> float | None:
    try:
        total = checked_fsum(values, name="counterfactual aggregate")
    except ValueError:
        return None
    return total


def _delta_group(values: list[float]) -> dict[str, Any]:
    total = _finite_sum(values)
    return {
        "available_tasks": len(values),
        "total": total,
        "mean": total / len(values) if total is not None else None,
        "status": "available" if total is not None else "unavailable",
    }
