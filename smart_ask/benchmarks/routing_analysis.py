"""Pure routing-flow derivation from canonical benchmark records.

Generation usage is attributed to the transition that launched the call and
exactly once to its complete path.  A terminal transition (``accept`` or
``error``) therefore never owns usage.  The classifier remains ordinary call
evidence and is deliberately not attributed to the generation routing flow.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .._numeric import checked_difference, checked_fsum
from .artifact_schema import validate_manifest, validate_records


ROUTING_FLOW_SCHEMA = "smart-ask.routing-flow/v1"

_TRANSITION_ORDER = (
    ("start", "cheap"),
    ("start", "expensive"),
    ("start", "fixed"),
    ("start", "error"),
    ("cheap", "accept"),
    ("cheap", "expensive"),
    ("cheap", "error"),
    ("expensive", "accept"),
    ("expensive", "error"),
    ("fixed", "accept"),
    ("fixed", "error"),
)
_TRANSITION_INDEX = {
    transition: index for index, transition in enumerate(_TRANSITION_ORDER)
}

_PATH_ORDER = (
    ("start", "cheap", "accept"),
    ("start", "cheap", "error"),
    ("start", "cheap", "expensive", "accept"),
    ("start", "cheap", "expensive", "error"),
    ("start", "expensive", "accept"),
    ("start", "expensive", "error"),
    ("start", "fixed", "accept"),
    ("start", "fixed", "error"),
    ("start", "error"),
)
_PATH_INDEX = {path: index for index, path in enumerate(_PATH_ORDER)}
_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "visible_output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
)
_DETAIL_TOKEN_FIELDS = _TOKEN_FIELDS[3:]


def derive_routing_flow(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive deterministic per-strategy routing transitions and paths.

    Inputs must be canonical records for the current artifact schema.  Passing
    the manifest enables the artifact layer's additional strategy/config
    consistency checks.
    Every generation call referenced by an attempt is attributed exactly once;
    classifier calls are outside the generation routing funnel.
    """

    if manifest is not None:
        validate_manifest(manifest)
    items = validate_records(records, manifest)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in items:
        grouped[str(record["strategy_id"])].append(record)

    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_id in sorted(grouped):
        strategy_records = sorted(
            grouped[strategy_id], key=lambda record: str(record["task_id"])
        )
        transition_counts: Counter[tuple[str, str]] = Counter()
        transition_calls: dict[
            tuple[str, str], list[Mapping[str, Any]]
        ] = defaultdict(list)
        path_counts: Counter[tuple[str, ...]] = Counter()
        path_calls: dict[
            tuple[str, ...], list[Mapping[str, Any]]
        ] = defaultdict(list)
        task_transition_ledger: list[dict[str, Any]] = []
        attempted_calls = 0

        for record in strategy_records:
            path, attributions, ledger = _analyze_record(record)
            attempted_calls += len(record["attempts"])
            path_counts[path] += 1
            task_transition_ledger.extend(ledger)
            for source, target in zip(path, path[1:]):
                transition_counts[(source, target)] += 1
            for transition, call in attributions:
                transition_calls[transition].append(call)
                path_calls[path].append(call)

        attributed_calls = sum(len(calls) for calls in transition_calls.values())
        if attributed_calls != attempted_calls:
            raise AssertionError("generation call attribution is not one-to-one")
        path_attributed_calls = sum(len(calls) for calls in path_calls.values())
        if path_attributed_calls != attempted_calls:
            raise AssertionError("complete paths did not account for every route call")

        transitions = []
        for source, target in _TRANSITION_ORDER:
            calls = transition_calls[(source, target)]
            transitions.append({
                "from_route": source,
                "to_route": target,
                "task_count": transition_counts[(source, target)],
                "launched_calls": len(calls),
                "failed_launched_calls": sum(
                    call["status"] == "error" for call in calls
                ),
                "usage": _usage_summary(calls),
                "cost": _cost_summary(calls),
            })

        paths = [
            {
                "states": list(path),
                "task_count": path_counts[path],
                "attempted_calls": len(path_calls[path]),
                "failed_attempted_calls": sum(
                    call["status"] == "error" for call in path_calls[path]
                ),
                "usage": _usage_summary(path_calls[path]),
                "cost": _cost_summary(path_calls[path]),
            }
            for path in sorted(path_counts, key=_path_sort_key)
        ]
        cheap_routes = transition_counts[("start", "cheap")]
        expensive_routes = transition_counts[("start", "expensive")]
        escalations = transition_counts[("cheap", "expensive")]
        rates = {
            "cheap_route_rate": _rate(
                cheap_routes,
                cheap_routes + expensive_routes,
                denominator_scope="completed_initial_route_decisions",
            ),
            "first_pass_acceptance_rate": _rate(
                transition_counts[("cheap", "accept")],
                cheap_routes,
                denominator_scope="cheap_route_tasks",
            ),
            "escalation_rate": _rate(
                escalations,
                cheap_routes,
                denominator_scope="cheap_route_tasks",
            ),
            "escalation_recovery_rate": _rate(
                path_counts[("start", "cheap", "expensive", "accept")],
                escalations,
                denominator_scope="escalated_tasks",
            ),
        }
        by_strategy[strategy_id] = {
            "tasks": len(strategy_records),
            "attempted_calls": attempted_calls,
            "attributed_calls": attributed_calls,
            "transitions": transitions,
            "paths": paths,
            "rates": rates,
            "task_transition_ledger": task_transition_ledger,
        }

    return {
        "schema": ROUTING_FLOW_SCHEMA,
        "by_strategy": by_strategy,
    }


def _analyze_record(
    record: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    list[tuple[tuple[str, str], Mapping[str, Any]]],
    list[dict[str, Any]],
]:
    attempts = list(record["attempts"])
    events = [
        (str(event["source"]), str(event["outcome"]))
        for event in record["routing_events"]
    ]
    error = record["error"]
    error_stage = None if error is None else str(error["stage"])

    if not attempts:
        if (
            error_stage != "routing"
            or events
            or record["classifier_decision"] is not None
        ):
            raise ValueError(
                "a record without route attempts must be a pre-decision "
                "routing error"
            )
        path = ("start", "error")
        return path, [], [_ledger_entry(
            task_id=str(record["task_id"]),
            sequence=1,
            gate_id="routing",
            decision="error",
            transition=("start", "error"),
            from_model=None,
            to_model=None,
            resulting_call_id=None,
            path=path,
        )]

    phases = tuple(str(attempt["route"]["phase"]) for attempt in attempts)
    valid_sequences = {
        ("fixed",),
        ("initial-easy",),
        ("initial-easy", "escalation"),
        ("initial-hard",),
    }
    if phases not in valid_sequences:
        raise ValueError(f"unsupported routing attempt sequence: {phases!r}")

    calls_by_id = {str(call["call_id"]): call for call in record["calls"]}
    attempt_calls = [calls_by_id[str(attempt["call_id"])] for attempt in attempts]
    statuses = tuple(str(call["status"]) for call in attempt_calls)
    _validate_terminal_state(phases, statuses, error_stage)
    _validate_route_events(record, phases, events, error_stage)

    terminal = "error" if error_stage in {"routing", "execution"} else "accept"
    initial_state = {
        "fixed": "fixed",
        "initial-easy": "cheap",
        "initial-hard": "expensive",
    }[phases[0]]
    if len(phases) == 2:
        path = ("start", initial_state, "expensive", terminal)
    else:
        path = ("start", initial_state, terminal)
    if path not in _PATH_INDEX:
        raise ValueError(f"unsupported routing path: {path!r}")

    attributions = [(("start", initial_state), attempt_calls[0])]
    if len(attempt_calls) == 2:
        attributions.append((("cheap", "expensive"), attempt_calls[1]))
    if any(transition not in _TRANSITION_INDEX for transition, _ in attributions):
        raise AssertionError("attempt was assigned to an unknown transition")
    ledger = _task_transition_ledger(
        record,
        path,
        attempts,
        attempt_calls,
        events,
        error_stage,
    )
    return path, attributions, ledger


def _task_transition_ledger(
    record: Mapping[str, Any],
    path: tuple[str, ...],
    attempts: Sequence[Mapping[str, Any]],
    attempt_calls: Sequence[Mapping[str, Any]],
    events: Sequence[tuple[str, str]],
    error_stage: str | None,
) -> list[dict[str, Any]]:
    task_id = str(record["task_id"])
    state_models = {
        {
            "fixed": "fixed",
            "initial-easy": "cheap",
            "initial-hard": "expensive",
            "escalation": "expensive",
        }[str(attempt["route"]["phase"])]: str(attempt["route"]["model"])
        for attempt in attempts
    }
    launched = {
        ("start", path[1]): attempt_calls[0],
    }
    if len(attempt_calls) == 2:
        launched[("cheap", "expensive")] = attempt_calls[1]

    ledger = []
    for sequence, transition in enumerate(zip(path, path[1:]), start=1):
        source, target = transition
        gate_id, decision = _gate_and_decision(
            transition,
            events,
            error_stage,
        )
        call = launched.get(transition)
        ledger.append(_ledger_entry(
            task_id=task_id,
            sequence=sequence,
            gate_id=gate_id,
            decision=decision,
            transition=transition,
            from_model=state_models.get(source),
            to_model=state_models.get(target),
            resulting_call_id=(
                None if call is None else str(call["call_id"])
            ),
            path=path,
        ))
    attributed_ids = [
        row["resulting_call_id"]
        for row in ledger
        if row["resulting_call_id"] is not None
    ]
    expected_ids = [str(call["call_id"]) for call in attempt_calls]
    if attributed_ids != expected_ids:
        raise AssertionError("task transition ledger lost or repeated a route call")
    return ledger


def _gate_and_decision(
    transition: tuple[str, str],
    events: Sequence[tuple[str, str]],
    error_stage: str | None,
) -> tuple[str, str]:
    source, target = transition
    if source == "start" and target in {"cheap", "expensive"}:
        return "difficulty-classifier", target
    if transition == ("start", "fixed"):
        return "fixed-method", "fixed"
    if transition == ("cheap", "expensive"):
        return "response-escalation", "escalate"
    if transition == ("cheap", "accept") and (
        "response-escalation", "accept"
    ) in events:
        return "response-escalation", "accept"
    if target == "error":
        return error_stage or "execution", "error"
    return "completion", "accept"


def _ledger_entry(
    *,
    task_id: str,
    sequence: int,
    gate_id: str,
    decision: str,
    transition: tuple[str, str],
    from_model: str | None,
    to_model: str | None,
    resulting_call_id: str | None,
    path: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "sequence": sequence,
        "gate_id": gate_id,
        "decision": decision,
        "from_route": transition[0],
        "to_route": transition[1],
        "from_model": from_model,
        "to_model": to_model,
        "task_count": 1,
        "resulting_call_id": resulting_call_id,
        "path": list(path),
    }


def _validate_terminal_state(
    phases: tuple[str, ...],
    statuses: tuple[str, ...],
    error_stage: str | None,
) -> None:
    if error_stage == "execution":
        if statuses[-1] != "error" or any(
            status != "ok" for status in statuses[:-1]
        ):
            raise ValueError(
                "execution errors require exactly the final attempted call to fail"
            )
        return
    if any(status != "ok" for status in statuses):
        raise ValueError("non-execution-error paths require successful route calls")
    if error_stage == "routing" and phases != ("initial-easy",):
        raise ValueError(
            "post-attempt routing errors are only valid after the easy route"
        )


def _validate_route_events(
    record: Mapping[str, Any],
    phases: tuple[str, ...],
    events: Sequence[tuple[str, str]],
    error_stage: str | None,
) -> None:
    decision = record["classifier_decision"]
    if phases == ("fixed",):
        if decision is not None or list(events) != [("fixed-method", "fixed")]:
            raise ValueError("fixed route evidence is contradictory")
        return

    expected_decision = "hard" if phases == ("initial-hard",) else "easy"
    if decision != expected_decision:
        raise ValueError("classifier decision contradicts the attempted route")
    expected_prefix = [("difficulty-classifier", expected_decision)]
    if phases == ("initial-easy", "escalation"):
        expected = expected_prefix + [("response-escalation", "escalate")]
        if list(events) != expected:
            raise ValueError("escalation route evidence is contradictory")
        return
    if phases == ("initial-hard",):
        if list(events) != expected_prefix:
            raise ValueError("hard route evidence is contradictory")
        return

    # A one-shot difficulty strategy has no response gate.  A completed
    # cascade adds an explicit accept event.  Failed easy execution/routing
    # cannot have reached that gate.
    allowed = [expected_prefix]
    if error_stage not in {"routing", "execution"}:
        allowed.append(expected_prefix + [("response-escalation", "accept")])
    if list(events) not in allowed:
        raise ValueError("easy route evidence is contradictory")


def _usage_summary(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_total = sum(call["usage"]["total_tokens"] is None for call in calls)
    missing_breakdown = sum(
        call["usage"]["prompt_tokens"] is None
        or call["usage"]["completion_tokens"] is None
        for call in calls
    )
    known = {
        field: sum(int(call["usage"][field] or 0) for call in calls)
        for field in _TOKEN_FIELDS
    }
    return {
        "known": known,
        "total_tokens": known["total_tokens"] if missing_total == 0 else None,
        "completeness": {
            "total": missing_total == 0,
            "breakdown": missing_breakdown == 0,
            "missing_total_calls": missing_total,
            "missing_breakdown_calls": missing_breakdown,
            "error_calls": sum(
                call["usage"]["status"] == "error" for call in calls
            ),
            "details": {
                field: {
                    "complete": all(
                        call["usage"][field] is not None for call in calls
                    ),
                    "missing_calls": sum(
                        call["usage"][field] is None for call in calls
                    ),
                }
                for field in _DETAIL_TOKEN_FIELDS
            },
        },
    }


def _cost_summary(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known_costs = [
        float(call["cost"]["usd"])
        for call in calls
        if call["cost"]["usd"] is not None
    ]
    missing = len(calls) - len(known_costs)
    known = checked_fsum(known_costs, name="routing aggregate catalog cost")
    provider_costs = [
        float(call["cost"]["provider_reported_usd"])
        for call in calls
        if call["cost"]["provider_reported_usd"] is not None
    ]
    provider_missing = len(calls) - len(provider_costs)
    provider_known = checked_fsum(
        provider_costs,
        name="routing aggregate provider-reported cost",
    )
    comparable = [
        call
        for call in calls
        if call["cost"]["usd"] is not None
        and call["cost"]["provider_reported_usd"] is not None
    ]
    known_difference = checked_fsum(
        (
            checked_difference(
                float(call["cost"]["usd"]),
                float(call["cost"]["provider_reported_usd"]),
                name="routing catalog/provider cost difference",
            )
            for call in comparable
        ),
        name="routing aggregate catalog/provider cost difference",
    )
    return {
        "known_usd": known,
        "total_usd": known if missing == 0 else None,
        "completeness": {
            "complete": missing == 0,
            "missing_calls": missing,
            "error_calls": sum(
                call["cost"]["status"] == "error" for call in calls
            ),
        },
        "provider_reported": {
            "known_usd": provider_known,
            "total_usd": provider_known if provider_missing == 0 else None,
            "complete": provider_missing == 0,
            "missing_calls": provider_missing,
        },
        "catalog_estimate_minus_provider": {
            "known_usd": known_difference,
            "total_usd": (
                known_difference if len(comparable) == len(calls) else None
            ),
            "comparable_calls": len(comparable),
            "missing_calls": len(calls) - len(comparable),
        },
    }


def _path_sort_key(path: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return (_PATH_INDEX.get(path, len(_PATH_INDEX)), path)


def _rate(
    numerator: int,
    denominator: int,
    *,
    denominator_scope: str,
) -> dict[str, int | float | str | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "denominator_scope": denominator_scope,
        "value": numerator / denominator if denominator else None,
    }


__all__ = ["ROUTING_FLOW_SCHEMA", "derive_routing_flow"]
