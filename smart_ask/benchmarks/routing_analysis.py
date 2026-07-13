"""Routing-funnel analysis derived from canonical decisions and model calls."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def analyze_routing(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = Counter()
    gates: dict[str, Counter] = {}
    transitions = Counter()
    calls_by_profile = Counter()
    tokens_by_profile = Counter()
    tokens_by_transition = Counter()
    for record in records:
        decisions = record.get("decisions", ())
        path = []
        previous = "start"
        transition_by_decision = {}
        for decision in decisions:
            gate = str(decision.get("gate") or "unknown")
            outcome = str(decision.get("outcome") or "unknown")
            gates.setdefault(gate, Counter())[outcome] += 1
            transition = f"{previous} → {outcome}"
            transitions[transition] += 1
            transition_by_decision[decision.get("decision_id")] = transition
            previous = outcome
            path.append(outcome)
        if path and previous != "accept":
            transitions[f"{previous} → response"] += 1
        paths[" → ".join(path) or "none"] += 1
        calls = {call["call_id"]: call for call in record.get("model_calls", ())}
        for request in record.get("provider_requests", ()):
            call = calls.get(request.get("call_id"), {})
            profile = str(call.get("profile_id") or "unknown")
            calls_by_profile[profile] += 1
            prompt = request.get("input_tokens")
            completion = request.get("output_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                tokens_by_profile[profile] += prompt + completion
                caused_by = call.get("caused_by_decision_id")
                transition = transition_by_decision.get(caused_by)
                if transition is None:
                    transition = (
                        "classification"
                        if call.get("role") == "classifier"
                        else "unattributed"
                    )
                tokens_by_transition[transition] += prompt + completion
    return {
        "tasks": len(records),
        "paths": dict(sorted(paths.items())),
        "gates": {
            gate: dict(sorted(values.items()))
            for gate, values in sorted(gates.items())
        },
        "transitions": dict(sorted(transitions.items())),
        "tokens_by_transition": dict(sorted(tokens_by_transition.items())),
        "model_calls_by_profile": dict(sorted(calls_by_profile.items())),
        "tokens_by_profile": dict(sorted(tokens_by_profile.items())),
    }
