from copy import deepcopy
import unittest

from smart_ask.benchmarks.artifact_schema import SCHEMA_VERSION
from smart_ask.benchmarks.routing_analysis import (
    ROUTING_FLOW_SCHEMA,
    derive_routing_flow,
)
from smart_ask.metrics import (
    CallStats,
    PriceCatalog,
    PriceQuote,
    RunStats,
    TokenUsage,
)


_CATALOG = PriceCatalog(
    catalog_id="routing-test",
    effective_date="2026-07-01",
    source="unit-test",
    prices={"model": {"input": 0.1, "output": 0.2}},
)


def _call(
    run_id,
    ordinal,
    *,
    channel,
    role,
    tokens=(2, 1),
    cost=0.4,
    failed=False,
):
    call_id = f"call-{ordinal}"
    if failed:
        usage = TokenUsage()
        usage_status = "unavailable"
        usage_diagnostic = "call failed"
        quote = PriceQuote(None, "unavailable", diagnostic="call failed")
        actual_model = priced_model = None
        error_type = "RuntimeError"
        error_message = "provider down"
    else:
        usage = (
            TokenUsage(total_tokens=tokens)
            if isinstance(tokens, int)
            else TokenUsage(prompt_tokens=tokens[0], completion_tokens=tokens[1])
        )
        usage_status = "total_only" if isinstance(tokens, int) else "complete"
        usage_diagnostic = (
            "provider returned only total tokens"
            if isinstance(tokens, int)
            else None
        )
        quote = (
            PriceQuote(
                cost,
                "priced",
                catalog=_CATALOG,
            )
            if cost is not None
            else PriceQuote(
                None,
                "unavailable",
                catalog=_CATALOG,
                diagnostic="pricing requires input/output token counts",
            )
        )
        actual_model = priced_model = "model"
        error_type = error_message = None

    stats = CallStats(
        run_id=run_id,
        call_id=call_id,
        ordinal=ordinal,
        channel=channel,
        role=role,
        requested_model="model",
        actual_model=actual_model,
        priced_model=priced_model,
        status="error" if failed else "ok",
        latency_ms=1.0,
        started_offset_ms=float(ordinal),
        usage=usage,
        usage_status=usage_status,
        usage_diagnostic=usage_diagnostic,
        price_quote=quote,
        finish_reason="error" if failed else "stop",
        output_status=None if failed else "usable",
        output_empty=None if failed else False,
        requested_max_tokens=None,
        applied_max_tokens=None,
        max_tokens_reached=False,
        provider_cost_usd=None,
        error_category="unknown" if failed else None,
        error_type=error_type,
        error_message=error_message,
    )
    payload = stats.to_dict()
    prompt = f"prompt-{ordinal}"
    payload["request"] = {
        "model": "model",
        "role": role,
        "prompt": prompt,
        "max_tokens": None,
        "temperature": None,
    }
    payload["output"] = (
        None
        if failed
        else {"model": "model", "text": "answer", "raw_text": "answer"}
    )
    return stats, payload


def _record(
    strategy_id,
    task_id,
    phases,
    *,
    error_stage=None,
    total_only=False,
    unknown_cost=False,
    cascade_accept=False,
):
    run_id = f"run-{strategy_id}-{task_id}"
    call_stats = []
    calls = []
    events = []
    decision = None

    if phases and phases[0] != "fixed":
        classifier_stats, classifier = _call(
            run_id,
            1,
            channel="classifier",
            role="classifier",
            tokens=(1, 1),
            cost=0.3,
        )
        call_stats.append(classifier_stats)
        calls.append(classifier)
        decision = "hard" if phases[0] == "initial-hard" else "easy"
        events.append({
            "source": "difficulty-classifier",
            "outcome": decision,
            "reason": "classified",
            "model": "model",
            "call_ids": [classifier["call_id"]],
        })
    elif phases:
        events.append({
            "source": "fixed-method",
            "outcome": "fixed",
            "reason": "configured",
            "model": None,
            "call_ids": [],
        })

    attempts = []
    for attempt_index, phase in enumerate(phases, start=1):
        ordinal = len(calls) + 1
        is_last = attempt_index == len(phases)
        failed = error_stage == "execution" and is_last
        role = {
            "fixed": "generator",
            "initial-easy": "generator",
            "initial-hard": "writer",
            "escalation": "fixer",
        }[phase]
        tokens = (
            (5, 2)
            if phase == "escalation"
            else 9
            if total_only
            else (2, 1)
        )
        call_cost = None if unknown_cost else 0.9 if phase == "escalation" else 0.4
        stats, call = _call(
            run_id,
            ordinal,
            channel="generation",
            role=role,
            tokens=tokens,
            cost=call_cost,
            failed=failed,
        )
        call_stats.append(stats)
        calls.append(call)
        attempts.append({
            "index": attempt_index,
            "route": {
                "action": "execute",
                "phase": phase,
                "label": phase,
                "model": "model",
                "role": role,
                "prompt": call["request"]["prompt"],
            },
            "call_id": call["call_id"],
            "status": call["status"],
            **(
                {"reconstructed": True}
                if error_stage in {"routing", "execution"}
                else {}
            ),
        })

    if phases == ("initial-easy", "escalation"):
        events.append({
            "source": "response-escalation",
            "outcome": "escalate",
            "reason": "retry",
            "model": None,
            "call_ids": [],
        })
    elif phases == ("initial-easy",) and cascade_accept:
        events.append({
            "source": "response-escalation",
            "outcome": "accept",
            "reason": "accepted",
            "model": None,
            "call_ids": [],
        })

    stats = RunStats(
        run_id=run_id,
        task_id=task_id,
        strategy_id=strategy_id,
        duration_ms=20.0,
        calls=tuple(call_stats),
        generation_attempts=len(attempts),
        routing_events=len(events),
        outcome=(
            "routing_error"
            if error_stage == "routing"
            else "execution_error"
            if error_stage == "execution"
            else "evaluation_error"
            if error_stage == "evaluation"
            else "passed"
        ),
    )
    pre_evaluation_error = error_stage in {"routing", "execution"}
    evaluation_error = error_stage == "evaluation"
    final_output = (
        None
        if pre_evaluation_error or not attempts
        else calls[-1]["output"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "strategy_digest": "a" * 64,
        "task_id": task_id,
        "input": {"prompt": "prompt"},
        "route": phases[-1] if phases else None,
        "classifier_decision": decision,
        "routing_events": events,
        "attempts": attempts,
        "calls": calls,
        "final_output": final_output,
        "evaluation": (
            {"passed": False, "score": 0.0, "details": {}}
            if error_stage is not None
            else {"passed": True, "score": 1.0, "details": {}}
        ),
        "metrics": stats.to_dict(include_calls=False),
        "evaluation_latency_ms": (
            None if pre_evaluation_error else 0.5
        ),
        "error": (
            None
            if error_stage is None
            else {
                "stage": error_stage,
                "type": "RuntimeError",
                "message": "failed",
            }
        ),
        "started_at": "2026-07-01T00:00:00Z",
        "finished_at": "2026-07-01T00:00:01Z",
    }


def _transition(strategy, source, target):
    return next(
        transition
        for transition in strategy["transitions"]
        if transition["from_route"] == source
        and transition["to_route"] == target
    )


def _path(strategy, states):
    return next(
        path for path in strategy["paths"] if path["states"] == list(states)
    )


class RoutingAnalysisTests(unittest.TestCase):
    def test_derives_paths_and_attributes_generation_calls_once(self):
        records = [
            _record("router", "task-2", ("initial-easy", "escalation")),
            _record(
                "router",
                "task-1",
                ("initial-easy",),
                cascade_accept=True,
            ),
        ]

        result = derive_routing_flow(reversed(records))
        self.assertEqual(result["schema"], ROUTING_FLOW_SCHEMA)
        router = result["by_strategy"]["router"]
        self.assertEqual(router["tasks"], 2)
        self.assertEqual(router["attempted_calls"], 3)
        self.assertEqual(router["attributed_calls"], 3)
        self.assertEqual(
            [(path["states"], path["task_count"]) for path in router["paths"]],
            [
                (["start", "cheap", "accept"], 1),
                (["start", "cheap", "expensive", "accept"], 1),
            ],
        )

        easy_path = _path(router, ("start", "cheap", "accept"))
        self.assertEqual(easy_path["attempted_calls"], 1)
        self.assertEqual(easy_path["failed_attempted_calls"], 0)
        self.assertEqual(easy_path["usage"]["total_tokens"], 3)
        self.assertAlmostEqual(easy_path["cost"]["total_usd"], 0.4)
        escalation_path = _path(
            router, ("start", "cheap", "expensive", "accept")
        )
        self.assertEqual(escalation_path["attempted_calls"], 2)
        self.assertEqual(escalation_path["failed_attempted_calls"], 0)
        self.assertEqual(escalation_path["usage"]["total_tokens"], 10)
        self.assertAlmostEqual(escalation_path["cost"]["total_usd"], 1.3)

        initial = _transition(router, "start", "cheap")
        self.assertEqual(initial["task_count"], 2)
        self.assertEqual(initial["launched_calls"], 2)
        self.assertEqual(initial["usage"]["total_tokens"], 6)
        self.assertAlmostEqual(initial["cost"]["total_usd"], 0.8)

        escalation = _transition(router, "cheap", "expensive")
        self.assertEqual(escalation["task_count"], 1)
        self.assertEqual(escalation["launched_calls"], 1)
        self.assertEqual(escalation["usage"]["total_tokens"], 7)
        self.assertAlmostEqual(escalation["cost"]["total_usd"], 0.9)

        terminal = _transition(router, "expensive", "accept")
        self.assertEqual(terminal["task_count"], 1)
        self.assertEqual(terminal["launched_calls"], 0)
        self.assertEqual(terminal["usage"]["total_tokens"], 0)
        self.assertEqual(terminal["cost"]["total_usd"], 0.0)
        self.assertEqual(router["rates"], {
            "cheap_route_rate": {
                "numerator": 2,
                "denominator": 2,
                "denominator_scope": "completed_initial_route_decisions",
                "value": 1.0,
            },
            "first_pass_acceptance_rate": {
                "numerator": 1,
                "denominator": 2,
                "denominator_scope": "cheap_route_tasks",
                "value": 0.5,
            },
            "escalation_rate": {
                "numerator": 1,
                "denominator": 2,
                "denominator_scope": "cheap_route_tasks",
                "value": 0.5,
            },
            "escalation_recovery_rate": {
                "numerator": 1,
                "denominator": 1,
                "denominator_scope": "escalated_tasks",
                "value": 1.0,
            },
        })

        ledger = router["task_transition_ledger"]
        self.assertEqual(len(ledger), 5)
        self.assertEqual(ledger[0], {
            "task_id": "task-1",
            "sequence": 1,
            "gate_id": "difficulty-classifier",
            "decision": "cheap",
            "from_route": "start",
            "to_route": "cheap",
            "from_model": None,
            "to_model": "model",
            "task_count": 1,
            "resulting_call_id": "call-2",
            "path": ["start", "cheap", "accept"],
        })
        self.assertEqual(ledger[1], {
            "task_id": "task-1",
            "sequence": 2,
            "gate_id": "response-escalation",
            "decision": "accept",
            "from_route": "cheap",
            "to_route": "accept",
            "from_model": "model",
            "to_model": None,
            "task_count": 1,
            "resulting_call_id": None,
            "path": ["start", "cheap", "accept"],
        })
        self.assertEqual(
            {
                key: ledger[3][key]
                for key in (
                    "gate_id",
                    "decision",
                    "from_route",
                    "to_route",
                    "from_model",
                    "to_model",
                    "resulting_call_id",
                )
            },
            {
                "gate_id": "response-escalation",
                "decision": "escalate",
                "from_route": "cheap",
                "to_route": "expensive",
                "from_model": "model",
                "to_model": "model",
                "resulting_call_id": "call-3",
            },
        )

    def test_strategies_and_transition_rows_have_deterministic_order(self):
        result = derive_routing_flow([
            _record("zeta", "task", ("fixed",)),
            _record("alpha", "task", ("initial-hard",)),
        ])

        self.assertEqual(list(result["by_strategy"]), ["alpha", "zeta"])
        transitions = result["by_strategy"]["alpha"]["transitions"]
        self.assertEqual(
            [(row["from_route"], row["to_route"]) for row in transitions[:4]],
            [
                ("start", "cheap"),
                ("start", "expensive"),
                ("start", "fixed"),
                ("start", "error"),
            ],
        )
        hard = result["by_strategy"]["alpha"]
        self.assertEqual(len(hard["paths"]), 1)
        self.assertEqual(
            hard["paths"][0]["states"],
            ["start", "expensive", "accept"],
        )
        self.assertEqual(hard["paths"][0]["task_count"], 1)
        self.assertEqual(hard["paths"][0]["attempted_calls"], 1)
        self.assertEqual(hard["rates"]["cheap_route_rate"], {
            "numerator": 0,
            "denominator": 1,
            "denominator_scope": "completed_initial_route_decisions",
            "value": 0.0,
        })
        for name in (
            "first_pass_acceptance_rate",
            "escalation_rate",
            "escalation_recovery_rate",
        ):
            self.assertEqual(hard["rates"][name]["denominator"], 0)
            self.assertIsNone(hard["rates"][name]["value"])
        for rate in result["by_strategy"]["zeta"]["rates"].values():
            self.assertEqual(rate["denominator"], 0)
            self.assertIsNone(rate["value"])

    def test_unknown_usage_and_cost_remain_incomplete(self):
        result = derive_routing_flow([
            _record("fixed", "task-1", ("fixed",)),
            _record(
                "fixed",
                "task-2",
                ("fixed",),
                total_only=True,
                unknown_cost=True,
            )
        ])
        launched = _transition(
            result["by_strategy"]["fixed"], "start", "fixed"
        )

        self.assertEqual(launched["usage"]["total_tokens"], 12)
        self.assertFalse(launched["usage"]["completeness"]["breakdown"])
        self.assertEqual(
            launched["usage"]["completeness"]["missing_breakdown_calls"], 1
        )
        self.assertIsNone(launched["cost"]["total_usd"])
        self.assertEqual(launched["cost"]["known_usd"], 0.4)
        self.assertEqual(launched["cost"]["completeness"]["missing_calls"], 1)
        path = _path(
            result["by_strategy"]["fixed"], ("start", "fixed", "accept")
        )
        self.assertEqual(path["task_count"], 2)
        self.assertEqual(path["attempted_calls"], 2)
        self.assertEqual(path["failed_attempted_calls"], 0)
        self.assertEqual(path["usage"]["total_tokens"], 12)
        self.assertFalse(path["usage"]["completeness"]["breakdown"])
        self.assertIsNone(path["cost"]["total_usd"])
        self.assertEqual(path["cost"]["known_usd"], 0.4)

    def test_failed_launched_call_and_predecision_error_are_distinct(self):
        failed = _record(
            "fixed",
            "execution",
            ("fixed",),
            error_stage="execution",
        )
        routing = _record("fixed", "routing", (), error_stage="routing")
        strategy = derive_routing_flow([routing, failed])["by_strategy"]["fixed"]

        launched = _transition(strategy, "start", "fixed")
        self.assertEqual(launched["failed_launched_calls"], 1)
        self.assertIsNone(launched["usage"]["total_tokens"])
        self.assertIsNone(launched["cost"]["total_usd"])
        self.assertEqual(_transition(strategy, "fixed", "error")["task_count"], 1)
        self.assertEqual(_transition(strategy, "start", "error")["task_count"], 1)
        self.assertEqual(
            [(path["states"], path["task_count"]) for path in strategy["paths"]],
            [
                (["start", "fixed", "error"], 1),
                (["start", "error"], 1),
            ],
        )
        failed_path = _path(strategy, ("start", "fixed", "error"))
        self.assertEqual(failed_path["attempted_calls"], 1)
        self.assertEqual(failed_path["failed_attempted_calls"], 1)
        self.assertIsNone(failed_path["usage"]["total_tokens"])
        self.assertIsNone(failed_path["cost"]["total_usd"])
        predecision_path = _path(strategy, ("start", "error"))
        self.assertEqual(predecision_path["attempted_calls"], 0)
        self.assertEqual(predecision_path["usage"]["total_tokens"], 0)
        ledger = strategy["task_transition_ledger"]
        self.assertEqual(ledger[0]["gate_id"], "fixed-method")
        self.assertEqual(ledger[0]["resulting_call_id"], "call-1")
        self.assertEqual(ledger[1]["gate_id"], "execution")
        self.assertIsNone(ledger[1]["resulting_call_id"])
        self.assertEqual(ledger[2]["gate_id"], "routing")
        self.assertEqual(ledger[2]["path"], ["start", "error"])

    def test_error_edges_follow_execution_not_evaluation_outcome(self):
        strategy = derive_routing_flow([
            _record(
                "routes",
                "easy-error",
                ("initial-easy",),
                error_stage="execution",
            ),
            _record(
                "routes",
                "escalation-error",
                ("initial-easy", "escalation"),
                error_stage="execution",
            ),
            _record(
                "routes",
                "hard-error",
                ("initial-hard",),
                error_stage="execution",
            ),
            _record(
                "routes",
                "evaluation-error",
                ("fixed",),
                error_stage="evaluation",
            ),
        ])["by_strategy"]["routes"]

        self.assertEqual(
            _transition(strategy, "cheap", "error")["task_count"],
            1,
        )
        self.assertEqual(
            _transition(strategy, "expensive", "error")["task_count"], 2
        )
        self.assertEqual(_transition(strategy, "fixed", "accept")["task_count"], 1)
        self.assertEqual(strategy["attempted_calls"], 5)
        self.assertEqual(strategy["attributed_calls"], 5)
        self.assertEqual(strategy["rates"], {
            "cheap_route_rate": {
                "numerator": 2,
                "denominator": 3,
                "denominator_scope": "completed_initial_route_decisions",
                "value": 2 / 3,
            },
            "first_pass_acceptance_rate": {
                "numerator": 0,
                "denominator": 2,
                "denominator_scope": "cheap_route_tasks",
                "value": 0.0,
            },
            "escalation_rate": {
                "numerator": 1,
                "denominator": 2,
                "denominator_scope": "cheap_route_tasks",
                "value": 0.5,
            },
            "escalation_recovery_rate": {
                "numerator": 0,
                "denominator": 1,
                "denominator_scope": "escalated_tasks",
                "value": 0.0,
            },
        })
        evaluation_terminal = next(
            row
            for row in strategy["task_transition_ledger"]
            if row["task_id"] == "evaluation-error" and row["sequence"] == 2
        )
        self.assertEqual(evaluation_terminal["gate_id"], "completion")
        self.assertEqual(evaluation_terminal["decision"], "accept")
        self.assertEqual(evaluation_terminal["to_route"], "accept")
        self.assertIsNone(evaluation_terminal["resulting_call_id"])

    def test_rejects_semantically_contradictory_and_duplicate_records(self):
        contradictory = _record("fixed", "task", ("fixed",))
        contradictory["route"] = "initial-hard"
        contradictory["attempts"][0]["route"]["phase"] = "initial-hard"
        with self.assertRaisesRegex(ValueError, "classifier decision"):
            derive_routing_flow([contradictory])

        record = _record("fixed", "task", ("fixed",))
        with self.assertRaisesRegex(ValueError, "Duplicate benchmark record"):
            derive_routing_flow([record, deepcopy(record)])

    def test_cost_overflow_is_a_deliberate_validation_error(self):
        records = [
            _record("fixed", f"task-{index}", ("fixed",))
            for index in range(2)
        ]
        for record in records:
            record["calls"][0]["cost"]["provider_reported_usd"] = 1.7e308
            record["metrics"]["cost"]["provider_reported"] = {
                "known_usd": 1.7e308,
                "total_usd": 1.7e308,
                "complete": True,
                "missing_calls": 0,
            }

        with self.assertRaisesRegex(ValueError, "finite aggregate"):
            derive_routing_flow(records)


if __name__ == "__main__":
    unittest.main()
