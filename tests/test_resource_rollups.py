from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import unittest

from smart_ask.metrics import CallStats, PriceQuote, RunStats, TokenUsage
from smart_ask.metrics.rollups import (
    aggregate_record_resources,
    aggregate_resources,
)


def _successful_call(
    run_id,
    ordinal,
    *,
    requested_model,
    actual_model,
    latency_ms,
    channel,
    role,
    usage,
    usage_status,
    usage_diagnostic,
    quote,
    finish_reason="stop",
    output_status="usable",
    output_empty=False,
    requested_max_tokens=None,
    applied_max_tokens=None,
    max_tokens_reached=False,
    provider_cost_usd=None,
):
    return CallStats(
        run_id=run_id,
        call_id=f"{run_id}-call-{ordinal}",
        ordinal=ordinal,
        channel=channel,
        role=role,
        requested_model=requested_model,
        actual_model=actual_model,
        priced_model=actual_model or requested_model,
        status="ok",
        latency_ms=latency_ms,
        started_offset_ms=float((ordinal - 1) * 30),
        usage=usage,
        usage_status=usage_status,
        usage_diagnostic=usage_diagnostic,
        price_quote=quote,
        finish_reason=finish_reason,
        output_status=output_status,
        output_empty=output_empty,
        requested_max_tokens=requested_max_tokens,
        applied_max_tokens=applied_max_tokens,
        max_tokens_reached=max_tokens_reached,
        provider_cost_usd=provider_cost_usd,
    )


def _failed_call(run_id, *, latency_ms=100.0):
    return CallStats(
        run_id=run_id,
        call_id=f"{run_id}-call-1",
        ordinal=1,
        channel="generation",
        role="fixer",
        requested_model="provider/cheap",
        actual_model=None,
        priced_model=None,
        status="error",
        latency_ms=latency_ms,
        started_offset_ms=0.0,
        usage=TokenUsage(),
        usage_status="unavailable",
        usage_diagnostic="provider call failed",
        price_quote=PriceQuote(
            None,
            "unavailable",
            diagnostic="provider call failed",
        ),
        finish_reason="error",
        output_status=None,
        output_empty=None,
        requested_max_tokens=None,
        applied_max_tokens=None,
        max_tokens_reached=False,
        provider_cost_usd=None,
        error_category="timeout",
        error_type="TimeoutError",
        error_message="timed out",
    )


def _runs():
    first_run_id = "run-alpha"
    rich_call = _successful_call(
        first_run_id,
        1,
        requested_model="cheap-alias",
        actual_model="provider/cheap",
        latency_ms=10.0,
        channel="classifier",
        role="classifier",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=4,
            visible_output_tokens=3,
            reasoning_tokens=1,
            cached_input_tokens=2,
            cache_write_input_tokens=1,
        ),
        usage_status="complete",
        usage_diagnostic=None,
        quote=PriceQuote(0.25, "priced", source="unit-test"),
        provider_cost_usd=0.20,
    )
    fallback_call = _successful_call(
        first_run_id,
        2,
        requested_model="fallback-model",
        actual_model=None,
        latency_ms=20.0,
        channel="generation",
        role="generator",
        usage=TokenUsage(total_tokens=8),
        usage_status="total_only",
        usage_diagnostic="provider returned only a total",
        quote=PriceQuote(
            None,
            "unpriced",
            diagnostic="model absent from catalog",
        ),
        finish_reason="length",
        output_status="truncated",
        output_empty=True,
        requested_max_tokens=8,
        applied_max_tokens=8,
        max_tokens_reached=True,
    )
    alpha = RunStats(
        run_id=first_run_id,
        task_id="task-a",
        strategy_id="alpha",
        duration_ms=80.0,
        calls=(rich_call, fallback_call),
        generation_attempts=1,
    )

    second_run_id = "run-beta"
    beta = RunStats(
        run_id=second_run_id,
        task_id="task-b",
        strategy_id="beta",
        duration_ms=100.0,
        calls=(_failed_call(second_run_id),),
    )
    return alpha, beta


def _record(run):
    return {
        "strategy_id": run.strategy_id,
        "metrics": {"identity": {"run_id": run.run_id}},
        "calls": [call.to_dict() for call in run.calls],
    }


class ResourceRollupTests(unittest.TestCase):
    def test_rolls_up_complete_resource_evidence_by_every_dimension(self):
        report = aggregate_resources(_runs()).to_dict()
        total = report["total"]

        self.assertEqual(total["calls"], 3)
        self.assertEqual(total["call_errors"], 1)
        self.assertEqual(total["model_attribution"], {
            "actual_model_calls": 1,
            "requested_model_fallback_calls": 2,
        })
        self.assertEqual(total["tokens"]["known"], {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 22,
            "visible_output_tokens": 3,
            "reasoning_tokens": 1,
            "cached_input_tokens": 2,
            "cache_write_input_tokens": 1,
        })
        self.assertEqual(total["tokens"]["missing_calls"], {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 1,
            "visible_output_tokens": 2,
            "reasoning_tokens": 2,
            "cached_input_tokens": 2,
            "cache_write_input_tokens": 2,
        })
        self.assertEqual(total["tokens"]["usage_error_calls"], 0)
        self.assertEqual(total["cost"]["known_usd"], 0.25)
        self.assertIsNone(total["cost"]["total_usd"])
        self.assertFalse(total["cost"]["complete"])
        self.assertEqual(total["cost"]["missing_calls"], 2)
        self.assertEqual(total["cost"]["unattributed_calls"], 1)
        self.assertEqual(total["cost"]["provider_reported"], {
            "known_usd": 0.20,
            "total_usd": None,
            "complete": False,
            "missing_calls": 2,
        })
        difference = total["cost"]["catalog_estimate_minus_provider"]
        self.assertAlmostEqual(difference["known_usd"], 0.05)
        self.assertEqual(difference["total_usd"], None)
        self.assertEqual(difference["comparable_calls"], 1)
        self.assertEqual(difference["missing_calls"], 2)
        self.assertEqual(
            total["cost"]["by_priced_model"]["provider/cheap"],
            {
                "calls": 1,
                "known_usd": 0.25,
                "total_usd": 0.25,
                "complete": True,
                "missing_calls": 0,
            },
        )
        self.assertEqual(total["latency_ms"]["mean"], 130.0 / 3)
        self.assertEqual(total["latency_ms"]["p50"], 20.0)
        self.assertEqual(total["latency_ms"]["p95"], 100.0)
        self.assertEqual(total["observed_output_throughput"], {
            "tokens_per_second": 300.0,
            "eligible_calls": 1,
            "missing_calls": 2,
        })
        self.assertEqual(total["responses"], {
            "finish_reasons": {"error": 1, "length": 1, "stop": 1},
            "output_statuses": {
                "truncated": 1,
                "unavailable": 1,
                "usable": 1,
            },
            "output_emptiness": {
                "empty": 1,
                "nonempty": 1,
                "unknown": 1,
            },
            "error_categories": {"timeout": 1},
            "max_tokens_reached_calls": 1,
        })

        self.assertEqual(
            list(report["by_model"]["actual"]),
            ["provider/cheap"],
        )
        self.assertEqual(
            list(report["by_model"]["requested_fallback"]),
            ["fallback-model", "provider/cheap"],
        )
        self.assertEqual(
            report["by_model"]["actual"]["provider/cheap"][
                "model_attribution"
            ],
            {
                "actual_model_calls": 1,
                "requested_model_fallback_calls": 0,
            },
        )
        self.assertEqual(
            report["by_model"]["requested_fallback"]["provider/cheap"][
                "model_attribution"
            ],
            {
                "actual_model_calls": 0,
                "requested_model_fallback_calls": 1,
            },
        )
        self.assertEqual(
            list(report["by_channel"]),
            ["classifier", "generation"],
        )
        self.assertEqual(
            list(report["by_role"]),
            ["classifier", "fixer", "generator"],
        )
        self.assertEqual(list(report["by_strategy"]), ["alpha", "beta"])
        model_calls = sum(
            group["calls"]
            for attribution in report["by_model"].values()
            for group in attribution.values()
        )
        self.assertEqual(model_calls, total["calls"])
        for dimension in ("by_channel", "by_role", "by_strategy"):
            self.assertEqual(
                sum(group["calls"] for group in report[dimension].values()),
                total["calls"],
            )

    def test_input_order_does_not_change_report(self):
        alpha, beta = _runs()
        forward = aggregate_resources([alpha, beta]).to_dict()
        reverse = aggregate_resources([beta, alpha]).to_dict()
        self.assertEqual(forward, reverse)
        self.assertEqual(
            json.dumps(forward, sort_keys=True),
            json.dumps(reverse, sort_keys=True),
        )

    def test_record_ledgers_produce_the_same_report_without_catalog_objects(self):
        runs = _runs()
        records = [_record(run) for run in reversed(runs)]
        self.assertEqual(
            aggregate_record_resources(records).to_dict(),
            aggregate_resources(runs).to_dict(),
        )

    def test_serialized_cost_is_grouped_by_priced_model(self):
        alpha, _ = _runs()
        record = deepcopy(_record(alpha))

        report = aggregate_record_resources([record]).to_dict()
        cost = report["by_model"]["actual"]["provider/cheap"]["cost"]
        self.assertEqual(
            cost["by_priced_model"]["provider/cheap"]["known_usd"],
            0.25,
        )

    def test_empty_report_is_complete_and_json_compatible(self):
        report = aggregate_resources([]).to_dict()
        self.assertEqual(report["total"]["calls"], 0)
        self.assertEqual(report["total"]["cost"]["total_usd"], 0.0)
        self.assertTrue(report["total"]["cost"]["complete"])
        self.assertEqual(report["total"]["latency_ms"], {
            "mean": None,
            "p50": None,
            "p95": None,
        })
        self.assertEqual(report["by_model"], {
            "actual": {},
            "requested_fallback": {},
        })
        json.dumps(report)

    def test_counts_usage_normalization_errors_separately_from_missing_usage(self):
        run_id = "run-usage-error"
        call = _successful_call(
            run_id,
            1,
            requested_model="model",
            actual_model="model",
            latency_ms=1.0,
            channel="generation",
            role="generator",
            usage=TokenUsage(),
            usage_status="error",
            usage_diagnostic="usage normalization failed",
            quote=PriceQuote(
                None,
                "error",
                diagnostic="usage unavailable for pricing",
            ),
        )
        run = RunStats(
            run_id=run_id,
            task_id="task",
            strategy_id="strategy",
            duration_ms=1.0,
            calls=(call,),
            generation_attempts=1,
        )

        total = aggregate_resources([run]).to_dict()["total"]
        self.assertEqual(total["tokens"]["usage_error_calls"], 1)
        self.assertEqual(total["tokens"]["missing_calls"]["total_tokens"], 1)
        self.assertEqual(total["cost"]["pricing_error_calls"], 1)

    def test_report_is_immutable_and_projections_are_fresh(self):
        report = aggregate_resources(_runs())
        with self.assertRaises(FrozenInstanceError):
            report._total = None

        first = report.to_dict()
        first["total"]["calls"] = 999
        self.assertEqual(report.to_dict()["total"]["calls"], 3)

    def test_rejects_duplicate_runs_and_wrong_types(self):
        alpha, _ = _runs()
        with self.assertRaisesRegex(ValueError, "run_id values must be unique"):
            aggregate_resources([alpha, alpha])
        with self.assertRaisesRegex(TypeError, r"runs\[0\]"):
            aggregate_resources([object()])
        with self.assertRaisesRegex(TypeError, "iterable"):
            aggregate_resources(None)

        record = _record(alpha)
        with self.assertRaisesRegex(ValueError, "run_id values must be unique"):
            aggregate_record_resources([record, record])
        with self.assertRaisesRegex(TypeError, r"records\[0\]"):
            aggregate_record_resources([object()])

    def test_numeric_overflow_is_a_deliberate_validation_error(self):
        runs = []
        for index in range(2):
            run_id = f"overflow-{index}"
            call = _successful_call(
                run_id,
                1,
                requested_model="model",
                actual_model="model",
                latency_ms=1.7e308,
                channel="generation",
                role="generator",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
                usage_status="complete",
                usage_diagnostic=None,
                quote=PriceQuote(1.7e308, "priced", source="test"),
                provider_cost_usd=1.7e308,
            )
            runs.append(RunStats(
                run_id=run_id,
                task_id=f"task-{index}",
                strategy_id="strategy",
                duration_ms=1.7e308,
                calls=(call,),
                generation_attempts=1,
            ))

        with self.assertRaisesRegex(ValueError, "finite aggregate"):
            aggregate_resources(runs).to_dict()

    def test_representable_extreme_mean_and_throughput_remain_available(self):
        huge_run_id = "huge-throughput"
        huge_call = _successful_call(
            huge_run_id,
            1,
            requested_model="model",
            actual_model="model",
            latency_ms=1e308,
            channel="generation",
            role="generator",
            usage=TokenUsage(
                total_tokens=10**400,
                visible_output_tokens=10**400,
            ),
            usage_status="total_only",
            usage_diagnostic="provider returned only a total",
            quote=PriceQuote(
                None,
                "unavailable",
                diagnostic="pricing requires a token breakdown",
            ),
        )
        huge_run = RunStats(
            run_id=huge_run_id,
            task_id="huge",
            strategy_id="strategy",
            duration_ms=1e308,
            calls=(huge_call,),
            generation_attempts=1,
        )
        huge_report = aggregate_resources([huge_run]).to_dict()["total"]
        self.assertAlmostEqual(
            huge_report["observed_output_throughput"]["tokens_per_second"],
            1e95,
        )

        tiny_runs = []
        for index in range(2):
            run_id = f"tiny-latency-{index}"
            tiny_call = _successful_call(
                run_id,
                1,
                requested_model="model",
                actual_model="model",
                latency_ms=5e-324,
                channel="generation",
                role="generator",
                usage=TokenUsage(),
                usage_status="unavailable",
                usage_diagnostic="usage missing",
                quote=PriceQuote(
                    None,
                    "unavailable",
                    diagnostic="usage missing",
                ),
            )
            tiny_runs.append(RunStats(
                run_id=run_id,
                task_id=f"tiny-{index}",
                strategy_id="strategy",
                duration_ms=5e-324,
                calls=(tiny_call,),
                generation_attempts=1,
            ))
        tiny_report = aggregate_resources(tiny_runs).to_dict()["total"]
        self.assertEqual(tiny_report["latency_ms"]["mean"], 5e-324)


if __name__ == "__main__":
    unittest.main()
