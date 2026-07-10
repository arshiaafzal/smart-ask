from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import unittest

from smart_ask import ExecutionRequest, ModelResult
from smart_ask._numeric import checked_fsum
from smart_ask.metrics import (
    DEFAULT_PRICE_CATALOG,
    METRICS_WIRE_SCHEMA,
    PriceCatalog,
    PriceQuote,
    RunStats,
    StatsCollector,
    TokenUsage,
    aggregate_metric_payloads,
    aggregate_stats,
    normalize_usage,
    price_usage,
)


def make_catalog(prices=None):
    return PriceCatalog(
        catalog_id="test-prices-v1",
        effective_date="2026-07-07",
        source="test-suite",
        prices=prices or {"model": {"input": 0.1, "output": 0.2}},
    )


class SequenceExecutor:
    captures_output = True

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def execute(self, request):
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StatisticsTests(unittest.TestCase):
    def test_capture_records_canonical_call_evidence(self):
        ticks = iter((0, 1_000_000, 5_000_000, 8_000_000))
        collector = StatsCollector(
            price_catalog=make_catalog(),
            clock=lambda: next(ticks),
            run_id_factory=lambda: "run-test",
        )
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "answer",
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
                provider_cost_usd=1.1,
            ),
        ]), "generation")

        with collector.capture(
            strategy_id="strategy",
            task_id="task",
        ) as capture:
            executor.execute(ExecutionRequest("model", "prompt", "writer"))

        stats = capture.stats
        self.assertEqual(stats.run_id, "run-test")
        self.assertEqual(stats.task_id, "task")
        self.assertEqual(stats.strategy_id, "strategy")
        self.assertEqual(stats.duration_ms, 8.0)
        self.assertEqual(stats.interaction_count, 1)
        self.assertEqual(stats.total_tokens, 10)
        self.assertAlmostEqual(stats.total_cost_usd, 1.3)
        self.assertAlmostEqual(stats.total_provider_cost_usd, 1.1)
        self.assertIsInstance(capture.calls, tuple)

        record = capture.calls[0]
        self.assertEqual(record.request.role, "writer")
        self.assertEqual(record.result.text, "answer")
        self.assertEqual(record.stats.call_id, "call-1")
        self.assertEqual(record.stats.run_id, "run-test")
        self.assertEqual(record.stats.channel, "generation")
        self.assertEqual(record.stats.role, "writer")
        self.assertEqual(record.stats.priced_model, "model")
        self.assertEqual(record.stats.usage_status, "complete")
        self.assertEqual(record.stats.price_quote.status, "priced")
        self.assertEqual(record.stats.price_quote.source, "test-prices-v1")
        self.assertEqual(record.stats.provider_cost_usd, 1.1)
        self.assertEqual(
            stats.to_dict()["cost"]["provider_reported"]["total_usd"],
            1.1,
        )
        self.assertEqual(
            stats.to_dict()["calls"][0]["cost"]["provider_reported_usd"],
            1.1,
        )
        self.assertIs(record.stats.price_quote.catalog, collector.price_catalog)
        self.assertEqual(record.stats.started_offset_ms, 1.0)
        self.assertEqual(record.stats.latency_ms, 4.0)
        self.assertEqual(record.stats.finish_reason, "unknown")
        self.assertEqual(record.stats.output_status, "usable")
        self.assertIs(record.stats.output_empty, False)
        self.assertIsNone(record.stats.max_tokens_reached)
        self.assertTrue(capture.closed)
        self.assertIs(capture.stats, stats)
        self.assertIs(capture.calls, capture.calls)
        self.assertFalse(hasattr(capture, "append"))
        self.assertFalse(hasattr(capture, "reserve_call"))
        with self.assertRaises(FrozenInstanceError):
            record.result = None
        with self.assertRaises(AttributeError):
            collector.price_catalog = None
        with self.assertRaises(AttributeError):
            collector.pricer = lambda _model, _usage: None
        with self.assertRaisesRegex(ValueError, "timing"):
            replace(stats, duration_ms=4.0)
        duplicate_id = replace(stats.calls[0], ordinal=2)
        with self.assertRaisesRegex(ValueError, "call_id"):
            replace(stats, calls=(stats.calls[0], duplicate_id))
        with self.assertRaisesRegex(ValueError, "run_id"):
            replace(
                stats,
                calls=(replace(stats.calls[0], run_id="different-run"),),
            )
        with self.assertRaisesRegex(TypeError, "TokenUsage"):
            replace(stats.calls[0], usage={})
        with self.assertRaisesRegex(ValueError, "complete usage"):
            replace(stats.calls[0], usage_diagnostic="unnecessary")
        for unsupported_status in ("truncated", "refused"):
            with self.subTest(unsupported_status=unsupported_status):
                with self.assertRaisesRegex(ValueError, "output_status"):
                    replace(
                        stats.calls[0],
                        finish_reason="stop",
                        output_status=unsupported_status,
                        max_tokens_reached=False,
                    )

    def test_empty_output_is_orthogonal_to_finish_and_refusal(self):
        collector = StatsCollector(price_catalog=make_catalog())
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 1024,
                    "reasoning_tokens": 1024,
                },
                finish_reason="length",
                visible_output_tokens=0,
                reasoning_tokens=1024,
            ),
            ModelResult(
                "model",
                "",
                finish_reason="length",
                refusal="Request refused.",
            ),
        ]), "generation")

        with collector.capture(run_id="response-evidence-run") as capture:
            executor.execute(ExecutionRequest("model", "first", "writer"))
            executor.execute(ExecutionRequest("model", "second", "writer"))

        qwen_like, refused = capture.stats.calls
        self.assertEqual(qwen_like.output_status, "truncated")
        self.assertIs(qwen_like.output_empty, True)
        self.assertEqual(refused.output_status, "refused")
        self.assertIs(refused.output_empty, True)
        self.assertEqual(capture.stats.output_emptiness, {"empty": 2})
        self.assertEqual(
            capture.stats.to_dict()["responses"]["output_emptiness"],
            {"empty": 2},
        )
        self.assertEqual(
            aggregate_metric_payloads([capture.stats.to_dict()]).output_emptiness,
            {"empty": 2},
        )

    def test_requested_actual_and_priced_models_are_distinct(self):
        catalog = make_catalog({
            "provider/actual": {"input": 0.1, "output": 0.2},
        })
        collector = StatsCollector(price_catalog=catalog)
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "provider/actual",
                "answer",
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            ),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest(
                "logical/requested",
                "prompt",
                "writer",
            ))

        stats = capture.stats
        call = stats.calls[0]
        self.assertEqual(call.requested_model, "logical/requested")
        self.assertEqual(call.actual_model, "provider/actual")
        self.assertEqual(call.priced_model, "provider/actual")
        self.assertEqual(
            stats.interactions_by_requested_model,
            {"logical/requested": 1},
        )
        self.assertEqual(
            stats.interactions_by_actual_model,
            {"provider/actual": 1},
        )
        self.assertEqual(
            stats.interactions_by_priced_model,
            {"provider/actual": 1},
        )
        with self.assertRaisesRegex(ValueError, "priced_model"):
            replace(call, priced_model="unrelated/model")

        fallback_collector = StatsCollector(price_catalog=None)
        fallback_executor = fallback_collector.wrap(SequenceExecutor([
            ModelResult(
                None,
                "answer",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            ),
        ]), "generation")
        with fallback_collector.capture() as fallback_capture:
            fallback_executor.execute(ExecutionRequest(
                "logical/requested",
                "prompt",
                "writer",
            ))
        fallback_call = fallback_capture.stats.calls[0]
        self.assertIsNone(fallback_call.actual_model)
        self.assertEqual(fallback_call.priced_model, "logical/requested")
        with self.assertRaisesRegex(ValueError, "priced_model"):
            replace(fallback_call, priced_model="unrelated/model")

    def test_total_only_usage_preserves_total_and_marks_breakdown_incomplete(self):
        collector = StatsCollector(price_catalog=make_catalog())
        executor = collector.wrap(SequenceExecutor([
            ModelResult("model", "answer", usage={"total_tokens": 9}),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest("model", "prompt", "writer"))

        stats = capture.stats
        call = stats.calls[0]
        self.assertEqual(call.usage_status, "total_only")
        self.assertEqual(call.usage.total_tokens, 9)
        self.assertTrue(call.usage.total_complete)
        self.assertFalse(call.usage.breakdown_complete)
        self.assertTrue(stats.total_usage_complete)
        self.assertFalse(stats.usage_breakdown_complete)
        self.assertEqual(stats.total_tokens, 9)
        self.assertIsNone(stats.total_cost_usd)
        self.assertIn("breakdown", call.usage_diagnostic)
        with self.assertRaisesRegex(ValueError, "trimmed diagnostic"):
            replace(call, usage_diagnostic="   ")
        self.assertEqual(
            aggregate_metric_payloads([stats.to_dict()]).interactions,
            1,
        )

    def test_failed_calls_keep_partial_run_metrics_and_diagnostics(self):
        collector = StatsCollector(price_catalog=make_catalog())
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "first",
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            ),
            RuntimeError("provider down"),
        ]), "generation")

        with collector.capture(task_id="task") as capture:
            executor.execute(ExecutionRequest("model", "first", "writer"))
            with self.assertRaisesRegex(RuntimeError, "provider down"):
                executor.execute(ExecutionRequest("model", "second", "fixer"))

        stats = capture.stats
        failed = stats.calls[1]
        self.assertEqual(stats.interaction_count, 2)
        self.assertEqual(stats.failed_interactions, 1)
        self.assertEqual(stats.known_total_tokens, 7)
        self.assertIsNone(stats.total_tokens)
        self.assertFalse(stats.total_usage_complete)
        self.assertAlmostEqual(stats.known_cost_usd, 0.9)
        self.assertIsNone(stats.total_cost_usd)
        self.assertEqual(failed.role, "fixer")
        self.assertEqual(failed.error_type, "RuntimeError")
        self.assertEqual(failed.error_category, "unknown")
        self.assertEqual(failed.finish_reason, "error")
        self.assertEqual(failed.usage_status, "unavailable")
        self.assertIn("failed", failed.usage_diagnostic)
        self.assertIn("failed", failed.price_quote.diagnostic)
        wire_summary = aggregate_metric_payloads([stats.to_dict()])
        self.assertEqual(wire_summary.failed_interactions, 1)
        self.assertIsNone(wire_summary.total_tokens)
        invalid_error_category = stats.to_dict(include_calls=False)
        invalid_error_category["interactions"]["errors_by_category"] = {
            "connection": 1,
        }
        with self.assertRaisesRegex(ValueError, "unknown values.*connection"):
            aggregate_metric_payloads([invalid_error_category])

        malformed = StatsCollector(price_catalog=make_catalog())
        malformed_executor = malformed.wrap(SequenceExecutor([
            ValueError("OpenRouter response must contain choices"),
        ]), "generation")
        with malformed.capture() as malformed_capture:
            with self.assertRaises(ValueError):
                malformed_executor.execute(
                    ExecutionRequest("model", "prompt", "writer")
                )
        self.assertEqual(
            malformed_capture.stats.calls[0].error_category,
            "invalid_response",
        )

    def test_usage_normalization_supports_breakdown_total_and_partial_shapes(self):
        sources = (
            SimpleNamespace(prompt_tokens=4, completion_tokens=3),
            {"prompt_tokens": 4, "completion_tokens": 3},
            {"input_tokens": 4, "output_tokens": 3},
        )

        for source in sources:
            with self.subTest(source=source):
                usage = normalize_usage(source)
                self.assertTrue(usage.breakdown_complete)
                self.assertTrue(usage.total_complete)
                self.assertEqual(usage.total_tokens, 7)

        total_only = normalize_usage({"total_tokens": 11})
        self.assertTrue(total_only.total_complete)
        self.assertFalse(total_only.breakdown_complete)
        self.assertEqual(total_only.total_tokens, 11)

        partial = normalize_usage({"prompt_tokens": 4})
        self.assertEqual(partial.prompt_tokens, 4)
        self.assertFalse(partial.total_complete)
        self.assertFalse(partial.breakdown_complete)
        self.assertFalse(normalize_usage(None).any_known)

        detailed = normalize_usage({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "reasoning_tokens": 3,
            "cached_tokens": 7,
        })
        self.assertEqual(detailed.reasoning_tokens, 3)
        self.assertEqual(detailed.cached_input_tokens, 7)

        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            normalize_usage({"prompt_tokens": 1.5, "completion_tokens": 2})
        with self.assertRaisesRegex(ValueError, "total_tokens"):
            TokenUsage(3, 2, 99)
        impossible_details = (
            (
                {"completion_tokens": 3, "reasoning_tokens": 4},
                "reasoning_tokens",
            ),
            (
                {"completion_tokens": 3, "visible_output_tokens": 4},
                "visible_output_tokens",
            ),
            (
                {
                    "completion_tokens": 3,
                    "visible_output_tokens": 2,
                    "reasoning_tokens": 2,
                },
                r"visible_output_tokens \+ reasoning_tokens",
            ),
            (
                {"prompt_tokens": 3, "cached_input_tokens": 4},
                "cached_input_tokens",
            ),
            (
                {
                    "prompt_tokens": 3,
                    "cached_input_tokens": 2,
                    "cache_write_input_tokens": 2,
                },
                r"cached_input_tokens \+ cache_write_input_tokens",
            ),
            (
                {"total_tokens": 3, "reasoning_tokens": 99},
                "reasoning_tokens",
            ),
            (
                {
                    "prompt_tokens": 2,
                    "total_tokens": 3,
                    "visible_output_tokens": 2,
                },
                "visible_output_tokens",
            ),
            (
                {
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "cached_input_tokens": 2,
                },
                "cached_input_tokens",
            ),
            (
                {
                    "total_tokens": 10,
                    "visible_output_tokens": 10,
                    "cached_input_tokens": 10,
                },
                "input and output token details",
            ),
        )
        for fields, message in impossible_details:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, message):
                    TokenUsage(**fields)

    def test_telemetry_failures_are_nonfatal_and_diagnostic(self):
        def broken_pricer(_model, _usage):
            raise RuntimeError(" pricing service failed ")

        collector = StatsCollector(
            price_catalog=None,
            pricer=broken_pricer,
        )
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "answer",
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            ),
        ]), "generation")

        with collector.capture() as capture:
            result = executor.execute(ExecutionRequest("model", "prompt", "writer"))

        call = capture.stats.calls[0]
        self.assertEqual(result.text, "answer")
        self.assertEqual(capture.stats.total_tokens, 7)
        self.assertEqual(call.price_quote.status, "error")
        self.assertIn("pricing service failed", call.price_quote.diagnostic)
        self.assertEqual(
            call.price_quote.diagnostic,
            call.price_quote.diagnostic.strip(),
        )

        invalid_usage = StatsCollector()
        invalid_executor = invalid_usage.wrap(SequenceExecutor([
            ModelResult("model", "answer", usage={"prompt_tokens": -1}),
        ]), "generation")
        with invalid_usage.capture() as invalid_capture:
            invalid_executor.execute(ExecutionRequest("model", "prompt", "writer"))
        invalid_call = invalid_capture.stats.calls[0]
        self.assertEqual(invalid_call.usage_status, "error")
        self.assertIn("normalization failed", invalid_call.usage_diagnostic)

        reconciliation = StatsCollector(price_catalog=make_catalog())
        reconciliation_executor = reconciliation.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "answer",
                usage={"prompt_tokens": 2, "completion_tokens": 3},
                reasoning_tokens=99,
            ),
        ]), "generation")
        with reconciliation.capture() as reconciliation_capture:
            result = reconciliation_executor.execute(
                ExecutionRequest("model", "prompt", "writer")
            )
        reconciled_call = reconciliation_capture.stats.calls[0]
        self.assertEqual(result.text, "answer")
        self.assertEqual(reconciled_call.status, "ok")
        self.assertEqual(reconciled_call.usage_status, "error")
        self.assertFalse(reconciled_call.usage.any_known)
        self.assertIn("reconciliation failed", reconciled_call.usage_diagnostic)
        self.assertEqual(reconciled_call.price_quote.status, "unavailable")
        self.assertEqual(reconciled_call.telemetry_status, "error")

        disagreement = StatsCollector(price_catalog=make_catalog())
        disagreement_executor = disagreement.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "answer",
                usage={
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "reasoning_tokens": 1,
                },
                reasoning_tokens=2,
            ),
        ]), "generation")
        with disagreement.capture() as disagreement_capture:
            disagreement_executor.execute(
                ExecutionRequest("model", "prompt", "writer")
            )
        disagreement_call = disagreement_capture.stats.calls[0]
        self.assertEqual(disagreement_call.status, "ok")
        self.assertEqual(disagreement_call.usage_status, "error")
        self.assertIn("evidence disagree", disagreement_call.usage_diagnostic)

    def test_requested_and_adapter_applied_token_limits_are_distinct(self):
        collector = StatsCollector(price_catalog=make_catalog())
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "answer",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
                applied_max_tokens=75,
            ),
            RuntimeError("provider down"),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest(
                "model",
                "prompt",
                "writer",
                max_tokens=50,
            ))
            with self.assertRaisesRegex(RuntimeError, "provider down"):
                executor.execute(ExecutionRequest(
                    "model",
                    "prompt",
                    "writer",
                    max_tokens=33,
                ))

        successful, failed = capture.stats.calls
        self.assertEqual(successful.requested_max_tokens, 50)
        self.assertEqual(successful.applied_max_tokens, 75)
        self.assertEqual(failed.requested_max_tokens, 33)
        self.assertIsNone(failed.applied_max_tokens)
        responses = [
            call["response"] for call in capture.stats.to_dict()["calls"]
        ]
        self.assertEqual(responses[0]["requested_max_tokens"], 50)
        self.assertEqual(responses[0]["applied_max_tokens"], 75)
        self.assertEqual(responses[1]["requested_max_tokens"], 33)
        self.assertIsNone(responses[1]["applied_max_tokens"])

    def test_uncaptured_success_remains_unavailable_on_the_wire(self):
        collector = StatsCollector(price_catalog=make_catalog())
        executor = collector.wrap(SequenceExecutor([
            ModelResult(None, "", output_status="unavailable"),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest("model", "prompt", "writer"))

        call = capture.stats.calls[0]
        self.assertEqual(call.status, "ok")
        self.assertEqual(call.output_status, "unavailable")
        self.assertIsNone(call.output_empty)
        self.assertEqual(capture.stats.output_statuses, {"unavailable": 1})
        self.assertEqual(capture.stats.output_emptiness, {"unknown": 1})
        payload = capture.stats.to_dict()
        self.assertEqual(
            payload["calls"][0]["response"]["output_status"],
            "unavailable",
        )
        self.assertEqual(
            aggregate_metric_payloads([payload]).output_statuses,
            {"unavailable": 1},
        )
        self.assertEqual(
            aggregate_metric_payloads([payload]).output_emptiness,
            {"unknown": 1},
        )

    def test_unpriced_model_keeps_tokens_and_cost_reason(self):
        collector = StatsCollector()
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "custom/model",
                "answer",
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
            ),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest(
                "custom/model",
                "prompt",
                "generator",
            ))

        call = capture.stats.calls[0]
        self.assertEqual(capture.stats.total_tokens, 9)
        self.assertIsNone(capture.stats.total_cost_usd)
        self.assertEqual(call.price_quote.status, "unpriced")
        self.assertIn("absent", call.price_quote.diagnostic)
        self.assertEqual(capture.stats.priced_calls_by_source, {})

    def test_price_catalog_is_validated_immutable_and_provenanced(self):
        prices = {"model": {"input": 0.1, "output": 0.2}}
        catalog = make_catalog(prices)
        prices["model"]["input"] = 99
        collector = StatsCollector(price_catalog=catalog)
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "answer",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            ),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest("model", "prompt", "writer"))

        self.assertAlmostEqual(capture.stats.total_cost_usd, 0.3)
        wire_catalog = capture.stats.to_dict()["cost"]["catalogs"][0]
        self.assertEqual(wire_catalog, {
            "catalog_id": "test-prices-v1",
            "effective_date": "2026-07-07",
            "source": "test-suite",
            "prices": {
                "model": {"input": 0.1, "output": 0.2},
            },
        })
        with self.assertRaises(TypeError):
            catalog.prices["model"]["input"] = 1.0
        with self.assertRaisesRegex(ValueError, "contain input/output"):
            make_catalog({"model": {"input": 0.1}})
        with self.assertRaisesRegex(ValueError, "supported optional"):
            make_catalog({
                "model": {"input": 0.1, "output": 0.2, "legacy": 1.0},
            })
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            PriceCatalog("bad-date", "July 7", "test", {})
        with self.assertRaisesRegex(TypeError, "diagnostic"):
            PriceQuote(None, "unavailable", diagnostic="   ")
        with self.assertRaisesRegex(TypeError, "TokenUsage"):
            price_usage("model", {}, catalog)
        with self.assertRaisesRegex(TypeError, "PriceCatalog"):
            price_usage("model", TokenUsage(), {})

    def test_catalog_pricing_accounts_for_cache_reasoning_and_request_rates(self):
        catalog = make_catalog({
            "model": {
                "input": 0.1,
                "output": 0.2,
                "input_cache_read": 0.01,
                "input_cache_write": 0.15,
                "internal_reasoning": 0.3,
                "request": 0.5,
            },
        })
        quote = price_usage(
            "model",
            TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                cached_input_tokens=2,
                cache_write_input_tokens=1,
                reasoning_tokens=3,
            ),
            catalog,
        )

        self.assertEqual(quote.status, "priced")
        self.assertAlmostEqual(quote.cost_usd, 2.67)
        missing_detail = price_usage(
            "model",
            TokenUsage(prompt_tokens=10, completion_tokens=5),
            catalog,
        )
        self.assertEqual(missing_detail.status, "unavailable")
        self.assertIn("cached-input", missing_detail.diagnostic)

    def test_default_catalog_prices_reported_cache_usage_at_snapshot_rates(self):
        gemini = price_usage(
            "google/gemini-2.5-flash-lite",
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=0,
                cached_input_tokens=100,
                cache_write_input_tokens=0,
                reasoning_tokens=0,
            ),
            DEFAULT_PRICE_CATALOG,
        )
        opus_read = price_usage(
            "anthropic/claude-opus-4.8",
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=0,
                cached_input_tokens=100,
                cache_write_input_tokens=0,
            ),
            DEFAULT_PRICE_CATALOG,
        )
        opus_write = price_usage(
            "anthropic/claude-opus-4.8",
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=0,
                cached_input_tokens=0,
                cache_write_input_tokens=100,
            ),
            DEFAULT_PRICE_CATALOG,
        )

        self.assertAlmostEqual(gemini.cost_usd, 0.000001)
        self.assertAlmostEqual(opus_read.cost_usd, 0.00005)
        self.assertAlmostEqual(opus_write.cost_usd, 0.000625)

        codex_mini = price_usage(
            "gpt-5.1-codex-mini",
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=10,
                cached_input_tokens=0,
            ),
            DEFAULT_PRICE_CATALOG,
        )
        codex_large = price_usage(
            "gpt-5.3-codex",
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=10,
                cached_input_tokens=0,
            ),
            DEFAULT_PRICE_CATALOG,
        )
        self.assertAlmostEqual(codex_mini.cost_usd, 0.000045)
        self.assertAlmostEqual(codex_large.cost_usd, 0.000315)

    def test_catalog_arithmetic_overflow_is_a_pricing_error(self):
        quote = price_usage(
            "model",
            TokenUsage(prompt_tokens=10**400, completion_tokens=0),
            make_catalog({"model": {"input": 1.0, "output": 1.0}}),
        )

        self.assertEqual(quote.status, "error")
        self.assertIsNone(quote.cost_usd)
        self.assertIn("finite aggregate", quote.diagnostic)

    def test_catalog_products_avoid_premature_integer_conversion_overflow(self):
        free_quote = price_usage(
            "model",
            TokenUsage(prompt_tokens=10**400, completion_tokens=0),
            make_catalog({"model": {"input": 0.0, "output": 0.0}}),
        )
        tiny_rate_quote = price_usage(
            "model",
            TokenUsage(prompt_tokens=10**400, completion_tokens=0),
            make_catalog({"model": {"input": 1e-100, "output": 0.0}}),
        )

        self.assertEqual(free_quote.status, "priced")
        self.assertEqual(free_quote.cost_usd, 0.0)
        self.assertEqual(tiny_rate_quote.status, "priced")
        self.assertAlmostEqual(tiny_rate_quote.cost_usd / 1e300, 1.0)

    def test_exact_sum_preserves_representable_mixed_sign_cancellation(self):
        self.assertEqual(
            checked_fsum(
                [1e308, 1e308, -1e308],
                name="test aggregate",
            ),
            1e308,
        )

    def test_oversized_numeric_inputs_raise_schema_errors_not_overflow(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            PriceCatalog(
                "huge-rate",
                "2026-07-07",
                "test",
                {"model": {"input": 10**400, "output": 0}},
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            PriceQuote(10**400, "priced", source="test")
        with self.assertRaisesRegex(ValueError, "provider_cost_usd"):
            ModelResult("model", "answer", provider_cost_usd=10**400)
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            RunStats(
                run_id="huge",
                task_id=None,
                duration_ms=10**400,
                calls=(),
            )

    def test_custom_usage_and_pricing_policies_are_injectable(self):
        catalog = make_catalog({"custom": {"input": 0.02, "output": 0.03}})
        collector = StatsCollector(
            price_catalog=None,
            usage_normalizer=lambda _raw: TokenUsage(2, 3),
            pricer=lambda model, usage: PriceQuote(
                0.42 if model == "custom" and usage.total_tokens == 5 else 0.0,
                "priced",
                catalog=catalog,
            ),
        )
        executor = collector.wrap(SequenceExecutor([
            ModelResult("custom", "answer", usage={"provider": "shape"}),
        ]), "generation")

        with collector.capture() as capture:
            executor.execute(ExecutionRequest("custom", "prompt", "writer"))

        call = capture.stats.calls[0]
        self.assertEqual(capture.stats.total_tokens, 5)
        self.assertEqual(capture.stats.total_cost_usd, 0.42)
        self.assertEqual(call.price_quote.source, "test-prices-v1")
        self.assertEqual(capture.stats.price_catalogs, (catalog,))
        self.assertEqual(
            capture.stats.to_dict()["cost"]["catalogs"],
            [catalog.to_dict()],
        )

    def test_aggregate_and_run_use_consistent_nested_wire_envelopes(self):
        collector = StatsCollector()
        executor = collector.wrap(SequenceExecutor([
            ModelResult("model", "one"),
            ModelResult("model", "two"),
        ]), "generation")
        snapshots = []
        for task_id, role in (("one", "writer"), ("two", "fixer")):
            with collector.capture(task_id=task_id) as capture:
                executor.execute(ExecutionRequest("model", task_id, role))
            snapshots.append(capture.stats)

        summary = aggregate_stats(snapshots)
        run_wire = snapshots[0].to_dict(include_calls=False)
        summary_wire = summary.to_dict()
        self.assertEqual(summary.runs, 2)
        self.assertEqual(summary.interactions, 2)
        self.assertEqual(summary.missing_total_usage_calls, 2)
        self.assertEqual(summary.interactions_by_role, {"fixer": 1, "writer": 1})
        self.assertEqual(summary.interactions_by_requested_model, {"model": 2})
        self.assertEqual(summary.interactions_by_actual_model, {"model": 2})
        self.assertEqual(summary.interactions_by_priced_model, {"model": 2})
        with self.assertRaisesRegex(ValueError, "actual-model"):
            replace(
                summary,
                interactions_by_actual_model_items=(("model", 3),),
            )
        self.assertEqual(run_wire["schema"], METRICS_WIRE_SCHEMA)
        self.assertEqual(summary_wire["schema"], METRICS_WIRE_SCHEMA)
        for section in ("interactions", "usage", "cost", "routing"):
            self.assertEqual(set(run_wire[section]), set(summary_wire[section]))

        wire_summary = aggregate_metric_payloads(
            snapshot.to_dict(include_calls=False) for snapshot in snapshots
        )
        self.assertEqual(wire_summary.to_dict(), summary_wire)

        wrong_scope = {**run_wire, "scope": "summary"}
        with self.assertRaisesRegex(ValueError, "scope must be 'run'"):
            aggregate_metric_payloads([wrong_scope])

        legacy_schema = {**run_wire, "schema": "smart-ask.metrics/v1"}
        with self.assertRaisesRegex(ValueError, "metrics/v2"):
            aggregate_metric_payloads([legacy_schema])

        stale_interactions = dict(run_wire["interactions"])
        stale_interactions["by_model"] = stale_interactions.pop(
            "by_requested_model"
        )
        stale_shape = {**run_wire, "interactions": stale_interactions}
        with self.assertRaisesRegex(ValueError, "by_requested_model"):
            aggregate_metric_payloads([stale_shape])

        invalid_finish_reason = deepcopy(run_wire)
        invalid_finish_reason["responses"]["finish_reasons"] = {"invented": 1}
        with self.assertRaisesRegex(ValueError, "unknown values.*invented"):
            aggregate_metric_payloads([invalid_finish_reason])

        invalid_output_status = deepcopy(run_wire)
        invalid_output_status["responses"]["output_statuses"] = {"lost": 1}
        with self.assertRaisesRegex(ValueError, "unknown values.*lost"):
            aggregate_metric_payloads([invalid_output_status])

        invalid_output_emptiness = deepcopy(run_wire)
        invalid_output_emptiness["responses"]["output_emptiness"] = {"lost": 1}
        with self.assertRaisesRegex(ValueError, "unknown values.*lost"):
            aggregate_metric_payloads([invalid_output_emptiness])

        impossible_truncation = deepcopy(run_wire)
        impossible_truncation["responses"]["output_statuses"] = {
            "truncated": 1,
        }
        with self.assertRaisesRegex(ValueError, "truncated.*length"):
            aggregate_metric_payloads([impossible_truncation])

        impossible_failed_call = deepcopy(run_wire)
        impossible_failed_call["interactions"].update({
            "failed": 1,
            "by_actual_model": {},
            "by_priced_model": {},
            "errors_by_category": {"unknown": 1},
        })
        with self.assertRaisesRegex(ValueError, "error finish reasons"):
            aggregate_metric_payloads([impossible_failed_call])

        detail_without_evidence = deepcopy(run_wire)
        detail_without_evidence["usage"]["known"][
            "visible_output_tokens"
        ] = 99
        with self.assertRaisesRegex(ValueError, "per-call evidence"):
            aggregate_metric_payloads([detail_without_evidence])

        components_exceed_total = deepcopy(run_wire)
        components_exceed_total["usage"]["known"].update({
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 3,
        })
        components_exceed_total["usage"]["total_tokens"] = 3
        components_exceed_total["usage"]["completeness"].update({
            "total": True,
            "breakdown": False,
            "missing_total_calls": 0,
            "missing_breakdown_calls": 1,
        })
        with self.assertRaisesRegex(ValueError, "components exceed"):
            aggregate_metric_payloads([components_exceed_total])

        details_exceed_total = deepcopy(run_wire)
        details_exceed_total["usage"]["known"].update({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 3,
            "visible_output_tokens": 3,
            "cached_input_tokens": 3,
        })
        details_exceed_total["usage"]["total_tokens"] = 3
        details_exceed_total["usage"]["completeness"].update({
            "total": True,
            "breakdown": False,
            "missing_total_calls": 0,
            "missing_breakdown_calls": 1,
            "details": {
                "visible_output_tokens": {"complete": True, "missing_calls": 0},
                "reasoning_tokens": {"complete": False, "missing_calls": 1},
                "cached_input_tokens": {"complete": True, "missing_calls": 0},
                "cache_write_input_tokens": {"complete": False, "missing_calls": 1},
            },
        })
        with self.assertRaisesRegex(ValueError, "input and output details"):
            aggregate_metric_payloads([details_exceed_total])

        impossible_length_status = deepcopy(run_wire)
        impossible_length_status["responses"].update({
            "finish_reasons": {"length": 1},
            "output_statuses": {"usable": 1},
            "max_tokens_reached_calls": 1,
        })
        with self.assertRaisesRegex(ValueError, "compatible output statuses"):
            aggregate_metric_payloads([impossible_length_status])

        for finish_reason in ("refusal", "content_filter"):
            impossible_refusal_status = deepcopy(run_wire)
            impossible_refusal_status["responses"].update({
                "finish_reasons": {finish_reason: 1},
                "output_statuses": {"usable": 1},
            })
            with self.subTest(finish_reason=finish_reason):
                with self.assertRaisesRegex(
                    ValueError,
                    "compatible output statuses",
                ):
                    aggregate_metric_payloads([impossible_refusal_status])

    def test_task_outcomes_are_explicit_mutually_exclusive_and_aggregated(self):
        first = RunStats(
            run_id="run-pass",
            task_id="task-pass",
            duration_ms=1.0,
            calls=(),
        ).with_outcome("passed")
        second = RunStats(
            run_id="run-chat",
            task_id="turn-chat",
            duration_ms=2.0,
            calls=(),
        )

        summary = aggregate_stats((first, second))
        self.assertEqual(summary.outcome_counts, {
            "passed": 1,
            "incorrect": 0,
            "routing_error": 0,
            "execution_error": 0,
            "evaluation_error": 0,
            "unrated": 1,
        })
        self.assertEqual(summary.cumulative_run_duration_ms, 3.0)
        self.assertEqual(first.to_dict()["outcomes"]["passed"], 1)
        self.assertEqual(
            aggregate_metric_payloads((first.to_dict(), second.to_dict()))
            .outcome_counts,
            summary.outcome_counts,
        )
        with self.assertRaisesRegex(ValueError, "task outcome"):
            replace(first, outcome="provider_ok")

        contradictory = deepcopy(first.to_dict(include_calls=False))
        contradictory["outcomes"]["passed"] = 0
        contradictory["outcomes"]["incorrect"] = 1
        contradictory["outcomes"]["unrated"] = 1
        with self.assertRaisesRegex(ValueError, "exactly one outcome"):
            aggregate_metric_payloads((contradictory,))

    def test_aggregation_rejects_wrong_types_duplicate_runs_and_boolean_numbers(self):
        run = RunStats(
            run_id="run-one",
            task_id=None,
            duration_ms=0.0,
            calls=(),
        )

        with self.assertRaisesRegex(TypeError, "RunStats"):
            aggregate_stats([object()])
        with self.assertRaisesRegex(TypeError, "metric mappings|object"):
            aggregate_metric_payloads([object()])
        with self.assertRaisesRegex(ValueError, "run_id.*unique"):
            aggregate_stats([run, run])
        with self.assertRaisesRegex(ValueError, "run_id.*unique"):
            aggregate_metric_payloads([run.to_dict(), run.to_dict()])
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            replace(run, duration_ms=True)
        empty_summary = aggregate_stats([])
        with self.assertRaisesRegex(ValueError, "empty summary"):
            replace(empty_summary, cumulative_run_duration_ms=1.0)
        with self.assertRaisesRegex(ValueError, "known catalog cost"):
            replace(
                empty_summary,
                runs=1,
                known_cost_usd=1.0,
                outcome_counts_items=(("unrated", 1),),
            )

    def test_wire_calls_are_canonical_and_reconciled_with_the_envelope(self):
        ticks = iter((0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 6_000_000))
        collector = StatsCollector(
            price_catalog=make_catalog(),
            clock=lambda: next(ticks),
            run_id_factory=lambda: "wire-run",
        )
        executor = collector.wrap(SequenceExecutor([
            ModelResult(
                "model",
                "one",
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            ),
            ModelResult(
                "model",
                "two",
                usage={"prompt_tokens": 3, "completion_tokens": 1},
            ),
        ]), "generation")
        with collector.capture(task_id="task") as capture:
            executor.execute(ExecutionRequest("model", "one", "writer"))
            executor.execute(ExecutionRequest("model", "two", "writer"))
        payload = capture.stats.to_dict()

        self.assertEqual(aggregate_metric_payloads([payload]).runs, 1)

        unknown_call_field = deepcopy(payload)
        unknown_call_field["calls"][0]["legacy"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            aggregate_metric_payloads([unknown_call_field])

        duplicate_call_id = deepcopy(payload)
        duplicate_call_id["calls"][1]["call_id"] = "call-1"
        with self.assertRaisesRegex(ValueError, "call_id"):
            aggregate_metric_payloads([duplicate_call_id])

        wrong_call_run = deepcopy(payload)
        wrong_call_run["calls"][0]["run_id"] = "different-run"
        with self.assertRaisesRegex(ValueError, "containing run"):
            aggregate_metric_payloads([wrong_call_run])

        noncontiguous_ordinal = deepcopy(payload)
        noncontiguous_ordinal["calls"][1]["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "ordinals"):
            aggregate_metric_payloads([noncontiguous_ordinal])

        boolean_ordinal = deepcopy(payload)
        boolean_ordinal["calls"][0]["ordinal"] = True
        with self.assertRaisesRegex(TypeError, "ordinal"):
            aggregate_metric_payloads([boolean_ordinal])

        overlong_timing = deepcopy(payload)
        overlong_timing["calls"][1]["timing"]["started_offset_ms"] = 6.0
        with self.assertRaisesRegex(ValueError, "timing"):
            aggregate_metric_payloads([overlong_timing])

        call_envelope_drift = deepcopy(payload)
        call_envelope_drift["calls"][0]["channel"] = "classifier"
        with self.assertRaisesRegex(ValueError, "contradict.*envelope"):
            aggregate_metric_payloads([call_envelope_drift])

        invalid_provider_cost = deepcopy(payload)
        invalid_provider_cost["calls"][0]["cost"][
            "provider_reported_usd"
        ] = -1
        with self.assertRaisesRegex(TypeError, "provider_reported_usd"):
            aggregate_metric_payloads([invalid_provider_cost])

        zero_counter = deepcopy(payload)
        zero_counter["interactions"]["by_channel"]["unused"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            aggregate_metric_payloads([zero_counter])

    def test_collector_rejects_missing_scope_and_double_instrumentation(self):
        collector = StatsCollector(require_active_capture=True)
        executor = collector.wrap(
            SequenceExecutor([ModelResult("model", "answer")]),
            "generation",
        )

        with self.assertRaisesRegex(RuntimeError, "outside a metrics capture"):
            executor.execute(ExecutionRequest("model", "prompt", "writer"))
        with self.assertRaisesRegex(ValueError, "already instrumented"):
            collector.wrap(executor, "generation")
        self.assertTrue(
            collector.is_instrumented(executor, channel="generation")
        )
        self.assertFalse(
            collector.is_instrumented(executor, channel="classifier")
        )
        self.assertFalse(StatsCollector().is_instrumented(executor))
        with self.assertRaisesRegex(ValueError, "channel"):
            collector.is_instrumented(executor, channel=" ")
        with self.assertRaisesRegex(TypeError, "execute"):
            collector.wrap(SimpleNamespace(captures_output=True), "generation")
        with self.assertRaisesRegex(TypeError, "captures_output"):
            collector.wrap(
                SimpleNamespace(captures_output="yes", execute=lambda _request: None),
                "generation",
            )

    def test_invalid_executor_result_is_recorded_as_a_failed_call(self):
        collector = StatsCollector()
        executor = collector.wrap(SequenceExecutor([object()]), "generation")

        with collector.capture() as capture:
            with self.assertRaisesRegex(TypeError, "ModelResult"):
                executor.execute(ExecutionRequest("model", "prompt", "writer"))

        call = capture.stats.calls[0]
        self.assertEqual(call.status, "error")
        self.assertEqual(call.error_type, "TypeError")
        self.assertIn("ModelResult", call.error_message)
        self.assertEqual(call.usage_status, "unavailable")
        self.assertEqual(call.price_quote.status, "unavailable")

        outside_capture = collector.wrap(SequenceExecutor([object()]), "classifier")
        with self.assertRaisesRegex(TypeError, "ModelResult"):
            outside_capture.execute(ExecutionRequest("model", "prompt", "reader"))


if __name__ == "__main__":
    unittest.main()
