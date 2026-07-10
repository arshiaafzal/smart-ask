import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from smart_ask.application import SmartAsk
from smart_ask.domain import (
    ExecutionRequest,
    ModelResult,
    RouteResult,
    RoutingEvent,
    Task,
)
from smart_ask.metrics import StatsCollector
from smart_ask.methods import DifficultyRoutingMethod, LLMDifficultyClassifier
from smart_ask.routing import SmartRouter


class OneShotMethod:
    requires_response_text = False

    def __init__(self):
        self.contexts = []

    def route(self, task, context):
        self.contexts.append(context)
        if context.attempts:
            return RouteResult(action="accept")
        return RouteResult(
            action="execute",
            model="chosen-model",
            prompt="transformed: " + task.prompt,
            role="writer",
            phase="initial-easy",
            routing_events=(RoutingEvent("test", "easy", "test route"),),
        )


class RecordingExecutor:
    captures_output = True

    def __init__(self):
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return ModelResult(request.model, "answer")


class NeverAcceptMethod:
    requires_response_text = False

    def __init__(self):
        self.calls = 0

    def route(self, task, context):
        self.calls += 1
        return RouteResult(
            action="execute",
            model="model",
            prompt=task.prompt,
            role="writer",
            phase="initial-hard",
        )


class UsageExecutor(RecordingExecutor):
    def execute(self, request):
        self.calls.append(request)
        return ModelResult(
            request.model,
            "answer",
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


class FailingExecutor:
    captures_output = True

    def execute(self, request):
        raise RuntimeError("provider down")


class ClassifierExecutor(UsageExecutor):
    def execute(self, request):
        self.calls.append(request)
        return ModelResult(
            request.model,
            '{"d":"easy"}',
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


class ApplicationTests(unittest.TestCase):
    def test_router_plans_without_generation_and_returns_stats(self):
        collector = StatsCollector()
        classifier = LLMDifficultyClassifier(
            ClassifierExecutor(),
            stats_collector=collector,
            model="classifier-model",
            prompt_prefix="Classify:\n",
            fallback="raise",
            max_prompt_chars=1200,
            max_tokens=20,
            temperature=0.0,
        )
        router = SmartRouter(
            DifficultyRoutingMethod(
                classifier,
                easy_model="easy-model",
                hard_model="hard-model",
            ),
            max_attempts=1,
            strategy_id="router-strategy",
            stats_collector=collector,
        )

        route, stats = router.plan_with_stats(Task("hello", task_id="turn-1"))

        self.assertEqual(route.model, "easy-model")
        self.assertEqual(stats.task_id, "turn-1")
        self.assertEqual(stats.strategy_id, "router-strategy")
        self.assertEqual(stats.interaction_count, 1)
        self.assertEqual(stats.generation_attempts, 0)
        self.assertEqual(stats.calls[0].channel, "classifier")

    def test_application_can_compose_an_existing_router(self):
        collector = StatsCollector()
        router = SmartRouter(
            OneShotMethod(),
            max_attempts=1,
            strategy_id="router-strategy",
            stats_collector=collector,
        )

        app = SmartAsk.from_router(router, UsageExecutor())
        result, stats = app.run_with_stats(Task("hello"))

        self.assertIs(app.router, router)
        self.assertEqual(result.text, "answer")
        self.assertEqual(stats.total_tokens, 10)

    def test_application_requires_an_explicit_positive_attempt_limit(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SmartAsk(OneShotMethod(), RecordingExecutor(), max_attempts=True)

    def test_application_rejects_malformed_collaborator_contracts(self):
        class MissingRoute:
            requires_response_text = False

        class MissingCaptureFlag:
            def execute(self, request):
                return ModelResult(request.model, "answer")

        with self.assertRaisesRegex(TypeError, "callable route"):
            SmartAsk(MissingRoute(), RecordingExecutor(), max_attempts=1)
        with self.assertRaisesRegex(TypeError, "captures_output"):
            SmartAsk(OneShotMethod(), MissingCaptureFlag(), max_attempts=1)

    def test_application_rejects_invalid_runtime_outputs(self):
        class InvalidRouteMethod:
            requires_response_text = False

            def route(self, task, context):
                return "model"

        class InvalidResultExecutor:
            captures_output = True

            def execute(self, request):
                return "answer"

        with self.assertRaisesRegex(TypeError, "RouteResult"):
            SmartAsk(
                InvalidRouteMethod(),
                RecordingExecutor(),
                max_attempts=1,
            ).run(Task("hello"))
        with self.assertRaisesRegex(TypeError, "ModelResult"):
            SmartAsk(
                OneShotMethod(),
                InvalidResultExecutor(),
                max_attempts=1,
            ).run(Task("hello"))

    def test_run_routes_then_executes_transformed_prompt(self):
        method = OneShotMethod()
        executor = RecordingExecutor()
        app = SmartAsk(method, executor, max_attempts=1)

        result = app.run(Task("hello"))

        self.assertEqual(result, ModelResult("chosen-model", "answer"))
        self.assertEqual(executor.calls, [
            ExecutionRequest("chosen-model", "transformed: hello", "writer"),
        ])
        self.assertFalse(method.contexts[0].attempts)
        self.assertEqual(len(method.contexts[1].attempts), 1)
        self.assertEqual(method.contexts[1].routing_events[0].source, "test")

    def test_reusing_application_does_not_leak_context(self):
        method = OneShotMethod()
        app = SmartAsk(method, RecordingExecutor(), max_attempts=1)

        app.run(Task("first"))
        app.run(Task("second"))

        initial_contexts = [method.contexts[0], method.contexts[2]]
        self.assertTrue(all(not context.attempts for context in initial_contexts))

    def test_loop_guard_rejects_nonterminating_method(self):
        method = NeverAcceptMethod()
        executor = RecordingExecutor()
        app = SmartAsk(method, executor, max_attempts=2)

        with self.assertRaisesRegex(RuntimeError, "safety limit"):
            app.run(Task("hello"))
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(method.calls, 3)

    def test_callbacks_receive_attempt_numbers(self):
        routes = []
        results = []
        app = SmartAsk(OneShotMethod(), RecordingExecutor(), max_attempts=1)

        app.run_detailed(
            Task("hello"),
            on_route=lambda route, number: routes.append((route.model, number)),
            on_result=lambda result, number: results.append((result.text, number)),
        )

        self.assertEqual(routes, [("chosen-model", 1)])
        self.assertEqual(results, [("answer", 1)])

    def test_run_with_stats_returns_an_immutable_per_run_snapshot(self):
        app = SmartAsk(
            OneShotMethod(),
            UsageExecutor(),
            max_attempts=1,
            strategy_id="test-strategy",
        )

        result, stats = app.run_with_stats(Task("hello", task_id="turn-1"))

        self.assertEqual(result.text, "answer")
        self.assertEqual(stats.task_id, "turn-1")
        self.assertEqual(stats.strategy_id, "test-strategy")
        self.assertEqual(stats.interaction_count, 1)
        self.assertEqual(stats.generation_attempts, 1)
        self.assertEqual(stats.total_tokens, 10)
        self.assertEqual(stats.calls[0].channel, "generation")
        self.assertEqual(stats.calls[0].role, "writer")
        with self.assertRaises(FrozenInstanceError):
            stats.duration_ms = 0

    def test_low_level_scope_keeps_partial_stats_when_run_raises(self):
        app = SmartAsk(OneShotMethod(), FailingExecutor(), max_attempts=1)

        with app.capture_stats(task_id="turn-1") as capture:
            with self.assertRaisesRegex(RuntimeError, "provider down"):
                app.run_detailed(Task("hello"))

        self.assertEqual(capture.stats.interaction_count, 1)
        self.assertEqual(capture.stats.failed_interactions, 1)
        self.assertIsNone(capture.stats.total_tokens)

    def test_explicit_collector_wraps_a_direct_application_executor(self):
        collector = StatsCollector()
        app = SmartAsk(
            OneShotMethod(),
            UsageExecutor(),
            max_attempts=1,
            stats_collector=collector,
        )

        _result, stats = app.run_with_stats(Task("hello"))

        self.assertEqual(stats.interaction_count, 1)
        self.assertEqual(stats.total_tokens, 10)

    def test_manual_model_classifier_must_share_and_uses_the_app_collector(self):
        collector = StatsCollector()
        classifier = LLMDifficultyClassifier(
            ClassifierExecutor(),
            stats_collector=collector,
            model="classifier-model",
            prompt_prefix="Classify:\n",
            fallback="raise",
            max_prompt_chars=1200,
            max_tokens=20,
            temperature=0.0,
        )
        method = DifficultyRoutingMethod(
            classifier,
            easy_model="easy-model",
            hard_model="hard-model",
        )
        app = SmartAsk(
            method,
            UsageExecutor(),
            max_attempts=1,
            stats_collector=collector,
        )

        _result, stats = app.run_with_stats(Task("hello"))

        self.assertEqual(stats.interaction_count, 2)
        self.assertEqual(
            [(call.channel, call.role) for call in stats.calls],
            [("classifier", "classifier"), ("generation", "generator")],
        )
        self.assertEqual(
            set(app.metrics_executors),
            {"classifier", "generation"},
        )
        self.assertTrue(collector.is_instrumented(
            app.metrics_executors["classifier"][0],
            channel="classifier",
        ))
        self.assertTrue(collector.is_instrumented(
            app.metrics_executors["generation"][0],
            channel="generation",
        ))
        with self.assertRaises(TypeError):
            app.metrics_executors["other"] = ()
        with self.assertRaisesRegex(ValueError, "share one StatsCollector"):
            SmartAsk(
                method,
                UsageExecutor(),
                max_attempts=1,
                stats_collector=StatsCollector(),
            )

        class RawMetricsClassifier:
            stats_collector = collector
            executor = ClassifierExecutor()

            def classify(self, _task):
                raise AssertionError("must not execute")

        with self.assertRaisesRegex(ValueError, "metrics-aware classifier"):
            SmartAsk(
                DifficultyRoutingMethod(
                    RawMetricsClassifier(),
                    easy_model="easy-model",
                    hard_model="hard-model",
                ),
                UsageExecutor(),
                max_attempts=1,
                stats_collector=collector,
            )


if __name__ == "__main__":
    unittest.main()
