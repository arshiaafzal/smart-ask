import unittest

from smart_ask.application import SmartAsk
from smart_ask.domain import (
    ExecutionRequest,
    ModelResult,
    RouteResult,
    RoutingEvent,
    Task,
)


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
        return RouteResult(action="execute", model="model", prompt=task.prompt)


class ApplicationTests(unittest.TestCase):
    def test_run_routes_then_executes_transformed_prompt(self):
        method = OneShotMethod()
        executor = RecordingExecutor()
        app = SmartAsk(method, executor)

        result = app.run(Task("hello"))

        self.assertEqual(result, ModelResult("chosen-model", "answer"))
        self.assertEqual(executor.calls, [
            ExecutionRequest("chosen-model", "transformed: hello"),
        ])
        self.assertFalse(method.contexts[0].attempts)
        self.assertEqual(len(method.contexts[1].attempts), 1)
        self.assertEqual(method.contexts[1].routing_events[0].source, "test")

    def test_reusing_application_does_not_leak_context(self):
        method = OneShotMethod()
        app = SmartAsk(method, RecordingExecutor())

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
        app = SmartAsk(OneShotMethod(), RecordingExecutor())

        app.run_detailed(
            Task("hello"),
            on_route=lambda route, number: routes.append((route.model, number)),
            on_result=lambda result, number: results.append((result.text, number)),
        )

        self.assertEqual(routes, [("chosen-model", 1)])
        self.assertEqual(results, [("answer", 1)])


if __name__ == "__main__":
    unittest.main()
