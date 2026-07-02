from dataclasses import FrozenInstanceError
import unittest

from smart_ask.domain import (
    Attempt,
    Context,
    ExecutionRequest,
    ModelResult,
    RouteResult,
    RoutingEvent,
    RunResult,
    Task,
)


class DomainTests(unittest.TestCase):
    def test_values_are_immutable(self):
        task = Task("hello")
        with self.assertRaises(FrozenInstanceError):
            task.prompt = "changed"

        request = ExecutionRequest("model", "prompt", max_tokens=20, temperature=0)
        with self.assertRaises(FrozenInstanceError):
            request.max_tokens = 40

    def test_context_exposes_latest_attempt(self):
        route = RouteResult(action="execute", model="model", prompt="prompt")
        result = ModelResult(model="model", text="answer")
        context = Context(attempts=(Attempt(route, result),))

        self.assertIs(context.previous_route, route)
        self.assertIs(context.previous_attempt, result)

    def test_execute_route_requires_model_and_prompt(self):
        with self.assertRaises(ValueError):
            RouteResult(action="execute")

    def test_run_result_returns_final_attempt(self):
        first_route = RouteResult(
            action="execute",
            model="easy",
            prompt="prompt",
            phase="initial-easy",
        )
        final_route = RouteResult(
            action="execute",
            model="hard",
            prompt="prompt",
            phase="escalation",
        )
        first = Attempt(first_route, ModelResult("easy", "draft"))
        final = Attempt(final_route, ModelResult("hard", "answer"))
        event = RoutingEvent("difficulty-classifier", "easy", "simple")
        run = RunResult(Task("prompt"), (first, final), (event,))

        self.assertEqual(run.final_result.text, "answer")
        self.assertEqual(run.final_route.model, "hard")
        self.assertEqual(run.final_route.phase, "escalation")


if __name__ == "__main__":
    unittest.main()
