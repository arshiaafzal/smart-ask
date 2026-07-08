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

        request = ExecutionRequest(
            "model",
            "prompt",
            "writer",
            max_tokens=20,
            temperature=0,
        )
        with self.assertRaises(FrozenInstanceError):
            request.max_tokens = 40

    def test_context_exposes_latest_attempt(self):
        route = RouteResult(
            action="execute",
            model="model",
            prompt="prompt",
            role="writer",
            phase="initial-hard",
        )
        result = ModelResult(model="model", text="answer")
        context = Context(attempts=(Attempt(route, result),))

        self.assertIs(context.previous_route, route)
        self.assertIs(context.previous_attempt, result)

    def test_execute_route_requires_model_and_prompt(self):
        with self.assertRaises(ValueError):
            RouteResult(action="execute")
        with self.assertRaisesRegex(ValueError, "role"):
            RouteResult(
                action="execute",
                model="model",
                prompt="prompt",
                phase="initial-hard",
            )
        with self.assertRaisesRegex(ValueError, "phase"):
            RouteResult(
                action="execute",
                model="model",
                prompt="prompt",
                role="writer",
            )

    def test_execution_request_requires_an_explicit_semantic_role(self):
        request = ExecutionRequest("model", "prompt", "classifier")
        self.assertEqual(request.role, "classifier")
        with self.assertRaisesRegex(ValueError, "role"):
            ExecutionRequest("model", "prompt", "")
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            ExecutionRequest("model", "prompt", "writer", max_tokens=0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            ExecutionRequest("model", "prompt", "writer", temperature=3.0)

    def test_requests_and_audit_reasons_cannot_be_blank(self):
        with self.assertRaisesRegex(ValueError, "prompt"):
            Task("   ")
        with self.assertRaisesRegex(ValueError, "prompt"):
            ExecutionRequest("model", "\n", "writer")
        with self.assertRaisesRegex(ValueError, "reason"):
            RoutingEvent("source", "outcome", "")

    def test_run_result_returns_final_attempt(self):
        first_route = RouteResult(
            action="execute",
            model="easy",
            prompt="prompt",
            role="generator",
            phase="initial-easy",
        )
        final_route = RouteResult(
            action="execute",
            model="hard",
            prompt="prompt",
            role="fixer",
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
