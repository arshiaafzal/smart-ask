import unittest

from smart_ask.application import SmartAsk
from smart_ask.config import EASY_MODEL, HARD_MODEL
from smart_ask.domain import ModelResult, Task
from smart_ask.executors import HermesExecutor, OpenRouterExecutor
from smart_ask.methods import (
    CascadeRoutingMethod,
    DifficultyClassification,
    DifficultyRoutingMethod,
    FixedRoutingMethod,
    LLMDifficultyClassifier,
    MarkerEscalationPolicy,
)
from smart_ask.methods.escalation import (
    DEFAULT_ESCALATION_PREFIX,
    DEFAULT_SELF_CHECK_SUFFIX,
)

from tests.helpers import FakeClient, RecordingExecutor, response, usage


class StaticClassifier:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.calls = 0

    def classify(self, task):
        self.calls += 1
        return DifficultyClassification(self.difficulty, "static")


class ClassifierTests(unittest.TestCase):
    def test_llm_classifier_uses_generic_executor_and_call_constraints(self):
        call_usage = usage()
        client = FakeClient([response('{"d":"hard"}', call_usage)])
        executor = OpenRouterExecutor(client)
        classifier = LLMDifficultyClassifier(executor, max_prompt_chars=5)

        result = classifier.classify(Task("abcdefgh"))

        self.assertEqual(result.difficulty, "hard")
        self.assertIs(result.usage, call_usage)
        call = client.completions.calls[0]
        self.assertTrue(call["messages"][0]["content"].endswith("abcde"))
        self.assertEqual(call["max_tokens"], 20)
        self.assertEqual(call["temperature"], 0)

    def test_llm_classifier_defaults_to_easy_on_bad_response(self):
        for value in ("not json", '{"d":"unexpected"}'):
            with self.subTest(value=value):
                classifier = LLMDifficultyClassifier(
                    OpenRouterExecutor(FakeClient([response(value)]))
                )
                self.assertEqual(classifier.classify(Task("task")).difficulty, "easy")

    def test_invalid_classifier_response_retains_billed_usage(self):
        call_usage = usage(7, 2)
        classifier = LLMDifficultyClassifier(
            OpenRouterExecutor(FakeClient([response("not json", call_usage)]))
        )

        result = classifier.classify(Task("task"))

        self.assertEqual(result.difficulty, "easy")
        self.assertIs(result.usage, call_usage)
        self.assertIs(result.to_routing_event().usage, call_usage)

    def test_llm_classifier_defaults_to_easy_on_executor_error(self):
        classifier = LLMDifficultyClassifier(
            OpenRouterExecutor(FakeClient([RuntimeError("offline")]))
        )

        result = classifier.classify(Task("task"))

        self.assertEqual(result.difficulty, "easy")
        self.assertIn("offline", result.reason)

    def test_llm_classifier_requires_captured_response_text(self):
        with self.assertRaisesRegex(ValueError, "captures response text"):
            LLMDifficultyClassifier(HermesExecutor())

        class UnspecifiedCaptureExecutor:
            def execute(self, request):
                return ModelResult(request.model, '{"d":"easy"}')

        with self.assertRaisesRegex(ValueError, "captures response text"):
            LLMDifficultyClassifier(UnspecifiedCaptureExecutor())


class EscalationPolicyTests(unittest.TestCase):
    def test_marker_must_appear_on_its_own_line(self):
        policy = MarkerEscalationPolicy()

        accepted = policy.assess(ModelResult("model", "mention ESCALATE_NOW in prose"))
        escalated = policy.assess(ModelResult("model", "code\nESCALATE_NOW\n"))

        self.assertEqual(accepted.outcome, "accept")
        self.assertEqual(escalated.outcome, "escalate")

    def test_marker_policy_owns_candidate_and_retry_prompts(self):
        policy = MarkerEscalationPolicy()
        task = Task("task")

        self.assertEqual(
            policy.prepare_candidate_prompt(task),
            "task" + DEFAULT_SELF_CHECK_SUFFIX,
        )
        self.assertEqual(
            policy.prepare_escalation_prompt(task),
            DEFAULT_ESCALATION_PREFIX + "task",
        )


class MethodTests(unittest.TestCase):
    def test_difficulty_method_selects_easy_and_hard_models(self):
        for difficulty, expected in (("easy", EASY_MODEL), ("hard", HARD_MODEL)):
            with self.subTest(difficulty=difficulty):
                classifier = StaticClassifier(difficulty)
                route = DifficultyRoutingMethod(classifier).route(Task("task"))
                self.assertEqual(route.model, expected)
                self.assertEqual(route.phase, f"initial-{difficulty}")
                self.assertEqual(classifier.calls, 1)

    def test_planned_difficulty_route_executes_without_reclassification(self):
        classifier = StaticClassifier("easy")
        executor = RecordingExecutor(["answer"])
        method = DifficultyRoutingMethod(classifier)
        app = SmartAsk(method, executor, max_attempts=1)
        task = Task("task")

        initial_route = app.plan(task)
        run = app.run_detailed(task, initial_route=initial_route)

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(run.final_result.text, "answer")

    def test_fixed_method_records_a_fixed_route_not_a_fake_check(self):
        route = FixedRoutingMethod(HARD_MODEL, "hard").route(Task("task"))

        self.assertEqual(route.model, HARD_MODEL)
        self.assertEqual(route.phase, "fixed")
        self.assertEqual(route.routing_events[0].source, "fixed-method")

    def test_cascade_accepts_successful_easy_response(self):
        executor = RecordingExecutor(["working code"])
        policy = MarkerEscalationPolicy()
        method = CascadeRoutingMethod(StaticClassifier("easy"), policy)

        run = SmartAsk(method, executor, max_attempts=2).run_detailed(Task("task"))

        self.assertEqual(len(run.attempts), 1)
        self.assertEqual(run.final_route.phase, "initial-easy")
        self.assertTrue(executor.calls[0].prompt.endswith(DEFAULT_SELF_CHECK_SUFFIX))
        self.assertEqual(run.routing_events[-1].outcome, "accept")

    def test_cascade_escalates_marker_response(self):
        executor = RecordingExecutor(["draft\nESCALATE_NOW", "fixed"])
        method = CascadeRoutingMethod(
            StaticClassifier("easy"),
            MarkerEscalationPolicy(),
        )

        run = SmartAsk(method, executor, max_attempts=2).run_detailed(Task("task"))

        self.assertEqual(
            [attempt.route.phase for attempt in run.attempts],
            ["initial-easy", "escalation"],
        )
        self.assertEqual(run.final_result.text, "fixed")
        self.assertEqual(executor.calls[1].model, HARD_MODEL)
        self.assertEqual(executor.calls[1].prompt, DEFAULT_ESCALATION_PREFIX + "task")

    def test_cascade_routes_hard_directly(self):
        executor = RecordingExecutor(["answer"])
        method = CascadeRoutingMethod(
            StaticClassifier("hard"),
            MarkerEscalationPolicy(),
        )

        run = SmartAsk(method, executor, max_attempts=2).run_detailed(Task("task"))

        self.assertEqual(len(run.attempts), 1)
        self.assertEqual(run.final_route.phase, "initial-hard")

    def test_cascade_rejects_executor_without_captured_responses(self):
        method = CascadeRoutingMethod(
            StaticClassifier("easy"),
            MarkerEscalationPolicy(),
        )

        with self.assertRaisesRegex(ValueError, "captures response text"):
            SmartAsk(method, HermesExecutor())


if __name__ == "__main__":
    unittest.main()
