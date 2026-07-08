from inspect import Parameter, signature
import unittest

from smart_ask.application import SmartAsk
from smart_ask.domain import ModelResult, Task
from smart_ask.executors import HermesExecutor, OpenRouterExecutor
from smart_ask.methods import (
    CascadeRoutingMethod,
    DifficultyClassification,
    DifficultyRoutingMethod,
    EscalationDecision,
    FixedRoutingMethod,
    LLMDifficultyClassifier,
    MarkerEscalationPolicy,
)
from smart_ask.metrics import StatsCollector

from tests.helpers import FakeClient, RecordingExecutor, response, usage


CLASSIFIER_MODEL = "vendor/classifier"
EASY_MODEL = "vendor/easy"
HARD_MODEL = "vendor/hard"
MARKER = "ESCALATE_NOW"
SELF_CHECK_SUFFIX = f"\nCheck the answer. Emit {MARKER} on failure."
ESCALATION_PREFIX = "Repair the previous failed attempt:\n"


class StaticClassifier:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.calls = 0

    def classify(self, task):
        self.calls += 1
        return DifficultyClassification(self.difficulty, "static")


def llm_classifier(responses, fallback="easy", **kwargs):
    collector = kwargs.pop("stats_collector", StatsCollector())
    return LLMDifficultyClassifier(
        OpenRouterExecutor(
            FakeClient(responses),
            default_max_tokens=1024,
            temperature=0.0,
        ),
        stats_collector=collector,
        model=CLASSIFIER_MODEL,
        prompt_prefix="Classify:\n",
        fallback=fallback,
        max_prompt_chars=kwargs.pop("max_prompt_chars", 1200),
        max_tokens=kwargs.pop("max_tokens", 20),
        temperature=kwargs.pop("temperature", 0.0),
        **kwargs,
    )


def marker_policy():
    return MarkerEscalationPolicy(
        marker=MARKER,
        self_check_suffix=SELF_CHECK_SUFFIX,
        escalation_prefix=ESCALATION_PREFIX,
    )


class ClassifierTests(unittest.TestCase):
    def test_llm_classifier_uses_generic_executor_and_call_constraints(self):
        client = FakeClient([response('{"d":"hard"}', usage())])
        executor = OpenRouterExecutor(
            client,
            default_max_tokens=1024,
            temperature=0.0,
        )
        classifier = LLMDifficultyClassifier(
            executor,
            stats_collector=StatsCollector(),
            model=CLASSIFIER_MODEL,
            prompt_prefix="Classify:\n",
            fallback="raise",
            max_prompt_chars=5,
            max_tokens=20,
            temperature=0.0,
        )

        result = classifier.classify(Task("abcdefgh"))

        self.assertEqual(result.difficulty, "hard")
        self.assertEqual(result.model, CLASSIFIER_MODEL)
        call = client.completions.calls[0]
        self.assertTrue(call["messages"][0]["content"].endswith("abcde"))
        self.assertEqual(call["max_tokens"], 20)
        self.assertEqual(call["temperature"], 0)

    def test_classifier_model_prompt_and_fallback_have_no_defaults(self):
        parameters = signature(LLMDifficultyClassifier).parameters

        for name in (
            "stats_collector",
            "model",
            "prompt_prefix",
            "fallback",
            "max_prompt_chars",
            "max_tokens",
            "temperature",
        ):
            with self.subTest(parameter=name):
                self.assertIs(parameters[name].default, Parameter.empty)

    def test_invalid_response_applies_easy_or_hard_fallback(self):
        for fallback in ("easy", "hard"):
            for value in (
                "not json",
                '{"d":"unexpected"}',
                '{"d":"easy","extra":true}',
                '{"d":"easy","d":"hard"}',
                '```json\n{"d":"easy"}\n```',
            ):
                with self.subTest(fallback=fallback, value=value):
                    result = llm_classifier(
                        [response(value)],
                        fallback=fallback,
                    ).classify(Task("task"))

                    self.assertEqual(result.difficulty, fallback)
                    self.assertIn(f"defaulted to {fallback}", result.reason)

    def test_raise_fallback_rejects_invalid_response(self):
        classifier = llm_classifier([response("not json")], fallback="raise")

        with self.assertRaisesRegex(ValueError, "invalid response"):
            classifier.classify(Task("task"))

    def test_classifier_inserts_an_unambiguous_prompt_boundary(self):
        client = FakeClient([response('{"d":"easy"}')])
        classifier = LLMDifficultyClassifier(
            OpenRouterExecutor(
                client,
                default_max_tokens=1024,
                temperature=0.0,
            ),
            stats_collector=StatsCollector(),
            model=CLASSIFIER_MODEL,
            prompt_prefix="Classify:",
            fallback="raise",
            max_prompt_chars=1200,
            max_tokens=20,
            temperature=0.0,
        )

        classifier.classify(Task("task"))

        self.assertEqual(
            client.completions.calls[0]["messages"][0]["content"],
            "Classify:\ntask",
        )

    def test_execution_error_applies_configured_fallback(self):
        for fallback in ("easy", "hard"):
            with self.subTest(fallback=fallback):
                result = llm_classifier(
                    [RuntimeError("offline")],
                    fallback=fallback,
                ).classify(Task("task"))

                self.assertEqual(result.difficulty, fallback)
                self.assertIn("offline", result.reason)

    def test_raise_fallback_preserves_executor_error(self):
        classifier = llm_classifier(
            [RuntimeError("offline")],
            fallback="raise",
        )

        with self.assertRaisesRegex(RuntimeError, "offline"):
            classifier.classify(Task("task"))

    def test_llm_classifier_requires_captured_response_text(self):
        with self.assertRaisesRegex(ValueError, "captures response text"):
            LLMDifficultyClassifier(
                HermesExecutor(provider="openrouter", command="hermes"),
                stats_collector=StatsCollector(),
                model=CLASSIFIER_MODEL,
                prompt_prefix="Classify:\n",
                fallback="raise",
                max_prompt_chars=1200,
                max_tokens=20,
                temperature=0.0,
            )

        class UnspecifiedCaptureExecutor:
            def execute(self, request):
                return ModelResult(request.model, '{"d":"easy"}')

        with self.assertRaisesRegex(TypeError, "captures_output"):
            LLMDifficultyClassifier(
                UnspecifiedCaptureExecutor(),
                stats_collector=StatsCollector(),
                model=CLASSIFIER_MODEL,
                prompt_prefix="Classify:\n",
                fallback="raise",
                max_prompt_chars=1200,
                max_tokens=20,
                temperature=0.0,
            )

    def test_classifier_fallback_covers_an_invalid_executor_result(self):
        class InvalidResultExecutor:
            captures_output = True

            def execute(self, request):
                return '{"d":"easy"}'

        classifier = LLMDifficultyClassifier(
            InvalidResultExecutor(),
            stats_collector=StatsCollector(),
            model=CLASSIFIER_MODEL,
            prompt_prefix="Classify:",
            fallback="easy",
            max_prompt_chars=1200,
            max_tokens=20,
            temperature=0.0,
        )

        result = classifier.classify(Task("task"))

        self.assertEqual(result.difficulty, "easy")
        self.assertIn("ModelResult", result.reason)


class EscalationPolicyTests(unittest.TestCase):
    def test_marker_policy_inputs_have_no_defaults(self):
        parameters = signature(MarkerEscalationPolicy).parameters

        for name in ("marker", "self_check_suffix", "escalation_prefix"):
            with self.subTest(parameter=name):
                self.assertIs(parameters[name].default, Parameter.empty)

    def test_marker_must_appear_on_its_own_line(self):
        policy = marker_policy()

        accepted = policy.assess(ModelResult("model", f"mention {MARKER} in prose"))
        escalated = policy.assess(ModelResult("model", f"code\n{MARKER}\n"))

        self.assertEqual(accepted.outcome, "accept")
        self.assertEqual(escalated.outcome, "escalate")

        for invalid in ("", " marker", "marker\nagain"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "marker"):
                    MarkerEscalationPolicy(
                        invalid,
                        SELF_CHECK_SUFFIX,
                        ESCALATION_PREFIX,
                    )

        with self.assertRaisesRegex(ValueError, "outcome"):
            EscalationDecision("retry", "invalid")

    def test_marker_policy_uses_explicit_candidate_and_retry_prompts(self):
        policy = marker_policy()
        task = Task("task")

        self.assertEqual(
            policy.prepare_candidate_prompt(task),
            "task" + SELF_CHECK_SUFFIX,
        )
        self.assertEqual(
            policy.prepare_escalation_prompt(task),
            ESCALATION_PREFIX + "task",
        )


class MethodTests(unittest.TestCase):
    def test_difficulty_method_requires_and_selects_configured_models(self):
        parameters = signature(DifficultyRoutingMethod).parameters
        self.assertIs(parameters["easy_model"].default, Parameter.empty)
        self.assertIs(parameters["hard_model"].default, Parameter.empty)

        with self.assertRaisesRegex(ValueError, "easy_model"):
            DifficultyRoutingMethod(StaticClassifier("easy"), "", HARD_MODEL)
        with self.assertRaisesRegex(ValueError, "difficulty"):
            DifficultyClassification("medium", "invalid")

        for difficulty, expected in (("easy", EASY_MODEL), ("hard", HARD_MODEL)):
            with self.subTest(difficulty=difficulty):
                classifier = StaticClassifier(difficulty)
                route = DifficultyRoutingMethod(
                    classifier,
                    EASY_MODEL,
                    HARD_MODEL,
                ).route(Task("task"))
                self.assertEqual(route.model, expected)
                self.assertEqual(route.phase, f"initial-{difficulty}")
                self.assertEqual(classifier.calls, 1)

    def test_plan_is_fresh_inspection_and_run_routes_independently(self):
        classifier = StaticClassifier("easy")
        executor = RecordingExecutor(["answer"])
        method = DifficultyRoutingMethod(classifier, EASY_MODEL, HARD_MODEL)
        app = SmartAsk(method, executor, max_attempts=1)
        task = Task("task")

        inspected_route = app.plan(task)
        run = app.run_detailed(task)

        self.assertEqual(inspected_route.model, EASY_MODEL)
        self.assertEqual(classifier.calls, 2)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(run.final_result.text, "answer")

    def test_fixed_method_records_role_without_a_fake_decision(self):
        route = FixedRoutingMethod(HARD_MODEL, role="writer").route(Task("task"))

        self.assertEqual(route.model, HARD_MODEL)
        self.assertEqual(route.role, "writer")
        self.assertEqual(route.phase, "fixed")
        self.assertEqual(route.routing_events[0].source, "fixed-method")
        self.assertEqual(route.routing_events[0].outcome, "fixed")

    def test_fixed_method_applies_explicit_prompt_transform(self):
        route = FixedRoutingMethod(
            HARD_MODEL,
            role="writer",
            prompt_prefix="prefix:",
            prompt_suffix=":suffix",
        ).route(Task("task"))

        self.assertEqual(route.prompt, "prefix:task:suffix")

    def test_cascade_accepts_successful_easy_response(self):
        executor = RecordingExecutor(["working code"])
        method = CascadeRoutingMethod(
            StaticClassifier("easy"),
            marker_policy(),
            EASY_MODEL,
            HARD_MODEL,
        )

        run = SmartAsk(method, executor, max_attempts=2).run_detailed(Task("task"))

        self.assertEqual(len(run.attempts), 1)
        self.assertEqual(run.final_route.phase, "initial-easy")
        self.assertTrue(executor.calls[0].prompt.endswith(SELF_CHECK_SUFFIX))
        self.assertEqual(run.routing_events[-1].outcome, "accept")

    def test_cascade_escalates_marker_response(self):
        executor = RecordingExecutor([f"draft\n{MARKER}", "fixed"])
        method = CascadeRoutingMethod(
            StaticClassifier("easy"),
            marker_policy(),
            EASY_MODEL,
            HARD_MODEL,
        )

        run = SmartAsk(method, executor, max_attempts=2).run_detailed(Task("task"))

        self.assertEqual(
            [attempt.route.phase for attempt in run.attempts],
            ["initial-easy", "escalation"],
        )
        self.assertEqual(run.final_result.text, "fixed")
        self.assertEqual(executor.calls[1].model, HARD_MODEL)
        self.assertEqual(executor.calls[1].role, "fixer")
        self.assertEqual(executor.calls[1].prompt, ESCALATION_PREFIX + "task")

    def test_cascade_routes_hard_directly(self):
        executor = RecordingExecutor(["answer"])
        method = CascadeRoutingMethod(
            StaticClassifier("hard"),
            marker_policy(),
            EASY_MODEL,
            HARD_MODEL,
        )

        run = SmartAsk(method, executor, max_attempts=2).run_detailed(Task("task"))

        self.assertEqual(len(run.attempts), 1)
        self.assertEqual(run.final_route.phase, "initial-hard")

    def test_cascade_rejects_executor_without_captured_responses(self):
        method = CascadeRoutingMethod(
            StaticClassifier("easy"),
            marker_policy(),
            EASY_MODEL,
            HARD_MODEL,
        )

        with self.assertRaisesRegex(ValueError, "captures response text"):
            SmartAsk(
                method,
                HermesExecutor(provider="openrouter", command="hermes"),
                max_attempts=2,
            )


if __name__ == "__main__":
    unittest.main()
