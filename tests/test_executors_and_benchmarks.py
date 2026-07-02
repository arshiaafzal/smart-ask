from types import SimpleNamespace
import importlib
from pathlib import Path
import subprocess
import unittest

from cost import TokenTracker
from harness import run_tests, strip_fences as harness_strip_fences
from smart_ask import ExecutionRequest, StrategyBuilder, Task, load_strategy
from smart_ask.executors import HermesExecutor, OpenRouterExecutor
from smart_ask.executors.openrouter import _strip_fences

from tests.helpers import FakeClient, response, usage


ROOT = Path(__file__).resolve().parent.parent


class ExecutorTests(unittest.TestCase):
    def test_token_tracker_can_opt_into_unpriced_usage(self):
        call_usage = usage(7, 2)

        with self.assertRaisesRegex(ValueError, "Unknown model"):
            TokenTracker().record("custom/model", "classifier", call_usage)

        tracker = TokenTracker(allow_unpriced=True)
        self.assertIsNone(tracker.record("custom/model", "classifier", call_usage))
        self.assertTrue(tracker.has_unpriced_calls())
        self.assertEqual(tracker.n_calls(), 1)
        self.assertEqual(tracker.by_model()["custom/model"]["prompt_tokens"], 7)
        self.assertIsNone(tracker.by_model()["custom/model"]["cost_usd"])

    def test_hermes_executor_builds_exact_command(self):
        calls = []

        def runner(command):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        result = HermesExecutor(runner=runner).execute(
            ExecutionRequest(
                "model",
                "prompt",
                max_tokens=20,
                temperature=0,
            )
        )

        self.assertEqual(calls, [[
            "hermes", "chat", "-q", "prompt",
            "-m", "model", "--provider", "openrouter",
        ]])
        self.assertEqual(result.model, "model")
        self.assertEqual(result.returncode, 0)

    def test_hermes_executor_raises_on_nonzero_exit(self):
        def runner(command):
            return SimpleNamespace(returncode=7)

        with self.assertRaises(subprocess.CalledProcessError) as raised:
            HermesExecutor(runner=runner).execute(ExecutionRequest("model", "prompt"))
        self.assertEqual(raised.exception.returncode, 7)

    def test_openrouter_executor_applies_per_request_generation_overrides(self):
        call_usage = usage(12, 3)
        client = FakeClient([response("answer", call_usage)])
        executor = OpenRouterExecutor(
            client,
            system_prompts={"model": "system"},
            max_tokens={"model": 99},
            temperature=0.7,
        )

        result = executor.execute(ExecutionRequest(
            "model",
            "prompt",
            max_tokens=20,
            temperature=0,
        ))

        self.assertEqual(result.text, "answer")
        self.assertEqual(result.raw_text, "answer")
        self.assertIs(result.usage, call_usage)
        call = client.completions.calls[0]
        self.assertEqual(call["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ])
        self.assertEqual(call["max_tokens"], 20)
        self.assertEqual(call["temperature"], 0)

    def test_strip_fences_preserves_unfenced_leading_indentation(self):
        self.assertEqual(_strip_fences("    indented\n"), "    indented")
        self.assertEqual(_strip_fences("```python\ncode\n```"), "code")


class BenchmarkCompositionTests(unittest.TestCase):
    def _build(self, filename, client):
        loaded = load_strategy(ROOT / "strategies" / filename)
        app = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda _base_url, _api_key: client,
        ).build(loaded)
        return loaded, app

    def _recorded_roles(self, run):
        return [
            *(
                event.role
                for event in run.routing_events
                if event.role and event.usage is not None
            ),
            *(attempt.route.role for attempt in run.attempts),
        ]

    def test_hard_route_uses_classifier_and_writer_without_prompt_leakage(self):
        client = FakeClient([
            response('{"d":"hard"}', usage(5, 1)),
            response("answer", usage(10, 4)),
        ])
        loaded, app = self._build("python-function-completion-cascade.yaml", client)

        run = app.run_detailed(Task("task"))

        self.assertEqual(run.routing_events[0].outcome, "hard")
        self.assertEqual(run.final_route.phase, "initial-hard")
        self.assertFalse(any(attempt.route.phase == "escalation" for attempt in run.attempts))
        self.assertEqual(self._recorded_roles(run), ["classifier", "writer"])
        classifier_messages = client.completions.calls[0]["messages"]
        self.assertEqual(len(classifier_messages), 1)
        self.assertEqual(classifier_messages[0]["role"], "user")
        self.assertTrue(classifier_messages[0]["content"].endswith("task"))
        self.assertEqual(client.completions.calls[0]["max_tokens"], 20)
        self.assertEqual(client.completions.calls[0]["temperature"], 0)
        self.assertEqual(client.completions.calls[1]["messages"], [
            {
                "role": "system",
                "content": loaded.resolve_prompt(loaded.config.method.hard.system_prompt),
            },
            {"role": "user", "content": "task"},
        ])

    def test_easy_route_uses_self_check_prompt_and_generator_role(self):
        client = FakeClient([
            response('{"d":"easy"}', usage(5, 1)),
            response("answer", usage(10, 4)),
        ])
        loaded, app = self._build("python-function-completion-cascade.yaml", client)

        run = app.run_detailed(Task("task"))

        self.assertEqual(run.final_route.phase, "initial-easy")
        self.assertEqual(run.routing_events[-1].outcome, "accept")
        self.assertEqual(self._recorded_roles(run), ["classifier", "generator"])
        easy_prompt = client.completions.calls[1]["messages"][1]["content"]
        suffix = loaded.resolve_prompt(loaded.config.method.escalation.self_check_suffix)
        self.assertEqual(easy_prompt.count(suffix), 1)

    def test_escalation_retains_raw_marker_and_retries_with_opus(self):
        client = FakeClient([
            response('{"d":"easy"}', usage(5, 1)),
            response("```python\ncode()\n```\nESCALATE_NOW", usage(10, 4)),
            response("fixed", usage(12, 5)),
        ])
        loaded, app = self._build("python-function-completion-cascade.yaml", client)

        run = app.run_detailed(Task("task"))

        self.assertEqual(run.attempts[0].result.text, "code()")
        self.assertIn("ESCALATE_NOW", run.attempts[0].result.raw_text)
        self.assertEqual(
            [attempt.route.phase for attempt in run.attempts],
            ["initial-easy", "escalation"],
        )
        self.assertTrue(any(attempt.route.phase == "escalation" for attempt in run.attempts))
        self.assertEqual(self._recorded_roles(run), ["classifier", "generator", "fixer"])
        self.assertEqual(
            client.completions.calls[2]["messages"][1]["content"],
            loaded.resolve_prompt(loaded.config.method.escalation.escalation_prefix) + "task",
        )

    def test_fixed_opus_application_uses_writer_role(self):
        client = FakeClient([response("answer", usage(10, 4))])
        loaded, app = self._build(
            "python-function-completion-fixed-opus.yaml", client
        )

        run = app.run_detailed(Task("task"))

        self.assertEqual(run.final_result.model, loaded.config.method.model.model)
        self.assertEqual(run.final_route.role, "writer")
        self.assertEqual(run.final_route.phase, "fixed")
        self.assertEqual(self._recorded_roles(run), ["writer"])

    def test_benchmark_entrypoints_import_directly(self):
        for module in (
            "benchmarks.humaneval.run_product",
            "benchmarks.livebench.run_product",
            "benchmarks.livebench.run_opus_baseline",
        ):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))


class HarnessTests(unittest.TestCase):
    def test_humaneval_harness_executes_partial_generated_code(self):
        prompt = "def add(a, b):\n"
        code = "    return a + b\n"
        tests = "def check(candidate):\n    assert candidate(2, 3) == 5"

        self.assertTrue(run_tests(prompt, code, tests, "add"))
        self.assertEqual(harness_strip_fences("    return 1\n"), "    return 1")


if __name__ == "__main__":
    unittest.main()
