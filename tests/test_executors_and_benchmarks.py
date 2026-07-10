from types import SimpleNamespace
import importlib
from inspect import Parameter, signature
from pathlib import Path
import subprocess
import unittest

from smart_ask.benchmarks.code_output import extract_code
from smart_ask.benchmarks.humaneval.harness import run_tests
from smart_ask import ExecutionRequest, StrategyBuilder, Task, load_strategy
from smart_ask.executors import HermesExecutor, OpenAIExecutor, OpenRouterExecutor

from tests.helpers import FakeClient, response, usage


ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ROOT / "smart_ask" / "resources" / "strategies"


class ExecutorTests(unittest.TestCase):
    def test_transport_configuration_has_no_constructor_fallbacks(self):
        openrouter_parameters = signature(OpenRouterExecutor).parameters
        self.assertIs(
            openrouter_parameters["default_max_tokens"].default,
            Parameter.empty,
        )
        self.assertIs(
            openrouter_parameters["temperature"].default,
            Parameter.empty,
        )
        hermes_parameters = signature(HermesExecutor).parameters
        self.assertIs(hermes_parameters["provider"].default, Parameter.empty)
        self.assertIs(hermes_parameters["command"].default, Parameter.empty)

    def test_hermes_executor_builds_exact_command(self):
        calls = []

        def runner(command):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        result = HermesExecutor(
            provider="openrouter",
            command="hermes",
            runner=runner,
        ).execute(
            ExecutionRequest(
                "model",
                "prompt",
                "writer",
                max_tokens=20,
                temperature=0,
            )
        )

        self.assertEqual(calls, [[
            "hermes", "chat", "-q", "prompt",
            "-m", "model", "--provider", "openrouter",
        ]])
        self.assertIsNone(result.model)

    def test_hermes_executor_raises_on_nonzero_exit(self):
        def runner(command):
            return SimpleNamespace(returncode=7)

        with self.assertRaises(subprocess.CalledProcessError) as raised:
            HermesExecutor(
                provider="openrouter",
                command="hermes",
                runner=runner,
            ).execute(
                ExecutionRequest("model", "prompt", "writer")
            )
        self.assertEqual(raised.exception.returncode, 7)

    def test_hermes_requires_a_well_formed_runner_result(self):
        executor = HermesExecutor(
            provider="openrouter",
            command="hermes",
            runner=lambda _command: SimpleNamespace(returncode=None),
        )

        with self.assertRaisesRegex(TypeError, "integer returncode"):
            executor.execute(ExecutionRequest("model", "prompt", "writer"))

    def test_openrouter_executor_applies_per_request_generation_overrides(self):
        call_usage = usage(12, 3)
        client = FakeClient([
            response("answer", call_usage, model="provider/actual-model")
        ])
        executor = OpenRouterExecutor(
            client,
            system_prompts={"model": "system"},
            max_tokens={"model": 99},
            default_max_tokens=1024,
            temperature=0.7,
        )

        result = executor.execute(ExecutionRequest(
            "model",
            "prompt",
            "writer",
            max_tokens=20,
            temperature=0,
        ))

        self.assertEqual(result.text, "answer")
        self.assertEqual(result.raw_text, "answer")
        self.assertEqual(result.model, "provider/actual-model")
        self.assertIs(result.usage, call_usage)
        call = client.completions.calls[0]
        self.assertEqual(call["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ])
        self.assertEqual(call["max_tokens"], 20)
        self.assertEqual(call["temperature"], 0)

    def test_openai_executor_uses_native_reasoning_request_fields(self):
        client = FakeClient([
            response("answer", usage(12, 7), model="gpt-5.3-codex")
        ])
        executor = OpenAIExecutor(
            client,
            reasoning_efforts={"gpt-5.3-codex": "high"},
            default_max_tokens=8192,
            reasoning_effort="medium",
        )

        result = executor.execute(ExecutionRequest(
            "gpt-5.3-codex",
            "prompt",
            "writer",
            max_tokens=2048,
            temperature=0,
        ))

        self.assertEqual(result.model, "gpt-5.3-codex")
        self.assertIsNone(result.provider_cost_usd)
        call = client.completions.calls[0]
        self.assertEqual(call["max_completion_tokens"], 2048)
        self.assertEqual(call["reasoning_effort"], "high")
        self.assertNotIn("max_tokens", call)
        self.assertNotIn("temperature", call)

    def test_openrouter_preserves_provider_text(self):
        text = "```python\ncode\n```\nexplanation"
        executor = OpenRouterExecutor(
            FakeClient([response(text)]),
            default_max_tokens=1024,
            temperature=0.0,
        )

        result = executor.execute(ExecutionRequest("model", "prompt", "writer"))

        self.assertEqual(result.text, text)
        self.assertEqual(result.raw_text, text)

    def test_openrouter_only_keeps_evidenced_actual_model_metadata(self):
        for reported_model in (None, "", "   ", " model "):
            with self.subTest(reported_model=reported_model):
                result = OpenRouterExecutor(
                    FakeClient([response("answer", model=reported_model)]),
                    default_max_tokens=1024,
                    temperature=0.0,
                ).execute(ExecutionRequest("requested", "prompt", "writer"))

                self.assertIsNone(result.model)

    def test_openrouter_snapshots_configuration_and_allows_missing_usage(self):
        prompts = {"model": "original system prompt"}
        response_without_usage = SimpleNamespace(
            model="provider/model",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="answer"),
            )],
        )
        client = FakeClient([response_without_usage])
        executor = OpenRouterExecutor(
            client,
            system_prompts=prompts,
            default_max_tokens=1024,
            temperature=0.0,
        )
        prompts["model"] = "mutated"

        result = executor.execute(ExecutionRequest("model", "prompt", "writer"))

        self.assertIsNone(result.usage)
        self.assertEqual(
            client.completions.calls[0]["messages"][0]["content"],
            "original system prompt",
        )

    def test_openrouter_rejects_malformed_text_responses(self):
        no_choices = SimpleNamespace(choices=[], usage=None, model=None)
        non_text = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=["text"]))],
            usage=None,
            model=None,
        )
        for response_value, error in (
            (no_choices, ValueError),
            (non_text, TypeError),
        ):
            with self.subTest(response=response_value):
                executor = OpenRouterExecutor(
                    FakeClient([response_value]),
                    default_max_tokens=1024,
                    temperature=0.0,
                )
                with self.assertRaises(error):
                    executor.execute(
                        ExecutionRequest("model", "prompt", "writer")
                    )


class BenchmarkCompositionTests(unittest.TestCase):
    def _build(self, filename, client):
        loaded = load_strategy(STRATEGIES / filename)
        app = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda _base_url, _api_key: client,
        ).build(loaded)
        return loaded, app

    def _recorded_roles(self, run):
        return [
            *(
                "classifier"
                for event in run.routing_events
                if event.source == "difficulty-classifier"
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

        self.assertEqual(
            run.attempts[0].result.text,
            "```python\ncode()\n```\nESCALATE_NOW",
        )
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

    def test_escalation_metrics_count_each_model_call_once(self):
        client = FakeClient([
            response('{"d":"easy"}', usage(5, 1)),
            response("code\nESCALATE_NOW", usage(10, 4)),
            response("fixed", usage(12, 5)),
        ])
        _loaded, app = self._build(
            "python-function-completion-cascade.yaml",
            client,
        )

        result, stats = app.run_with_stats(Task("task"))

        self.assertEqual(result.text, "fixed")
        self.assertEqual(stats.interaction_count, 3)
        self.assertEqual(stats.generation_attempts, 2)
        self.assertEqual(stats.total_tokens, 37)

    def test_fixed_opus_application_uses_writer_role(self):
        client = FakeClient([response("answer", usage(10, 4))])
        loaded, app = self._build(
            "python-function-completion-fixed-opus.yaml", client
        )

        run = app.run_detailed(Task("task"))

        self.assertIsNone(run.final_result.model)
        self.assertEqual(run.final_route.model, loaded.config.method.model.model)
        self.assertEqual(run.final_route.role, "writer")
        self.assertEqual(run.final_route.phase, "fixed")
        self.assertEqual(self._recorded_roles(run), ["writer"])

    def test_canonical_benchmark_entrypoints_import(self):
        for module in (
            "smart_ask.benchmarks.humaneval.__main__",
            "smart_ask.benchmarks.livebench.__main__",
        ):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))


class HarnessTests(unittest.TestCase):
    def test_humaneval_harness_executes_partial_generated_code(self):
        prompt = "def add(a, b):\n"
        code = "    return a + b\n"
        tests = "def check(candidate):\n    assert candidate(2, 3) == 5"

        self.assertTrue(run_tests(prompt, code, tests, "add", timeout=10))
        self.assertEqual(extract_code("    return 1\n"), "    return 1")
        self.assertEqual(
            extract_code("```python\ncode\n```\nexplanation"),
            "code",
        )


if __name__ == "__main__":
    unittest.main()
