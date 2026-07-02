from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import tempfile
import tomllib
import unittest

from cost import TokenTracker
from smart_ask import Task
from smart_ask.strategy import (
    StrategyBuildError,
    StrategyBuilder,
    StrategyConfig,
    StrategyConfigError,
    load_strategy,
)

from tests.helpers import FakeClient, response, usage


ROOT = Path(__file__).resolve().parent.parent


class ProductPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = runpy.run_path(str(ROOT / "smart-ask"))

    def test_setuptools_installs_exact_command_and_strategy_assets(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        setuptools = metadata["tool"]["setuptools"]

        self.assertEqual(setuptools["script-files"], ["smart-ask"])
        self.assertEqual(
            setuptools["data-files"]["share/smart-ask/strategies"],
            ["strategies/*.yaml"],
        )
        self.assertEqual(
            setuptools["data-files"]["share/smart-ask/prompts"],
            ["prompts/*.txt"],
        )

    def test_default_strategy_falls_back_to_installed_data_layout(self):
        self.assertEqual(
            self.cli["_default_strategy_path"](ROOT, ROOT / "unused-data-root"),
            ROOT / "strategies" / "product.yaml",
        )

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            installed = (
                data_root / "share" / "smart-ask" / "strategies" / "product.yaml"
            )
            installed.parent.mkdir(parents=True)
            installed.write_text("schema_version: 1\n")

            located = self.cli["_default_strategy_path"](
                data_root / "bin",
                data_root / "different-sysconfig-prefix",
            )

        self.assertEqual(located, installed)

    def test_cli_presentation_uses_configured_names_and_unknown_cost(self):
        config = StrategyConfig.model_validate({
            "schema_version": 1,
            "name": "custom-fixed",
            "method": {
                "type": "fixed",
                "decision": "easy",
                "model": {"model": "vendor/custom-model"},
            },
            "generation": {"type": "openrouter"},
        })
        tracker = TokenTracker(allow_unpriced=True)
        tracker.record("custom/classifier", "classifier", usage(3, 1))

        output = StringIO()
        with redirect_stdout(output):
            self.cli["show_welcome"](config)
            self.cli["print_route"](
                "vendor/custom-model",
                "easy",
                "CustomTransport",
                "configured route",
            )
            self.cli["_print_classifier_cost"](tracker, 1, tracker)
        rendered = output.getvalue()

        self.assertIn("smart-ask", rendered)
        self.assertIn("OpenRouter", rendered)
        self.assertIn("custom-model", rendered)
        self.assertIn("CustomTransport", rendered)
        self.assertIn("cost unknown", rendered)
        self.assertIn("Session routing", rendered)
        self.assertNotIn("Gemini", rendered)
        self.assertNotIn("Opus", rendered)
        self.assertNotIn("Hermes", rendered)
        self.assertNotIn("--force-hard", rendered)


class StrategyLoaderTests(unittest.TestCase):
    def test_all_shipped_strategies_are_valid_and_named_uniquely(self):
        strategy_paths = sorted((ROOT / "strategies").glob("*.yaml"))
        loaded = [load_strategy(path) for path in strategy_paths]

        self.assertGreaterEqual(len(loaded), 5)
        self.assertEqual(len({item.config.name for item in loaded}), len(loaded))
        self.assertTrue(all(len(item.digest) == 64 for item in loaded))

        benchmark_names = ("humaneval", "livebench")
        named_assets = [*strategy_paths, *(ROOT / "prompts").glob("*.txt")]
        for path in named_assets:
            with self.subTest(asset=path.name):
                self.assertFalse(any(name in path.name.lower() for name in benchmark_names))
        for item in loaded:
            with self.subTest(strategy=item.config.name):
                self.assertFalse(
                    any(name in item.config.name.lower() for name in benchmark_names)
                )

    def test_prompt_paths_resolve_relative_to_strategy_file(self):
        loaded = load_strategy(
            ROOT / "strategies" / "python-function-completion-difficulty-v1.yaml"
        )

        text = loaded.resolve_prompt(loaded.config.method.classifier.prompt)

        self.assertIn("routing a coding/AI task", text)

    def test_manifest_is_json_serializable_and_snapshots_prompts(self):
        loaded = load_strategy(
            ROOT / "strategies" / "python-function-completion-cascade.yaml"
        )

        manifest = loaded.manifest()

        json.dumps(manifest)
        self.assertEqual(manifest["name"], "python-function-completion-cascade-v1")
        self.assertEqual(len(manifest["prompts"]), 4)
        self.assertTrue(all(item["sha256"] for item in manifest["prompts"]))

    def test_duplicate_and_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.yaml"
            duplicate.write_text(
                "schema_version: 1\nname: first\nname: second\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StrategyConfigError, "duplicate YAML key"):
                load_strategy(duplicate)

            unknown = Path(directory) / "unknown.yaml"
            unknown.write_text(
                """\
schema_version: 1
name: invalid
method:
  type: fixed
  decision: hard
  model: {model: model}
generation: {type: hermes}
surprise: true
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StrategyConfigError, "surprise"):
                load_strategy(unknown)


class StrategyBuilderTests(unittest.TestCase):
    def test_builder_applies_classifier_and_generation_configuration(self):
        client = FakeClient([
            response('{"d":"easy"}', usage(5, 1)),
            response("answer", usage(10, 4)),
        ])
        loaded = load_strategy(
            ROOT / "strategies" / "python-function-completion-difficulty-v1.yaml"
        )
        builder = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda base_url, api_key: client,
        )

        run = builder.build(loaded).run_detailed(Task("task"))

        self.assertEqual(run.final_result.text, "answer")
        self.assertEqual(client.completions.calls[0]["max_tokens"], 20)
        self.assertEqual(client.completions.calls[0]["temperature"], 0.0)
        generation_call = client.completions.calls[1]
        self.assertEqual(generation_call["max_tokens"], 1024)
        self.assertEqual(generation_call["temperature"], 0.0)
        self.assertEqual(generation_call["messages"][0]["role"], "system")

    def test_forced_product_route_does_not_require_classifier_credentials(self):
        calls = []

        def runner(command):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        loaded = load_strategy(ROOT / "strategies" / "product.yaml")
        app = StrategyBuilder(env={}, hermes_runner=runner).build(loaded, force="easy")

        app.run(Task("task"))

        self.assertEqual(calls[0][calls[0].index("-m") + 1], loaded.config.method.easy.model)

    def test_unforced_product_route_requires_classifier_credentials(self):
        loaded = load_strategy(ROOT / "strategies" / "product.yaml")

        with self.assertRaisesRegex(StrategyBuildError, "OPENROUTER_API_KEY"):
            StrategyBuilder(env={}).build(loaded)

    def test_executor_wrapper_receives_classifier_and_generation_roles(self):
        client = FakeClient([])
        roles = []

        def wrapper(executor, role):
            roles.append(role)
            return executor

        loaded = load_strategy(
            ROOT / "strategies" / "python-function-completion-difficulty-v2.yaml"
        )
        StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda base_url, api_key: client,
            executor_wrapper=wrapper,
        ).build(loaded)

        self.assertEqual(roles, ["classifier", "generation"])


if __name__ == "__main__":
    unittest.main()
