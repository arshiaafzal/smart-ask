from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tomllib
import unittest

from smart_ask import (
    CallStats,
    RunStats,
    StatsCollector,
    Task,
    TokenUsage,
    aggregate_stats,
)
from smart_ask import _terminal, cli
from smart_ask.metrics import PriceQuote
from smart_ask.strategy import (
    LoadedStrategy,
    StrategyBuildError,
    StrategyBuilder,
    StrategyConfig,
    StrategyConfigError,
    load_strategy,
)
from smart_ask.strategy.loader import compute_strategy_digest
from smart_ask.strategy.schema import FilePromptConfig, InlinePromptConfig

from tests.helpers import FakeClient, response, usage


ROOT = Path(__file__).resolve().parent.parent
RESOURCE_ROOT = ROOT / "smart_ask" / "resources"
STRATEGY_ROOT = RESOURCE_ROOT / "strategies"
PROMPT_ROOT = RESOURCE_ROOT / "prompts"


def difficulty_config() -> dict:
    return {
        "schema_version": 2,
        "name": "configured-difficulty",
        "method": {
            "type": "difficulty",
            "classifier": {
                "type": "llm",
                "model": "vendor/classifier",
                "executor": {"type": "openrouter"},
                "prompt": {"type": "inline", "text": "Classify this task"},
                "fallback": "easy",
            },
            "easy": {"model": "vendor/easy"},
            "hard": {"model": "vendor/hard"},
        },
        "generation": {"type": "openrouter"},
    }


class ProductPackagingTests(unittest.TestCase):
    def test_console_entrypoint_and_resources_are_package_native(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

        self.assertEqual(
            metadata["project"]["scripts"]["smart-ask"],
            "smart_ask.cli:main",
        )
        setuptools = metadata["tool"]["setuptools"]
        self.assertNotIn("script-files", setuptools)
        self.assertNotIn("data-files", setuptools)
        self.assertEqual(
            setuptools["package-data"]["smart_ask"],
            [
                "resources/prompts/*.txt",
                "resources/strategies/*.yaml",
                "benchmarks/humaneval/README.md",
            ],
        )
        self.assertEqual(setuptools["packages"]["find"]["include"], ["smart_ask*"])

    def test_default_strategy_is_a_packaged_resource(self):
        strategy = load_strategy("builtin:product").path

        self.assertEqual(strategy, STRATEGY_ROOT / "product.yaml")
        self.assertTrue(strategy.is_file())
        self.assertFalse((ROOT / "strategies").exists())
        self.assertFalse((ROOT / "prompts").exists())
        with self.assertRaisesRegex(StrategyConfigError, "unknown bundled"):
            load_strategy("builtin:not-shipped")
        with self.assertRaisesRegex(StrategyConfigError, "path separators"):
            load_strategy("builtin:../product")

    def test_cli_presentation_uses_configured_names_and_unknown_cost(self):
        config = StrategyConfig.model_validate({
            "schema_version": 2,
            "name": "custom-fixed",
            "method": {
                "type": "fixed",
                "role": "generator",
                "model": {"model": "vendor/custom-model"},
            },
            "generation": {"type": "openrouter"},
        })
        turn_stats = RunStats(
            run_id="run-1",
            task_id="turn-1",
            duration_ms=1.0,
            calls=(CallStats(
                run_id="run-1",
                call_id="call-1",
                ordinal=1,
                channel="classifier",
                role="classifier",
                requested_model="custom/classifier",
                actual_model="custom/classifier",
                priced_model="custom/classifier",
                status="ok",
                latency_ms=1.0,
                started_offset_ms=0.0,
                usage=TokenUsage(3, 1, 4),
                usage_status="complete",
                price_quote=PriceQuote(
                    None,
                    "unpriced",
                    diagnostic="model is not present in the price catalog",
                ),
                finish_reason="stop",
                output_status="usable",
                output_empty=False,
                requested_max_tokens=None,
                applied_max_tokens=None,
                max_tokens_reached=False,
                provider_cost_usd=None,
            ),),
        )

        output = StringIO()
        with redirect_stdout(output):
            _terminal.show_welcome(config)
            _terminal.print_route(
                "vendor/custom-model",
                "easy",
                "CustomTransport",
                "configured route",
            )
            _terminal.print_turn_stats(
                turn_stats,
                1,
                aggregate_stats([turn_stats]),
            )
        rendered = output.getvalue()

        self.assertIn("smart-ask", rendered)
        self.assertIn("OpenRouter", rendered)
        self.assertIn("custom-model", rendered)
        self.assertIn("CustomTransport", rendered)
        self.assertIn("cost unknown", rendered)
        self.assertIn("4 tok", rendered)
        self.assertIn("Session", rendered)
        self.assertNotIn("Gemini", rendered)
        self.assertNotIn("Opus", rendered)
        self.assertNotIn("Hermes", rendered)
        self.assertNotIn("--force-hard", rendered)


class StrategyLoaderTests(unittest.TestCase):
    def test_all_shipped_strategies_are_schema_v2_and_named_uniquely(self):
        strategy_paths = sorted(STRATEGY_ROOT.glob("*.yaml"))
        loaded = [load_strategy(path) for path in strategy_paths]

        self.assertGreaterEqual(len(loaded), 5)
        self.assertEqual(len({item.config.name for item in loaded}), len(loaded))
        self.assertTrue(all(item.config.schema_version == 2 for item in loaded))
        self.assertTrue(all(len(item.digest) == 64 for item in loaded))

        for item in loaded:
            classifier = getattr(item.config.method, "classifier", None)
            if classifier is not None:
                self.assertEqual(classifier.fallback, "easy")

        benchmark_names = ("humaneval", "livebench")
        named_assets = [*strategy_paths, *PROMPT_ROOT.glob("*.txt")]
        for path in named_assets:
            with self.subTest(asset=path.name):
                self.assertFalse(any(name in path.name.lower() for name in benchmark_names))
        for item in loaded:
            with self.subTest(strategy=item.config.name):
                self.assertFalse(
                    any(name in item.config.name.lower() for name in benchmark_names)
                )

    def test_shipped_counterfactual_baselines_match_routed_profiles(self):
        for contract in (
            "python-function-completion",
            "python-code-generation",
        ):
            with self.subTest(contract=contract):
                cascade = load_strategy(
                    STRATEGY_ROOT / f"{contract}-cascade.yaml"
                )
                cheap = load_strategy(
                    STRATEGY_ROOT / f"{contract}-fixed-gemini-self-check.yaml"
                )
                expensive = load_strategy(
                    STRATEGY_ROOT / f"{contract}-fixed-opus.yaml"
                )

                self.assertEqual(cheap.config.method.role, "generator")
                self.assertEqual(
                    cheap.config.method.model,
                    cascade.config.method.easy,
                )
                self.assertEqual(
                    cheap.resolve_prompt(cheap.config.method.prompt_suffix),
                    cascade.resolve_prompt(
                        cascade.config.method.escalation.self_check_suffix
                    ),
                )
                self.assertEqual(expensive.config.method.role, "writer")
                self.assertEqual(
                    expensive.config.method.model,
                    cascade.config.method.hard,
                )
                self.assertEqual(
                    cheap.config.generation,
                    cascade.config.generation,
                )
                self.assertEqual(
                    expensive.config.generation,
                    cascade.config.generation,
                )

        difficulty = load_strategy(
            STRATEGY_ROOT / "python-function-completion-difficulty-v1.yaml"
        )
        plain_cheap = load_strategy(
            STRATEGY_ROOT / "python-function-completion-fixed-gemini.yaml"
        )
        self.assertEqual(plain_cheap.config.method.role, "generator")
        self.assertIsNone(plain_cheap.config.method.prompt_prefix)
        self.assertIsNone(plain_cheap.config.method.prompt_suffix)
        self.assertEqual(
            plain_cheap.config.method.model,
            difficulty.config.method.easy,
        )
        self.assertEqual(plain_cheap.config.generation, difficulty.config.generation)

    def test_schema_v2_and_classifier_fallback_are_required(self):
        version_one = difficulty_config()
        version_one["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema_version"):
            StrategyConfig.model_validate(version_one)

        missing_fallback = difficulty_config()
        del missing_fallback["method"]["classifier"]["fallback"]
        with self.assertRaisesRegex(ValueError, "fallback"):
            StrategyConfig.model_validate(missing_fallback)

        null_classifier_limit = difficulty_config()
        null_classifier_limit["method"]["classifier"]["parameters"] = {
            "max_tokens": None,
            "temperature": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            StrategyConfig.model_validate(null_classifier_limit)

        null_executor_default = difficulty_config()
        null_executor_default["generation"]["defaults"] = {
            "max_tokens": 1024,
            "temperature": None,
        }
        with self.assertRaisesRegex(ValueError, "temperature"):
            StrategyConfig.model_validate(null_executor_default)

        inert_classifier_defaults = difficulty_config()
        inert_classifier_defaults["method"]["classifier"]["executor"][
            "defaults"
        ] = {"max_tokens": 999, "temperature": 1.0}
        with self.assertRaisesRegex(ValueError, "defaults"):
            StrategyConfig.model_validate(inert_classifier_defaults)

        obsolete_attempt_limit = difficulty_config()
        obsolete_attempt_limit["max_attempts"] = 1
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            StrategyConfig.model_validate(obsolete_attempt_limit)

    def test_file_prompt_paths_are_portable_relative_declarations(self):
        valid = FilePromptConfig.model_validate({
            "type": "file",
            "path": "../prompts/system.txt",
        })
        self.assertEqual(valid.path, "../prompts/system.txt")

        for path in (
            "/tmp/system.txt",
            "~/system.txt",
            "~other/system.txt",
            r"C:\prompts\system.txt",
            r"relative\system.txt",
            r"C:relative-system.txt",
            " ../prompts/system.txt",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "prompt path"):
                    FilePromptConfig.model_validate({
                        "type": "file",
                        "path": path,
                    })

    def test_prompt_paths_resolve_relative_to_packaged_strategy(self):
        loaded = load_strategy(
            STRATEGY_ROOT / "python-function-completion-difficulty-v1.yaml"
        )

        text = loaded.resolve_prompt(loaded.config.method.classifier.prompt)

        self.assertIn("routing a coding/AI task", text)
        self.assertTrue(loaded.path.is_relative_to(RESOURCE_ROOT))

    def test_manifest_is_json_serializable_and_snapshots_prompts(self):
        loaded = load_strategy(
            STRATEGY_ROOT / "python-function-completion-cascade.yaml"
        )

        manifest = loaded.manifest()

        json.dumps(manifest)
        self.assertEqual(manifest["name"], "python-function-completion-cascade-v1")
        self.assertEqual(
            set(manifest),
            {"name", "digest", "config", "prompts"},
        )
        self.assertEqual(len(manifest["prompts"]), 4)
        self.assertTrue(all(item["sha256"] for item in manifest["prompts"]))
        self.assertTrue(all(
            set(item) == {"declared_path", "sha256", "text"}
            for item in manifest["prompts"]
        ))
        self.assertTrue(all(
            not Path(item["declared_path"]).is_absolute()
            for item in manifest["prompts"]
        ))

        inline = StrategyConfig.model_validate({
            "schema_version": 2,
            "name": "inline-fixed",
            "method": {
                "type": "fixed",
                "role": "writer",
                "model": {"model": "model"},
            },
            "generation": {"type": "openrouter"},
        })
        self.assertEqual(
            compute_strategy_digest(inline, {}),
            compute_strategy_digest(inline, {}),
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            compute_strategy_digest(inline, {"undeclared.txt": "0" * 64})

    def test_loaded_strategy_copies_prompts_and_verifies_its_identity(self):
        loaded = load_strategy(
            STRATEGY_ROOT / "python-function-completion-cascade.yaml"
        )
        prompt_texts = {
            item["declared_path"]: item["text"]
            for item in loaded.manifest()["prompts"]
        }
        reconstructed = LoadedStrategy(
            config=loaded.config,
            path=loaded.path,
            digest=loaded.digest,
            _prompt_texts=prompt_texts,
        )
        classifier_prompt = loaded.config.method.classifier.prompt
        expected = reconstructed.resolve_prompt(classifier_prompt)
        prompt_texts[classifier_prompt.path] = "mutated"

        self.assertEqual(reconstructed.resolve_prompt(classifier_prompt), expected)
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            LoadedStrategy(
                config=loaded.config,
                path=loaded.path,
                digest="0" * 64,
                _prompt_texts={
                    item["declared_path"]: item["text"]
                    for item in loaded.manifest()["prompts"]
                },
            )
        with self.assertRaisesRegex(ValueError, "not declared"):
            reconstructed.resolve_prompt(FilePromptConfig(
                type="file",
                path="../prompts/undeclared.txt",
            ))
        with self.assertRaisesRegex(ValueError, "not declared"):
            reconstructed.resolve_prompt(InlinePromptConfig(
                type="inline",
                text="undeclared",
            ))

    def test_duplicate_and_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.yaml"
            duplicate.write_text(
                "schema_version: 2\nname: first\nname: second\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StrategyConfigError, "duplicate YAML key"):
                load_strategy(duplicate)

            unknown = Path(directory) / "unknown.yaml"
            unknown.write_text(
                """\
schema_version: 2
name: invalid
method:
  type: fixed
  role: writer
  model: {model: model}
generation: {type: hermes}
surprise: true
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StrategyConfigError, "surprise"):
                load_strategy(unknown)


class StrategyBuilderTests(unittest.TestCase):
    def test_force_override_is_closed(self):
        loaded = load_strategy(STRATEGY_ROOT / "product.yaml")
        with self.assertRaisesRegex(StrategyBuildError, "force"):
            StrategyBuilder(env={}).build(loaded, force="other")

    def test_builder_derives_the_closed_method_attempt_limits(self):
        builder = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda _url, _key: FakeClient([]),
        )
        fixed = builder.build(load_strategy(
            STRATEGY_ROOT / "python-function-completion-fixed-opus.yaml"
        ))
        difficulty = builder.build(load_strategy(
            STRATEGY_ROOT / "python-function-completion-difficulty-v1.yaml"
        ))
        cascade = builder.build(load_strategy(
            STRATEGY_ROOT / "python-function-completion-cascade.yaml"
        ))

        self.assertEqual(fixed.max_attempts, 1)
        self.assertEqual(difficulty.max_attempts, 1)
        self.assertEqual(cascade.max_attempts, 2)

    def test_fixed_baseline_can_reproduce_a_prompt_transform(self):
        config = StrategyConfig.model_validate({
            "schema_version": 2,
            "name": "transformed-fixed",
            "method": {
                "type": "fixed",
                "role": "generator",
                "model": {"model": "vendor/easy"},
                "prompt_prefix": {"type": "inline", "text": "before:"},
                "prompt_suffix": {"type": "inline", "text": ":after"},
            },
            "generation": {"type": "openrouter"},
        })
        loaded = LoadedStrategy(
            config=config,
            path=(ROOT / "transformed-fixed.yaml").resolve(),
            digest=compute_strategy_digest(config, {}),
            _prompt_texts={},
        )
        client = FakeClient([response("answer", usage(3, 1))])
        app = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda _base_url, _api_key: client,
        ).build(loaded)

        app.run(Task("task"))

        self.assertEqual(
            client.completions.calls[0]["messages"][-1]["content"],
            "before:task:after",
        )

    def test_builder_snapshots_its_environment(self):
        env = {"OPENROUTER_API_KEY": "original-key"}
        factory_calls = []

        def factory(base_url, api_key):
            factory_calls.append((base_url, api_key))
            return FakeClient([])

        builder = StrategyBuilder(
            env=env,
            openrouter_client_factory=factory,
        )
        env["OPENROUTER_API_KEY"] = "mutated-key"
        builder.build(load_strategy(
            STRATEGY_ROOT / "python-function-completion-fixed-opus.yaml"
        ))

        self.assertEqual(factory_calls[0][1], "original-key")

    def test_builder_applies_classifier_and_generation_configuration(self):
        client = FakeClient([
            response('{"d":"easy"}', usage(5, 1)),
            response("answer", usage(10, 4)),
        ])
        loaded = load_strategy(
            STRATEGY_ROOT / "python-function-completion-difficulty-v1.yaml"
        )
        builder = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda _base_url, _api_key: client,
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

        loaded = load_strategy(STRATEGY_ROOT / "product.yaml")
        app = StrategyBuilder(env={}, hermes_runner=runner).build(loaded, force="easy")

        app.run(Task("task"))

        self.assertEqual(calls[0][calls[0].index("-m") + 1], loaded.config.method.easy.model)

    def test_unforced_product_route_requires_classifier_credentials(self):
        loaded = load_strategy(STRATEGY_ROOT / "product.yaml")

        with self.assertRaisesRegex(StrategyBuildError, "OPENROUTER_API_KEY"):
            StrategyBuilder(env={}).build(loaded)

    def test_builder_uses_the_explicit_stats_collector_for_every_call(self):
        def fully_detailed_usage(prompt_tokens, completion_tokens):
            result = usage(prompt_tokens, completion_tokens)
            result.prompt_tokens_details = SimpleNamespace(
                cached_tokens=0,
                cache_write_tokens=0,
            )
            result.completion_tokens_details = SimpleNamespace(
                reasoning_tokens=0,
            )
            return result

        client = FakeClient([
            response('{"d":"easy"}', fully_detailed_usage(5, 1)),
            response("answer", fully_detailed_usage(10, 4)),
        ])
        collector = StatsCollector()
        loaded = load_strategy(
            STRATEGY_ROOT / "python-function-completion-difficulty-v1.yaml"
        )
        app = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_client_factory=lambda _base_url, _api_key: client,
            stats_collector=collector,
        ).build(loaded)

        result, stats = app.run_with_stats(Task("task", task_id="turn-1"))

        self.assertIs(app.stats_collector, collector)
        self.assertEqual(result.text, "answer")
        self.assertEqual(stats.strategy_id, loaded.config.name)
        self.assertEqual(stats.interaction_count, 2)
        self.assertEqual(
            [(call.channel, call.role) for call in stats.calls],
            [("classifier", "classifier"), ("generation", "generator")],
        )
        self.assertEqual(stats.total_tokens, 20)
        self.assertTrue(stats.cost_complete)
        self.assertAlmostEqual(stats.total_cost_usd, 0.0000035)


if __name__ == "__main__":
    unittest.main()
