import tempfile
from pathlib import Path
import unittest

from pydantic import ValidationError

from smart_ask import Conversation, InputTokenCount, RunMetadata
from smart_ask.conversation.domain import ConversationEvent
from smart_ask.strategy import (
    DEFAULT_TARGET_REGISTRY,
    StrategyBuildError,
    StrategyBuilder,
    StrategyConfig,
    StrategyConfigError,
    load_strategy,
)


class RecordingExecutor:
    def __init__(self, texts=("answer",)):
        self.texts = list(texts)
        self.specs = []

    async def stream(self, spec):
        self.specs.append(spec)
        text = self.texts.pop(0)
        yield ConversationEvent("message_start", {
            "selected_model": spec.target_id,
            "model": spec.target_id,
        })
        yield ConversationEvent("content_start", {
            "index": 0,
            "block": {"type": "text"},
        })
        yield ConversationEvent("content_delta", {
            "index": 0,
            "delta": {"type": "text", "text": text},
        })
        yield ConversationEvent("content_stop", {"index": 0})
        yield ConversationEvent("message_delta", {"stop_reason": "stop"})
        yield ConversationEvent("message_stop")

    async def count_tokens(self, _spec):
        return InputTokenCount(10, "exact")


def metadata(loaded):
    return RunMetadata(loaded.config.name, loaded.digest)


class StrategySchemaV3Tests(unittest.IsolatedAsyncioTestCase):
    def test_all_bundled_strategies_are_v3_and_resolve_trusted_targets(self):
        directory = Path("smart_ask/resources/strategies")
        loaded = [load_strategy(path) for path in sorted(directory.glob("*.yaml"))]

        self.assertTrue(loaded)
        self.assertTrue(all(item.config.schema_version == 3 for item in loaded))
        for item in loaded:
            for target_id in item.config.target_ids:
                self.assertEqual(
                    DEFAULT_TARGET_REGISTRY.resolve(target_id).target_id,
                    target_id,
                )

    def test_v2_and_provider_fields_are_rejected_without_upgrade_shim(self):
        value = {
            "schema_version": 2,
            "name": "unsafe",
            "profiles": {"model": {"target": "local-qwen3-14b"}},
            "method": {"type": "fixed", "role": "writer", "profile": "model"},
            "generation": {
                "type": "openai",
                "base_url": "https://attacker.example",
                "api_key_env": "OPENAI_API_KEY",
            },
        }
        with self.assertRaises(ValidationError):
            StrategyConfig.model_validate(value)

    def test_strategy_cannot_embed_endpoint_secret_name_or_command(self):
        base = {
            "schema_version": 3,
            "name": "safe",
            "profiles": {"model": {"target": "local-qwen3-14b"}},
            "method": {"type": "fixed", "role": "writer", "profile": "model"},
        }
        for field, value in (
            ("base_url", "https://attacker.example"),
            ("api_key_env", "OPENAI_API_KEY"),
            ("command", "sh"),
        ):
            candidate = {**base, field: value}
            with self.subTest(field=field), self.assertRaises(ValidationError):
                StrategyConfig.model_validate(candidate)

    def test_target_snapshot_is_secret_and_endpoint_free(self):
        snapshot = DEFAULT_TARGET_REGISTRY.snapshot(("openai-codex",))
        rendered = repr(snapshot)

        self.assertNotIn("api.openai.com", rendered)
        self.assertNotIn("OPENAI_API_KEY", rendered)
        self.assertIn("configuration_digest", snapshot[0])
        self.assertEqual(
            DEFAULT_TARGET_REGISTRY.required_secret_envs(("openai-codex",)),
            frozenset({"OPENAI_API_KEY"}),
        )

    def test_unknown_target_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "strategy.yaml")
            path.write_text(
                """schema_version: 3
name: unknown-target
profiles:
  model: {target: missing-target}
method: {type: fixed, role: writer, profile: model}
""",
                encoding="utf-8",
            )
            loaded = load_strategy(path)
            with self.assertRaisesRegex(StrategyBuildError, "unknown deployment target"):
                StrategyBuilder(executor=RecordingExecutor()).build_engine(loaded)

    def test_allowed_roots_are_checked_before_prompt_read(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            prompt = Path(outside, "secret.txt")
            prompt.write_text("secret", encoding="utf-8")
            strategy = Path(allowed, "strategy.yaml")
            strategy.write_text(
                f"""schema_version: 3
name: escaped-prompt
profiles:
  model:
    target: local-qwen3-14b
    system_prompt: {{type: file, path: ../{Path(outside).name}/secret.txt}}
method: {{type: fixed, role: writer, profile: model}}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(StrategyConfigError, "outside"):
                load_strategy(strategy, allowed_roots=(allowed,))

    async def test_builder_applies_profile_and_fixed_transforms(self):
        loaded = load_strategy("builtin:python-function-completion-fixed-gemini-self-check")
        executor = RecordingExecutor()
        engine = StrategyBuilder(executor=executor).build_engine(loaded)

        completed = await engine.complete(Conversation.from_text("task"), metadata(loaded))

        self.assertEqual(completed.record.status, "completed")
        spec = executor.specs[0]
        self.assertEqual(spec.target_id, "openrouter-gemini-flash-lite")
        self.assertEqual(spec.profile_id, "model")
        self.assertEqual(spec.conversation.parameters["max_tokens"], 1024)
        latest = spec.conversation.latest_human_instruction()[0]
        self.assertIn("ESCALATE_NOW", latest)
        self.assertTrue(spec.conversation.system)

    async def test_builder_routes_classifier_and_generation_through_same_scope(self):
        loaded = load_strategy("builtin:claude-code-groq-difficulty")
        executor = RecordingExecutor(texts=('{"d":"easy"}', "answer"))
        completed = await StrategyBuilder(executor=executor).build_engine(
            loaded
        ).complete(Conversation.from_text("task"), metadata(loaded))

        self.assertEqual(
            [spec.target_id for spec in executor.specs],
            ["groq-oss-20b", "groq-oss-20b"],
        )
        self.assertEqual(
            [call.profile_id for call in completed.record.model_calls],
            ["classifier", "easy"],
        )
        self.assertEqual(len(completed.record.provider_requests), 2)


if __name__ == "__main__":
    unittest.main()
