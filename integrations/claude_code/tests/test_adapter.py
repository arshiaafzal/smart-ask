import json
from pathlib import Path
import tempfile
import unittest

import httpx

from smart_ask.conversation import ConversationEvent, ConversationRuntime
from smart_ask.strategy import StrategyBuilder, load_strategy
from smart_ask_claude_code import (
    AdapterConfig,
    AdapterConfigError,
    JsonlSink,
    JsonlTraceSink,
    StrategyCatalog,
    create_app,
)
from smart_ask_claude_code.config import MetricsConfig, SecurityConfig


MODEL_ID = "claude-smart-ask-local-qwen"


class RecordingExecutor:
    def __init__(self, actual_model):
        self.actual_model = actual_model
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ConversationEvent("message_start", {"model": self.actual_model})
        yield ConversationEvent("content_start", {
            "index": 0,
            "block": {"type": "text"},
        })
        yield ConversationEvent("content_delta", {
            "index": 0,
            "delta": {"type": "text", "text": "adapter-ok"},
        })
        yield ConversationEvent("content_stop", {"index": 0})
        yield ConversationEvent("usage", {
            "input_tokens": 12,
            "output_tokens": 3,
        })
        yield ConversationEvent("message_delta", {"stop_reason": "stop"})
        yield ConversationEvent("message_stop")

    async def count_tokens(self, request):
        self.requests.append(request)
        return 55


class AdapterContractTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self):
        loaded = load_strategy("builtin:local-qwen")
        executor = RecordingExecutor("qwen3:14b")
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=StrategyBuilder(env={}).build_router(loaded),
            executor=executor,
        )
        config = AdapterConfig(
            schema_version=1,
            strategies=("builtin:local-qwen",),
        )
        catalog = StrategyCatalog.from_config(
            config,
            env={},
            runtime_builder=lambda _loaded: runtime,
        )
        app = create_app(
            config,
            catalog,
            env={"SMART_ASK_CLAUDE_CODE_TOKEN": "local-secret"},
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://adapter.test",
        )
        self.addAsyncCleanup(client.aclose)
        return client, app, executor

    async def test_streams_complete_structured_request_through_smartask(self):
        client, app, executor = await self.make_client()
        body = {
            "model": MODEL_ID,
            "stream": True,
            "max_tokens": 100,
            "system": [{
                "type": "text",
                "text": "system",
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "thinking",
                        "thinking": "thought",
                        "signature": "signature",
                    }],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AA==",
                            },
                        },
                    ],
                    "future_message_field": True,
                },
            ],
            "tools": [{
                "name": "read",
                "description": "read file",
                "input_schema": {"type": "object"},
                "future_tool_field": 1,
            }],
            "future_request_field": {"keep": True},
        }
        response = await client.post(
            "/v1/messages?beta=true",
            headers={
                "Authorization": "Bearer local-secret",
                "x-claude-code-session-id": "session-1",
            },
            json=body,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"adapter-ok", response.content)
        self.assertIn(b'"input_tokens":55', response.content)
        submitted = executor.requests[0].conversation
        self.assertEqual(submitted.system[0]["cache_control"]["type"], "ephemeral")
        self.assertEqual(submitted.messages[0].content[0]["signature"], "signature")
        self.assertEqual(submitted.messages[1].content[1]["data"], "AA==")
        self.assertTrue(
            submitted.messages[1].extensions["future_message_field"]
        )
        self.assertEqual(submitted.tools[0]["extensions"]["future_tool_field"], 1)
        self.assertTrue(submitted.extensions["future_request_field"]["keep"])
        run = next(iter(app.state.strategy_catalog)).runtime.metrics.records[0]
        self.assertEqual(run["attempts"][0]["actual_model"], "qwen3:14b")
        self.assertEqual(run["totals"]["total_tokens"], 15)

    async def test_discovery_and_token_count_use_same_runtime(self):
        client, _app, executor = await self.make_client()
        headers = {"x-api-key": "local-secret"}

        models = await client.get("/v1/models?limit=1000", headers=headers)
        count = await client.post(
            "/v1/messages/count_tokens",
            headers=headers,
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(models.json()["data"][0]["id"], MODEL_ID)
        self.assertEqual(count.json(), {"input_tokens": 55})
        self.assertEqual(executor.requests[-1].model, "qwen3:14b")

    async def test_rejects_malformed_stream_and_token_limits(self):
        client, _app, _executor = await self.make_client()
        headers = {"x-api-key": "local-secret"}
        base = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
        }

        bad_stream = await client.post(
            "/v1/messages",
            headers=headers,
            json={**base, "stream": "yes", "max_tokens": 10},
        )
        bad_limit = await client.post(
            "/v1/messages",
            headers=headers,
            json={**base, "max_tokens": 0},
        )

        self.assertEqual(bad_stream.status_code, 400)
        self.assertEqual(bad_limit.status_code, 400)

    async def test_adapter_source_contains_no_backend_or_routing_policy(self):
        source_root = Path(__file__).parents[1] / "src" / "smart_ask_claude_code"
        source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in source_root.glob("*.py")
        )
        for forbidden in (
            "openrouter",
            "ollama",
            "qwen",
            "hermes",
            "cheap_model",
            "hard_model",
            "api_key_env",
        ):
            self.assertNotIn(forbidden, source)


class DependencyBoundaryTests(unittest.TestCase):
    def test_smartask_core_has_no_claude_adapter_or_asgi_dependency(self):
        root = Path(__file__).parents[3] / "smart_ask"
        source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in root.rglob("*.py")
        )
        for forbidden in (
            "smart_ask_claude_code",
            "claude code",
            "starlette",
            "uvicorn",
            "text/event-stream",
            '"/v1/messages"',
        ):
            self.assertNotIn(forbidden, source)

    def test_jsonl_sink_persists_complete_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            sink = JsonlSink(str(path))
            sink.write({"run": {"run_id": "one"}, "session": {"runs": 1}})
            sink.close()

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"run": {"run_id": "one"}, "session": {"runs": 1}},
            )

    def test_metrics_and_content_traces_cannot_share_a_file(self):
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            MetricsConfig(
                jsonl_path="/tmp/combined.jsonl",
                trace_jsonl_path="/tmp/combined.jsonl",
            )

    def test_trace_sink_uses_one_schema_header_and_small_run_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            sink = JsonlTraceSink(str(path))
            sink.write({
                "schema": "smart-ask.conversation-trace-event/v1",
                "run_id": "full-run-id-one",
                "sequence": 1,
                "event": "run_start",
                "session_id": "session",
            })
            sink.write({
                "schema": "smart-ask.conversation-trace-event/v1",
                "run_id": "full-run-id-one",
                "sequence": 2,
                "event": "context_block",
                "text": "hello",
            })
            sink.write({
                "schema": "smart-ask.conversation-trace-event/v1",
                "run_id": "full-run-id-two",
                "sequence": 1,
                "event": "run_start",
                "session_id": "session",
            })
            sink.write({
                "schema": "smart-ask.conversation-trace-event/v1",
                "run_id": "full-run-id-one",
                "sequence": 3,
                "event": "route",
                "route": {"action": "execute"},
            })
            sink.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0], {
                "event": "trace_start",
                "schema": "smart-ask.conversation-trace-log/v1",
            })
            self.assertEqual(rows[1]["run"], 1)
            self.assertEqual(rows[1]["run_id"], "full-run-id-one")
            self.assertEqual(rows[2], {
                "event": "context_block",
                "run": 1,
                "text": "hello",
            })
            self.assertEqual(rows[3]["run"], 2)
            self.assertEqual(rows[4]["run"], 1)
            self.assertNotIn("schema", rows[4])
            self.assertNotIn("run_id", rows[4])
            self.assertNotIn("sequence", rows[4])

    def test_custom_strategy_cannot_escape_prompt_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "allowed"
            allowed.mkdir()
            (root / "secret.txt").write_text("secret", encoding="utf-8")
            strategy = allowed / "escape.yaml"
            strategy.write_text(
                """schema_version: 2
name: escape-v1
method:
  type: fixed
  role: writer
  model:
    model: local-model
    system_prompt:
      type: file
      path: ../secret.txt
generation:
  type: ollama
""",
                encoding="utf-8",
            )
            config = AdapterConfig(
                schema_version=1,
                strategies=(str(strategy),),
                security=SecurityConfig(
                    allowed_strategy_roots=(str(allowed),),
                ),
            )

            with self.assertRaisesRegex(AdapterConfigError, "prompts"):
                StrategyCatalog.from_config(config, env={})


if __name__ == "__main__":
    unittest.main()
