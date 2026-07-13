import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from smart_ask.conversation import (
    Conversation,
    ConversationEvent,
    InputTokenCount,
    ModelCallSpec,
    RunMetadata,
    StrategyEngine,
)
from smart_ask.methods import FixedStrategyMethod, ModelProfile, RequestTransform
from smart_ask.strategy import load_strategy
from smart_ask_claude_code import (
    AdapterConfig,
    AdapterConfigError,
    JsonlSink,
    StrategyCatalog,
    TraceSessionSink,
    create_app,
)
from smart_ask_claude_code.config import MetricsConfig, SecurityConfig


MODEL_ID = "claude-smart-ask-local-qwen"


def text_events(text="adapter-ok", *, actual_model="qwen3:14b"):
    return (
        ConversationEvent("message_start", {
            "model": actual_model,
            "selected_model": "qwen3:14b",
        }),
        ConversationEvent("content_start", {
            "index": 0,
            "block": {"type": "text"},
        }),
        ConversationEvent("content_delta", {
            "index": 0,
            "delta": {"type": "text", "text": text},
        }),
        ConversationEvent("content_stop", {"index": 0}),
        ConversationEvent("usage", {
            "input_tokens": 12,
            "output_tokens": 3,
        }),
        ConversationEvent("message_delta", {"stop_reason": "stop"}),
        ConversationEvent("message_stop"),
    )


def tool_events():
    return (
        ConversationEvent("message_start", {
            "model": "qwen3:14b",
            "selected_model": "qwen3:14b",
        }),
        ConversationEvent("content_start", {
            "index": 0,
            "block": {"type": "tool_call", "id": "call-1", "name": "read"},
        }),
        ConversationEvent("content_delta", {
            "index": 0,
            "delta": {
                "type": "tool_arguments_json",
                "json": '{"path":"a.py"}',
            },
        }),
        ConversationEvent("content_stop", {"index": 0}),
        ConversationEvent("usage", {
            "input_tokens": 14,
            "output_tokens": 5,
        }),
        ConversationEvent("message_delta", {"stop_reason": "tool_call"}),
        ConversationEvent("message_stop"),
    )


class RecordingModelCallExecutor:
    def __init__(self, responses=None, *, token_count=55):
        self.responses = list(responses or [text_events()])
        self.token_count = token_count
        self.requests = []
        self.token_requests = []
        self.closed = False

    async def stream(self, spec):
        self.requests.append(spec)
        for event in self.responses.pop(0):
            yield event

    async def count_tokens(self, spec):
        self.token_requests.append(spec)
        return InputTokenCount(self.token_count, "exact")

    async def aclose(self):
        self.closed = True


class BlockingModelCallExecutor(RecordingModelCallExecutor):
    def __init__(self):
        super().__init__(responses=[])
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(self, spec):
        self.requests.append(spec)
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        if False:
            yield ConversationEvent("message_stop")


class FailingModelCallExecutor(RecordingModelCallExecutor):
    async def stream(self, spec):
        self.requests.append(spec)
        raise RuntimeError("provider exploded")
        if False:
            yield ConversationEvent("message_stop")


def engine_for(executor, observer=None):
    method = FixedStrategyMethod(
        profile=ModelProfile(
            profile_id="writer",
            target_id="local-qwen",
            transform=RequestTransform(),
        ),
        role="writer",
        transform=RequestTransform(),
    )
    return StrategyEngine(
        method,
        executor,
        heartbeat_seconds=0.01,
        observer=observer,
    )


class AdapterContractTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, executor=None, *, trace_sink=None):
        loaded = load_strategy("builtin:local-qwen")
        executor = executor or RecordingModelCallExecutor()
        config = AdapterConfig(
            schema_version=1,
            strategies=("builtin:local-qwen",),
        )
        catalog = StrategyCatalog.from_config(
            config,
            env={},
            engine_builder=lambda _loaded, observer: engine_for(executor, observer),
            trace_observer=trace_sink,
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

    async def test_streams_complete_structured_request_through_engine(self):
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
        self.assertIn(b'"input_tokens":0', response.content)
        submitted = executor.requests[0].conversation
        self.assertEqual(submitted.system[0]["cache_control"]["type"], "ephemeral")
        self.assertEqual(submitted.messages[0].content[0]["signature"], "signature")
        self.assertEqual(submitted.messages[1].content[1]["data"], "AA==")
        self.assertTrue(submitted.messages[1].extensions["future_message_field"])
        self.assertEqual(submitted.tools[0]["extensions"]["future_tool_field"], 1)
        self.assertTrue(submitted.extensions["future_request_field"]["keep"])

        run = app.state.strategy_catalog.metrics.records[0]
        self.assertEqual(run["schema"], "smart-ask.run/v2")
        self.assertEqual(run["provider_requests"][0]["actual_model"], "qwen3:14b")
        session = app.state.strategy_catalog.metrics.sessions["session-1"]
        self.assertEqual(
            session["resources"]["overall"]["known_total_tokens"],
            15,
        )

    async def test_non_streaming_response_uses_the_same_engine(self):
        client, app, _executor = await self.make_client()

        response = await client.post(
            "/v1/messages",
            headers={"x-api-key": "local-secret"},
            json={
                "model": MODEL_ID,
                "stream": False,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"][0]["text"], "adapter-ok")
        self.assertEqual(response.json()["stop_reason"], "end_turn")
        self.assertEqual(len(app.state.strategy_catalog.metrics.records), 1)

    async def test_streams_tool_use_and_partial_json(self):
        executor = RecordingModelCallExecutor([tool_events()])
        client, app, _executor = await self.make_client(executor)

        response = await client.post(
            "/v1/messages",
            headers={"x-api-key": "local-secret"},
            json={
                "model": MODEL_ID,
                "stream": True,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "read a.py"}],
                "tools": [{
                    "name": "read",
                    "input_schema": {"type": "object"},
                }],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"type":"tool_use"', response.content)
        self.assertIn(b'"type":"input_json_delta"', response.content)
        self.assertIn(b'\\"path\\":\\"a.py\\"', response.content)
        run = app.state.strategy_catalog.metrics.records[0]
        self.assertEqual(run["provider_requests"][0]["tool_call_count"], 1)

    async def test_discovery_and_token_count_use_compiled_engine(self):
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
        self.assertEqual(executor.token_requests[-1].target_id, "local-qwen")
        self.assertEqual(executor.requests, [])

    async def test_requires_authentication_and_accepts_both_credential_headers(self):
        client, _app, _executor = await self.make_client()
        body = {
            "model": MODEL_ID,
            "stream": False,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}],
        }

        missing = await client.post("/v1/messages", json=body)
        wrong = await client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer wrong"},
            json=body,
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)

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

    async def test_client_cancellation_reaches_model_call_and_is_recorded(self):
        executor = BlockingModelCallExecutor()
        client, app, _executor = await self.make_client(executor)
        request = asyncio.create_task(client.post(
            "/v1/messages",
            headers={"x-api-key": "local-secret"},
            json={
                "model": MODEL_ID,
                "stream": True,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "wait"}],
            },
        ))
        await asyncio.wait_for(executor.started.wait(), timeout=1)

        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(executor.cancelled.wait(), timeout=1)
        for _ in range(10):
            if app.state.strategy_catalog.metrics.records:
                break
            await asyncio.sleep(0)

        run = app.state.strategy_catalog.metrics.records[0]
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(run["provider_requests"][0]["status"], "cancelled")

    async def test_trace_records_conversation_once_and_canonical_run_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            trace = TraceSessionSink(str(path))
            client, _app, _executor = await self.make_client(
                trace_sink=trace,
            )

            response = await client.post(
                "/v1/messages",
                headers={"x-api-key": "local-secret"},
                json={
                    "model": MODEL_ID,
                    "stream": False,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "trace me"}],
                },
            )
            self.assertEqual(response.status_code, 200)
            trace.close()

            index = json.loads(
                (path / "session.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["schema"], "smart-ask.trace-session-index/v2")
            self.assertEqual(index["invocation_count"], 1)
            self.assertEqual(len(index["contexts"]), 1)
            self.assertEqual(len(index["inputs"]), 1)
            self.assertEqual(index["invocations"][0]["ordinal"], 1)
            self.assertEqual(index["invocations"][0]["status"], "completed")
            invocation_path = path / index["invocations"][0]["file"]
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
            self.assertEqual(
                (path / "session.json").stat().st_mode & 0o777,
                0o600,
            )
            self.assertEqual(invocation_path.stat().st_mode & 0o777, 0o600)
            rows = [
                json.loads(line)
                for line in invocation_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0], {
                "event": "trace_start",
                "schema": "smart-ask.method-invocation-trace/v2",
            })
            self.assertEqual(rows[1]["event"], "run_start")
            self.assertIn("run_id", rows[1])
            self.assertEqual(rows[1]["ordinal"], 1)
            self.assertTrue(all("schema" not in row for row in rows[1:]))
            self.assertTrue(all("run_id" not in row for row in rows[2:]))
            self.assertEqual(sum(row["event"] == "conversation" for row in rows), 1)
            self.assertEqual(sum(row["event"] == "decision" for row in rows), 1)
            self.assertEqual(sum(row["event"] == "model_call" for row in rows), 1)
            self.assertEqual(sum(row["event"] == "provider_start" for row in rows), 1)
            self.assertEqual(sum(row["event"] == "model_output" for row in rows), 1)
            self.assertEqual(
                sum(row["event"] == "model_output_chunk" for row in rows),
                0,
            )
            self.assertEqual(rows[-1]["event"], "run_end")
            self.assertEqual(rows[-1]["status"], "completed")
            conversation = next(
                row["conversation"] for row in rows if row["event"] == "conversation"
            )
            self.assertEqual(conversation["messages"][0]["content"][0]["text"], "trace me")
            model_call = next(row for row in rows if row["event"] == "model_call")
            self.assertEqual(model_call["conversation_ref"], "run_input")
            self.assertNotIn("conversation", model_call)
            output = next(row for row in rows if row["event"] == "model_output")
            self.assertEqual(output["text"], "adapter-ok")
            self.assertNotIn("extensions", conversation["messages"][0])
            self.assertNotIn("agent_id", rows[1]["metadata"])

    async def test_trace_is_updated_before_a_slow_model_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            trace = TraceSessionSink(str(path))
            executor = BlockingModelCallExecutor()
            client, _app, _executor = await self.make_client(
                executor,
                trace_sink=trace,
            )
            request = asyncio.create_task(client.post(
                "/v1/messages",
                headers={"x-api-key": "local-secret"},
                json={
                    "model": MODEL_ID,
                    "stream": True,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "slow trace"}],
                },
            ))
            await asyncio.wait_for(executor.started.wait(), timeout=1)

            index = json.loads(
                (path / "session.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["invocations"][0]["status"], "running")
            invocation_path = path / index["invocations"][0]["file"]
            rows = [
                json.loads(line)
                for line in invocation_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("run_start", [row["event"] for row in rows])
            self.assertIn("conversation", [row["event"] for row in rows])
            self.assertIn("model_call", [row["event"] for row in rows])
            self.assertNotIn("run_end", [row["event"] for row in rows])

            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(executor.cancelled.wait(), timeout=1)
            trace.close()

    async def test_concurrent_invocations_receive_separate_trace_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            trace = TraceSessionSink(str(path))
            executor = RecordingModelCallExecutor(responses=[
                text_events("first"),
                text_events("second"),
            ])
            client, _app, _executor = await self.make_client(
                executor,
                trace_sink=trace,
            )

            responses = await asyncio.gather(*(
                client.post(
                    "/v1/messages",
                    headers={"x-api-key": "local-secret"},
                    json={
                        "model": MODEL_ID,
                        "stream": False,
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "repeat"}],
                    },
                )
                for _ in range(2)
            ))
            self.assertEqual(
                [response.status_code for response in responses],
                [200, 200],
            )
            trace.close()

            index = json.loads(
                (path / "session.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["invocation_count"], 2)
            self.assertEqual(len(index["contexts"]), 1)
            self.assertEqual(len(index["inputs"]), 1)
            self.assertEqual(index["invocations"][1]["same_input_as"], 1)
            self.assertEqual(
                [value["status"] for value in index["invocations"]],
                ["completed", "completed"],
            )
            files = [path / value["file"] for value in index["invocations"]]
            self.assertEqual(len(set(files)), 2)
            for invocation_path in files:
                rows = [
                    json.loads(line)
                    for line in invocation_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(
                    rows[0]["schema"],
                    "smart-ask.method-invocation-trace/v2",
                )
                self.assertEqual(rows[-1]["event"], "run_end")

    async def test_trace_references_propagated_model_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            trace = TraceSessionSink(str(path))
            client, _app, _executor = await self.make_client(
                FailingModelCallExecutor(),
                trace_sink=trace,
            )

            response = await client.post(
                "/v1/messages",
                headers={"x-api-key": "local-secret"},
                json={
                    "model": MODEL_ID,
                    "stream": False,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "fail"}],
                },
            )
            self.assertEqual(response.status_code, 500)
            trace.close()

            index = json.loads(
                (path / "session.json").read_text(encoding="utf-8")
            )
            invocation_path = path / index["invocations"][0]["file"]
            rows = [
                json.loads(line)
                for line in invocation_path.read_text(encoding="utf-8").splitlines()
            ]
            model_error = next(row for row in rows if row["event"] == "model_error")
            run_error = next(row for row in rows if row["event"] == "run_error")
            self.assertEqual(model_error["message"], "provider exploded")
            self.assertNotIn("message", run_error)
            self.assertEqual(run_error["caused_by"], {
                "event": "model_error",
                "call_id": "call-1",
            })

    async def test_long_output_remains_incremental_in_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            trace = TraceSessionSink(str(path))
            client, _app, _executor = await self.make_client(
                RecordingModelCallExecutor(responses=[text_events("x" * 600)]),
                trace_sink=trace,
            )

            response = await client.post(
                "/v1/messages",
                headers={"x-api-key": "local-secret"},
                json={
                    "model": MODEL_ID,
                    "stream": False,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": "long"}],
                },
            )
            self.assertEqual(response.status_code, 200)
            trace.close()

            index = json.loads(
                (path / "session.json").read_text(encoding="utf-8")
            )
            invocation_path = path / index["invocations"][0]["file"]
            events = [
                json.loads(line)["event"]
                for line in invocation_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("model_output_start", events)
            self.assertIn("model_output_chunk", events)
            self.assertIn("model_output_end", events)
            self.assertNotIn("model_output", events)

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
    def test_trace_call_replaces_only_changed_conversation_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            trace = TraceSessionSink(str(path))
            conversation = Conversation.from_text(
                "hello",
                system="system",
                parameters={"max_tokens": 100, "temperature": 0.0},
            )
            metadata = RunMetadata(
                strategy_name="test",
                strategy_digest="a" * 64,
                session_id="session",
            )
            trace.run_started("b" * 32, conversation, metadata)
            trace.model_call_planned(
                "b" * 32,
                "call-1",
                1,
                ModelCallSpec(
                    profile_id="profile",
                    target_id="target",
                    role="writer",
                    phase="generation",
                    conversation=conversation.with_parameters({
                        "max_tokens": 50,
                    }),
                ),
                None,
            )
            trace.close()

            invocation = next(path.glob("*.jsonl"))
            rows = [
                json.loads(line)
                for line in invocation.read_text(encoding="utf-8").splitlines()
            ]
            call = next(row for row in rows if row["event"] == "model_call")
            self.assertEqual(call["conversation_ref"], "run_input")
            self.assertEqual(call["replace"], {
                "parameters": {"max_tokens": 50, "temperature": 0.0},
            })
            self.assertNotIn("conversation", call)

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

    def test_metrics_file_and_trace_directory_must_be_distinct(self):
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            MetricsConfig(
                jsonl_path="/tmp/combined.jsonl",
                trace_directory="/tmp/combined.jsonl",
            )

    def test_custom_strategy_cannot_escape_prompt_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "allowed"
            allowed.mkdir()
            (root / "secret.txt").write_text("secret", encoding="utf-8")
            strategy = allowed / "escape.yaml"
            strategy.write_text(
                """schema_version: 3
name: escape-v1
profiles:
  local:
    target: local-model
    system_prompt:
      type: file
      path: ../secret.txt
method:
  type: fixed
  role: writer
  profile: local
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

            with self.assertRaisesRegex(
                AdapterConfigError,
                r"outside .*allowed roots",
            ):
                StrategyCatalog.from_config(config, env={})


if __name__ == "__main__":
    unittest.main()
