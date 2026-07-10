import asyncio
import json
import unittest

import httpx

from smart_ask.conversation import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationMetricsStore,
    ConversationMessage,
    ConversationRequest,
    ConversationRuntime,
    SessionContext,
)
from smart_ask.executors import (
    OllamaConversationExecutor,
    OpenAIConversationExecutor,
    OpenRouterConversationExecutor,
)
from smart_ask.strategy import StrategyBuilder, load_strategy
from tests.helpers import FakeClient, response


def request(text="write code", *, tools=()):
    return ConversationRequest(
        system=({"type": "text", "text": "caller system"},),
        messages=(ConversationMessage(
            "user",
            ({"type": "text", "text": text},),
        ),),
        tools=tools,
        parameters={"max_tokens": 100},
        extensions={"future_request_field": {"keep": True}},
    )


def text_events(text, model, *, input_tokens=10, output_tokens=3):
    return (
        ConversationEvent("message_start", {"model": model}),
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
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
        }),
        ConversationEvent("message_delta", {"stop_reason": "stop"}),
        ConversationEvent("message_stop"),
    )


class FakeConversationExecutor:
    def __init__(self, event_sequences):
        self.event_sequences = list(event_sequences)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        for event in self.event_sequences.pop(0):
            yield event

    async def count_tokens(self, request):
        self.requests.append(request)
        return 123


class ConversationDomainTests(unittest.TestCase):
    def test_retains_structured_open_schema_and_projects_only_human_text(self):
        value = ConversationRequest(
            system=({"type": "text", "text": "system", "future": 1},),
            messages=(
                ConversationMessage("user", (
                    {"type": "text", "text": "do it"},
                    {"type": "image", "data": "AA==", "future": True},
                )),
                ConversationMessage("assistant", (
                    {"type": "tool_call", "id": "x", "name": "read"},
                )),
                ConversationMessage("user", (
                    {"type": "tool_result", "id": "x", "content": "ok"},
                )),
            ),
            tools=({"name": "read", "input_schema": {"type": "object"}},),
            extensions={"unknown": [1, 2]},
        )

        text, fingerprint = value.latest_human_instruction()

        self.assertEqual(text, "do it")
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(value.system[0]["future"], 1)
        self.assertEqual(value.messages[0].content[1]["data"], "AA==")
        self.assertEqual(value.extensions["unknown"], (1, 2))


class ConversationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_local_qwen_runtime_owns_execution_and_metrics(self):
        loaded = load_strategy("builtin:local-qwen")
        router = StrategyBuilder(env={}).build_router(loaded)
        executor = FakeConversationExecutor([
            text_events("answer", "qwen3:14b", input_tokens=20, output_tokens=4),
        ])
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=router,
            executor=executor,
        )

        events = [event async for event in runtime.stream(
            request(),
            SessionContext(session_id="session-1"),
        )]

        self.assertEqual(events[-1].kind, "message_stop")
        self.assertEqual(executor.requests[0].model, "qwen3:14b")
        self.assertEqual(
            executor.requests[0].conversation.extensions["future_request_field"][
                "keep"
            ],
            True,
        )
        run = runtime.metrics.records[0]
        self.assertEqual(run["attempts"][0]["actual_model"], "qwen3:14b")
        self.assertEqual(run["attempts"][0]["total_tokens"], 24)
        session = runtime.metrics.sessions["session-1"]
        self.assertEqual(session["known_total_tokens"], 24)
        self.assertEqual(session["by_model"]["qwen3:14b"]["attempts"], 1)

    async def test_cascade_is_owned_by_runtime_not_transport_adapter(self):
        loaded = load_strategy("builtin:python-code-generation-cascade")
        classifier = FakeClient([response('{"d":"easy"}')])
        router = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "classifier-key"},
            openrouter_client_factory=lambda _url, _key: classifier,
        ).build_router(loaded)
        executor = FakeConversationExecutor([
            text_events("draft\nESCALATE_NOW\n", "cheap-model"),
            text_events("correct", "hard-model", input_tokens=12, output_tokens=2),
        ])
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=router,
            executor=executor,
        )

        events = [event async for event in runtime.stream(request())]
        visible = "".join(
            event.data["delta"]["text"]
            for event in events
            if event.kind == "content_delta"
            and event.data["delta"].get("type") == "text"
        )

        self.assertEqual(visible, "correct")
        self.assertEqual([item.model for item in executor.requests], [
            "google/gemini-2.5-flash-lite",
            "anthropic/claude-opus-4.8",
        ])
        run = runtime.metrics.records[0]
        self.assertEqual(run["route_path"], ["initial-easy", "escalation"])
        self.assertEqual(len(run["attempts"]), 2)

    async def test_buffered_cascade_emits_transport_neutral_heartbeats(self):
        loaded = load_strategy("builtin:python-code-generation-cascade")
        classifier = FakeClient([response('{"d":"easy"}')])
        router = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "classifier-key"},
            openrouter_client_factory=lambda _url, _key: classifier,
        ).build_router(loaded)

        class SlowExecutor(FakeConversationExecutor):
            async def stream(self, execution):
                self.requests.append(execution)
                events = self.event_sequences.pop(0)
                await asyncio.sleep(0.02)
                for event in events:
                    yield event

        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=router,
            executor=SlowExecutor([text_events("accepted", "cheap-model")]),
            heartbeat_seconds=0.005,
        )

        events = [event async for event in runtime.stream(request())]

        self.assertTrue(any(event.kind == "heartbeat" for event in events))
        self.assertEqual(events[-1].kind, "message_stop")

    async def test_count_tokens_delegates_to_strategy_executor(self):
        loaded = load_strategy("builtin:local-qwen")
        router = StrategyBuilder(env={}).build_router(loaded)
        executor = FakeConversationExecutor([])
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=router,
            executor=executor,
        )

        count = await runtime.count_tokens(request())

        self.assertEqual(count, 123)
        self.assertEqual(executor.requests[0].model, "qwen3:14b")

    async def test_metrics_sink_receives_prompt_free_envelope_nonfatally(self):
        received = []
        loaded = load_strategy("builtin:local-qwen")
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=StrategyBuilder(env={}).build_router(loaded),
            executor=FakeConversationExecutor([
                text_events("secret output", "qwen3:14b"),
            ]),
            metrics=ConversationMetricsStore(sink=received.append),
        )

        _events = [event async for event in runtime.stream(request("secret prompt"))]

        self.assertEqual(len(received), 1)
        serialized = json.dumps(received[0])
        self.assertNotIn("secret prompt", serialized)
        self.assertNotIn("secret output", serialized)

        def broken_sink(_value):
            raise OSError("disk unavailable")

        failing = ConversationMetricsStore(sink=broken_sink)
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=StrategyBuilder(env={}).build_router(loaded),
            executor=FakeConversationExecutor([text_events("ok", "qwen3:14b")]),
            metrics=failing,
        )
        _events = [event async for event in runtime.stream(request())]
        self.assertEqual(failing.sink_errors, ("OSError: disk unavailable",))

    async def test_cancellation_is_recorded_and_propagated(self):
        started = asyncio.Event()

        class BlockingExecutor:
            async def stream(self, _execution):
                started.set()
                await asyncio.Event().wait()
                if False:
                    yield ConversationEvent("message_stop")

            async def count_tokens(self, _execution):
                return 1

        loaded = load_strategy("builtin:local-qwen")
        runtime = ConversationRuntime(
            loaded_strategy=loaded,
            router=StrategyBuilder(env={}).build_router(loaded),
            executor=BlockingExecutor(),
        )

        async def consume():
            return [event async for event in runtime.stream(request())]

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        run = runtime.metrics.records[0]
        self.assertTrue(run["cancelled"])
        self.assertTrue(run["attempts"][0]["cancelled"])


class OllamaConversationExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_native_qwen_text_tools_and_usage(self):
        observed = {}
        chunks = [
            {
                "model": "qwen3:14b",
                "message": {"role": "assistant", "content": "hello"},
                "done": False,
            },
            {
                "model": "qwen3:14b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {"name": "read", "arguments": {"path": "x"}}
                    }],
                },
                "done": False,
            },
            {
                "model": "qwen3:14b",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 8,
                "eval_count": 3,
            },
        ]

        async def handler(http_request):
            observed["body"] = json.loads(http_request.content)
            content = b"".join(
                json.dumps(chunk).encode() + b"\n" for chunk in chunks
            )
            return httpx.Response(200, content=content)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        executor = OllamaConversationExecutor(
            base_url="http://ollama.test/api",
            default_max_tokens=100,
            temperature=0,
            client=client,
        )
        conversation = request(
            tools=({
                "name": "read",
                "description": "read a file",
                "input_schema": {"type": "object"},
            },)
        )

        events = [event async for event in executor.stream(
            ConversationExecutionRequest("qwen3:14b", "writer", conversation)
        )]

        self.assertEqual(observed["body"]["model"], "qwen3:14b")
        self.assertEqual(observed["body"]["tools"][0]["function"]["name"], "read")
        self.assertTrue(any(
            event.kind == "content_start"
            and event.data["block"].get("type") == "tool_call"
            for event in events
        ))
        usage = next(event for event in events if event.kind == "usage")
        self.assertEqual(usage.data["total_tokens"], 11)

    async def test_translates_structured_thinking_request(self):
        observed = {}

        async def handler(http_request):
            observed["body"] = json.loads(http_request.content)
            chunk = {
                "model": "qwen3:14b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 1,
            }
            return httpx.Response(200, content=json.dumps(chunk).encode() + b"\n")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        executor = OllamaConversationExecutor(
            base_url="http://ollama.test/api",
            default_max_tokens=100,
            temperature=0,
            client=client,
        )
        conversation = request().with_parameters({
            "thinking": {"type": "adaptive"},
        })

        _events = [event async for event in executor.stream(
            ConversationExecutionRequest("qwen3:14b", "writer", conversation)
        )]

        self.assertIs(observed["body"]["think"], True)


class OpenRouterConversationExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_text_tool_arguments_and_usage(self):
        observed = {}
        chunks = (
            {
                "model": "provider/model-actual",
                "choices": [{
                    "delta": {"content": "hello"},
                    "finish_reason": None,
                }],
            },
            {
                "model": "provider/model-actual",
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "read", "arguments": "{\"path\":"},
                    }]},
                    "finish_reason": None,
                }],
            },
            {
                "model": "provider/model-actual",
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "\"x\"}"},
                    }]},
                    "finish_reason": "tool_calls",
                }],
            },
            {
                "model": "provider/model-actual",
                "choices": [],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                    "cost": 0.001,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
        )

        async def handler(http_request):
            observed["body"] = json.loads(http_request.content)
            data = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n"
                for chunk in chunks
            ) + b"data: [DONE]\n\n"
            return httpx.Response(
                200,
                content=data,
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://router.test/api/v1",
        )
        self.addAsyncCleanup(client.aclose)
        executor = OpenRouterConversationExecutor(
            client,
            default_max_tokens=100,
            temperature=0,
        )
        conversation = request(
            tools=({
                "name": "read",
                "description": "read a file",
                "input_schema": {"type": "object"},
            },)
        ).with_parameters({"thinking": {"type": "enabled", "budget_tokens": 32}})

        events = [event async for event in executor.stream(
            ConversationExecutionRequest("provider/model", "writer", conversation)
        )]

        self.assertEqual(observed["body"]["model"], "provider/model")
        self.assertEqual(observed["body"]["reasoning"]["max_tokens"], 32)
        fragments = [
            event.data["delta"]["json"]
            for event in events
            if event.kind == "content_delta"
            and event.data["delta"].get("type") == "tool_arguments_json"
        ]
        self.assertEqual("".join(fragments), '{"path":"x"}')
        usage = next(event for event in events if event.kind == "usage")
        self.assertEqual(usage.data["total_tokens"], 13)
        self.assertEqual(usage.data["reasoning_tokens"], 2)

    async def test_strategy_builder_selects_openrouter_without_adapter_policy(self):
        observed = {}

        async def handler(http_request):
            observed["body"] = json.loads(http_request.content)
            data = (
                b'data: {"model":"actual","choices":[{"delta":'
                b'{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":2,'
                b'"completion_tokens":1,"total_tokens":3}}\n\n'
                b'data: [DONE]\n\n'
            )
            return httpx.Response(200, content=data)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://router.test/api/v1",
        )
        self.addAsyncCleanup(client.aclose)
        builder = StrategyBuilder(
            env={"OPENROUTER_API_KEY": "test-key"},
            openrouter_conversation_client_factory=(
                lambda _url, _key: client
            ),
        )
        runtime = builder.build_conversation_runtime(
            load_strategy("builtin:python-code-generation-fixed-opus")
        )

        events = [event async for event in runtime.stream(request())]

        self.assertEqual(events[-1].kind, "message_stop")
        self.assertEqual(
            observed["body"]["model"],
            "anthropic/claude-opus-4.8",
        )
        await runtime.aclose()
        self.assertTrue(client.is_closed)


class OpenAIConversationExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_responses_streaming_tools_usage_and_reasoning(self):
        observed = {}

        async def handler(http_request):
            observed["path"] = http_request.url.path
            observed["body"] = json.loads(http_request.content)
            chunks = (
                {
                    "type": "response.created",
                    "response": {"model": "gpt-5.3-codex"},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "item_1",
                        "call_id": "call_1",
                        "name": "read",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "item_id": "item_1",
                    "delta": '{"path":"x"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "model": "gpt-5.3-codex",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 9,
                            "output_tokens": 4,
                            "total_tokens": 13,
                            "input_tokens_details": {
                                "cached_tokens": 1,
                                "cache_write_tokens": 0,
                            },
                            "output_tokens_details": {"reasoning_tokens": 2},
                        },
                    },
                },
            )
            data = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n"
                for chunk in chunks
            ) + b"data: [DONE]\n\n"
            return httpx.Response(200, content=data)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.openai.test/v1",
        )
        self.addAsyncCleanup(client.aclose)
        executor = OpenAIConversationExecutor(
            client,
            default_max_tokens=8192,
            reasoning_effort="medium",
        )
        conversation = request(
            tools=({
                "name": "read",
                "description": "read a file",
                "input_schema": {"type": "object"},
            },)
        ).with_parameters({
            "reasoning_effort": "high",
            "tool_choice": {"type": "tool", "name": "read"},
        })

        events = [event async for event in executor.stream(
            ConversationExecutionRequest("gpt-5.3-codex", "writer", conversation)
        )]

        body = observed["body"]
        self.assertEqual(observed["path"], "/v1/responses")
        self.assertEqual(body["max_output_tokens"], 100)
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertIs(body["store"], False)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["tool_choice"], {
            "type": "function",
            "name": "read",
        })
        self.assertEqual(body["tools"][0]["name"], "read")
        self.assertNotIn("function", body["tools"][0])
        tool_start = next(
            event for event in events
            if event.kind == "content_start"
            and event.data["block"].get("type") == "tool_call"
        )
        self.assertEqual(tool_start.data["block"]["id"], "call_1")
        arguments = next(
            event for event in events
            if event.kind == "content_delta"
        )
        self.assertEqual(arguments.data["delta"]["json"], '{"path":"x"}')
        usage_event = next(event for event in events if event.kind == "usage")
        self.assertEqual(usage_event.data["reasoning_tokens"], 2)
        stop = next(event for event in events if event.kind == "message_delta")
        self.assertEqual(stop.data["stop_reason"], "tool_call")

    async def test_includes_openai_error_body_in_failure(self):
        async def handler(_http_request):
            return httpx.Response(429, json={
                "error": {
                    "message": "You exceeded your current quota.",
                    "code": "insufficient_quota",
                },
            })

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.openai.test/v1",
        )
        self.addAsyncCleanup(client.aclose)
        executor = OpenAIConversationExecutor(
            client,
            default_max_tokens=100,
            reasoning_effort="low",
        )

        with self.assertRaisesRegex(RuntimeError, "current quota"):
            _events = [event async for event in executor.stream(
                ConversationExecutionRequest(
                    "gpt-5.1-codex-mini",
                    "writer",
                    request(),
                )
            )]


if __name__ == "__main__":
    unittest.main()
