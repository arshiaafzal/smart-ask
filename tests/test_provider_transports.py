import unittest
from unittest import mock
import json

import httpx

from smart_ask import Conversation
from smart_ask.conversation import ConversationMessage
from smart_ask.executors._protocol import ProviderCall
from smart_ask.executors.anthropic import AnthropicTransport
from smart_ask.executors.openai import OpenAITransport
from smart_ask.executors.openrouter import OpenRouterTransport


class ProviderTerminalEvidenceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def request():
        return ProviderCall("provider/model", "writer", Conversation.from_text("hi"))

    async def test_chat_completion_disconnect_is_not_success(self):
        async def handler(_request):
            return httpx.Response(200, content=(
                b'data: {"model":"provider/model","choices":['
                b'{"delta":{"content":"partial"}}]}\n\n'
            ))

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://provider.test/v1",
        )
        self.addAsyncCleanup(client.aclose)
        transport = OpenRouterTransport(
            client,
            default_max_tokens=100,
            temperature=0.0,
        )

        with self.assertRaisesRegex(RuntimeError, "terminal evidence"):
            _events = [event async for event in transport.stream(self.request())]

    async def test_chat_completion_error_envelope_is_not_output(self):
        async def handler(_request):
            return httpx.Response(200, content=(
                b'data: {"error":{"message":"rate limited"}}\n\n'
            ))

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://provider.test/v1",
        )
        self.addAsyncCleanup(client.aclose)
        transport = OpenRouterTransport(
            client,
            default_max_tokens=100,
            temperature=0.0,
        )

        with self.assertRaisesRegex(RuntimeError, "rate limited"):
            _events = [event async for event in transport.stream(self.request())]

    async def test_responses_disconnect_is_not_success(self):
        async def handler(_request):
            return httpx.Response(200, content=(
                b'data: {"type":"response.created","response":'
                b'{"model":"provider/model"}}\n\n'
                b'data: {"type":"response.output_text.delta",'
                b'"output_index":0,"delta":"partial"}\n\n'
            ))

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://provider.test/v1",
        )
        self.addAsyncCleanup(client.aclose)
        transport = OpenAITransport(
            client,
            default_max_tokens=100,
            reasoning_effort="low",
        )

        with self.assertRaisesRegex(RuntimeError, "terminal evidence"):
            _events = [event async for event in transport.stream(self.request())]

    async def test_anthropic_preserves_native_blocks_usage_and_cost(self):
        observed = {}
        chunks = [
            {"type": "message_start", "message": {
                "model": "claude-sonnet-4-6",
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                    "output_tokens": 0,
                },
            }},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "plan"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "signature_delta", "signature": "signed"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "tool-1",
                               "name": "read", "input": {}}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta",
                       "partial_json": '{"path":"a.py"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
             "usage": {"output_tokens": 7}},
            {"type": "message_stop"},
        ]
        content = b"".join(
            b"event: message\n" + b"data: "
            + json.dumps(chunk).encode() + b"\n\n"
            for chunk in chunks
        )

        async def handler(request):
            observed["body"] = json.loads(request.content)
            return httpx.Response(200, content=content)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://anthropic.test",
        )
        self.addAsyncCleanup(client.aclose)
        transport = AnthropicTransport(client, default_max_tokens=100)
        conversation = Conversation(
            system=({
                "type": "text",
                "text": "system",
                "cache_control": {"type": "ephemeral"},
            },),
            messages=(
                ConversationMessage("assistant", ({
                    "type": "thinking",
                    "thinking": "prior",
                    "signature": "prior-signature",
                },)),
                ConversationMessage("user", ({"type": "text", "text": "go"},)),
                ConversationMessage("system", ({
                    "type": "text", "text": "late harness context",
                },)),
            ),
            tools=({
                "name": "read",
                "description": "Read a file",
                "input_schema": {"type": "object"},
            },),
        )

        events = [event async for event in transport.stream(
            ProviderCall("claude-sonnet-4-6", "writer", conversation)
        )]

        self.assertEqual(observed["body"]["system"][0]["cache_control"]["type"],
                         "ephemeral")
        self.assertEqual(
            observed["body"]["system"][-1]["text"],
            "late harness context",
        )
        self.assertEqual([m["role"] for m in observed["body"]["messages"]],
                         ["assistant", "user"])
        self.assertEqual(
            observed["body"]["messages"][0]["content"][0]["signature"],
            "prior-signature",
        )
        deltas = [event.data["delta"] for event in events
                  if event.kind == "content_delta"]
        self.assertIn({"type": "signature", "signature": "signed"}, deltas)
        usage = [event.data for event in events if event.kind == "usage"][-1]
        self.assertEqual(usage["input_tokens"], 35)
        self.assertEqual(usage["output_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 42)
        self.assertAlmostEqual(usage["provider_cost_usd"], 0.00015975)

    async def test_anthropic_count_tokens_uses_native_endpoint(self):
        observed = {}

        async def handler(request):
            observed["path"] = request.url.path
            observed["body"] = json.loads(request.content)
            return httpx.Response(200, json={"input_tokens": 42})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://anthropic.test",
        )
        self.addAsyncCleanup(client.aclose)
        count = await AnthropicTransport(
            client, default_max_tokens=100
        ).count_tokens(self.request())

        self.assertEqual(observed["path"], "/v1/messages/count_tokens")
        self.assertNotIn("stream", observed["body"])
        self.assertNotIn("max_tokens", observed["body"])
        self.assertEqual(count.value, 42)
        self.assertEqual(count.provenance, "exact")

    async def test_anthropic_marks_previous_user_boundary_for_cache_reuse(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json={"input_tokens": 1})
            ),
            base_url="https://anthropic.test",
        )
        self.addAsyncCleanup(client.aclose)
        conversation = Conversation(
            system=(
                {
                    "type": "text",
                    "text": "static system",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "harness system",
                    "cache_control": {"type": "ephemeral"},
                },
            ),
            messages=(
                ConversationMessage("user", ({"type": "text", "text": "task"},)),
                ConversationMessage("assistant", ({
                    "type": "tool_call",
                    "id": "read-1",
                    "name": "Read",
                    "arguments": {"file_path": "a.py"},
                },)),
                ConversationMessage("user", ({
                    "type": "tool_result",
                    "id": "read-1",
                    "content": "source",
                },)),
                ConversationMessage("assistant", ({
                    "type": "tool_call",
                    "id": "test-1",
                    "name": "Bash",
                    "arguments": {"command": "pytest"},
                },)),
                ConversationMessage("user", ({
                    "type": "tool_result",
                    "id": "test-1",
                    "content": "5 passed",
                    "cache_control": {"type": "ephemeral"},
                },)),
            ),
        )

        payload = AnthropicTransport(
            client,
            default_max_tokens=100,
        )._payload(
            ProviderCall("claude-opus-4-6", "writer", conversation),
            stream=True,
        )

        user_blocks = [
            message["content"][-1]
            for message in payload["messages"]
            if message["role"] == "user"
        ]
        self.assertNotIn("cache_control", user_blocks[0])
        self.assertEqual(
            user_blocks[1]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(
            user_blocks[2]["cache_control"],
            {"type": "ephemeral"},
        )
        breakpoints = sum(
            "cache_control" in block
            for block in payload["system"] + [
                block
                for message in payload["messages"]
                for block in message["content"]
            ]
        )
        self.assertEqual(breakpoints, 4)

    async def test_anthropic_retries_failure_before_stream_output(self):
        attempts = 0
        completed = (
            b'data: {"type":"message_start","message":{"model":'
            b'"claude-sonnet-4-6","usage":{"input_tokens":1}}}\n\n'
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"ok"}}\n\n'
            b'data: {"type":"content_block_stop","index":0}\n\n'
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":1}}\n\n'
            b'data: {"type":"message_stop"}\n\n'
        )

        async def handler(_request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, json={
                    "error": {"message": "temporarily rate limited"},
                })
            return httpx.Response(200, content=completed)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://anthropic.test",
        )
        self.addAsyncCleanup(client.aclose)
        transport = AnthropicTransport(client, default_max_tokens=100)

        with mock.patch("smart_ask.executors.anthropic.asyncio.sleep") as sleep:
            events = [event async for event in transport.stream(self.request())]

        self.assertEqual(attempts, 2)
        sleep.assert_awaited_once_with(1)
        self.assertEqual(events[-1].kind, "message_stop")

    async def test_anthropic_does_not_retry_non_transient_failure(self):
        attempts = 0

        async def handler(_request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, json={
                "error": {"message": "invalid request"},
            })

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://anthropic.test",
        )
        self.addAsyncCleanup(client.aclose)
        transport = AnthropicTransport(client, default_max_tokens=100)

        with self.assertRaisesRegex(RuntimeError, "400: invalid request"):
            _events = [event async for event in transport.stream(self.request())]

        self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()
