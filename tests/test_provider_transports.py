import unittest

import httpx

from smart_ask import Conversation
from smart_ask.executors._protocol import ProviderCall
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


if __name__ == "__main__":
    unittest.main()
