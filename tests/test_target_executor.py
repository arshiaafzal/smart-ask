import json
import unittest

import httpx

from smart_ask import Conversation
from smart_ask.conversation.model import ModelCallSpec
from smart_ask.executors import TargetExecutorRegistry
from smart_ask.strategy.targets import TargetDefinition, TargetRegistry


class TargetExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_call_by_trusted_target_and_records_selected_model(self):
        observed = {}

        async def handler(request):
            observed["url"] = str(request.url)
            observed["body"] = json.loads(request.content)
            return httpx.Response(200, content=(
                b'data: {"model":"provider-actual","choices":[{"delta":'
                b'{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":2,'
                b'"completion_tokens":1,"total_tokens":3}}\n\n'
                b'data: [DONE]\n\n'
            ))

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://router.test/api/v1",
        )
        registry = TargetRegistry((TargetDefinition(
            "approved-model",
            "openrouter",
            "vendor/model",
            base_url="https://router.test/api/v1",
            credential_env="TEST_KEY",
            capabilities=frozenset({"text", "streaming"}),
        ),))
        executor = TargetExecutorRegistry(
            registry,
            env={"TEST_KEY": "secret"},
            http_client_factory=lambda **_kwargs: client,
        )
        self.addAsyncCleanup(executor.aclose)
        events = [event async for event in executor.stream(ModelCallSpec(
            profile_id="writer",
            target_id="approved-model",
            role="writer",
            conversation=Conversation.from_text("hello"),
        ))]

        self.assertEqual(observed["body"]["model"], "vendor/model")
        start = events[0]
        self.assertEqual(start.data["selected_model"], "vendor/model")
        self.assertEqual(start.data["model"], "provider-actual")

    async def test_strategy_cannot_override_target_output_ceiling(self):
        registry = TargetRegistry((TargetDefinition(
            "limited-model",
            "ollama",
            "local/model",
            base_url="http://127.0.0.1:11434/api",
        ),))
        executor = TargetExecutorRegistry(registry, env={})
        spec = ModelCallSpec(
            profile_id="writer",
            target_id="limited-model",
            role="writer",
            conversation=Conversation.from_text(
                "hello",
                parameters={"max_tokens": 999999},
            ),
        )
        with self.assertRaisesRegex(ValueError, "allows at most"):
            _events = [event async for event in executor.stream(spec)]

if __name__ == "__main__":
    unittest.main()
