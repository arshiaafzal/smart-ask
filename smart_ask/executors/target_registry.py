"""Async execution of model calls through trusted deployment targets."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
import os

import httpx

from ..conversation.domain import (
    ConversationEvent,
    thaw_value,
)
from ..conversation.model import InputTokenCount, ModelCallSpec
from ..strategy.errors import StrategyBuildError
from ..strategy.targets import TargetDefinition, TargetRegistry
from ._protocol import ProviderCall
from .groq import GroqTransport
from .ollama import OllamaTransport
from .openai import OpenAITransport
from .openrouter import OpenRouterTransport


class TargetExecutorRegistry:
    """Resolve each logical call to one pooled async target executor."""

    def __init__(
        self,
        targets: TargetRegistry,
        *,
        env: Mapping[str, str] | None = None,
        http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        if not isinstance(targets, TargetRegistry):
            raise TypeError("targets must be a TargetRegistry")
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("env must be a mapping or None")
        self._targets = targets
        self._env = dict(os.environ if env is None else env)
        self._http_client_factory = http_client_factory or httpx.AsyncClient
        self._executors: dict[str, object] = {}
        self._clients: dict[tuple[str, str, str], httpx.AsyncClient] = {}

    @property
    def targets(self) -> TargetRegistry:
        return self._targets

    async def stream(self, spec: ModelCallSpec) -> AsyncIterator[ConversationEvent]:
        if not isinstance(spec, ModelCallSpec):
            raise TypeError("spec must be a ModelCallSpec")
        target = self._targets.resolve(spec.target_id)
        self._validate_call(target, spec)
        executor = self._executor(target)
        request = ProviderCall(
            model=target.model,
            role=spec.role,
            conversation=spec.conversation,
        )
        async for event in executor.stream(request):
            if event.kind == "message_start":
                data = thaw_value(event.data)
                data["selected_model"] = target.model
                data["target_id"] = target.target_id
                yield ConversationEvent("message_start", data)
            else:
                yield event

    async def count_tokens(self, spec: ModelCallSpec) -> InputTokenCount | None:
        target = self._targets.resolve(spec.target_id)
        self._validate_call(target, spec)
        executor = self._executor(target)
        request = ProviderCall(
            model=target.model,
            role=spec.role,
            conversation=spec.conversation,
        )
        return await executor.count_tokens(request)

    def selected_model(self, target_id: str) -> str:
        return self._targets.resolve(target_id).model

    def _executor(self, target: TargetDefinition):
        existing = self._executors.get(target.target_id)
        if existing is not None:
            return existing
        if target.transport == "ollama":
            executor = OllamaTransport(
                base_url=target.base_url or "http://127.0.0.1:11434/api",
                default_max_tokens=min(8192, target.limits.max_output_tokens),
                temperature=0.0,
                think=target.think,
                timeout_seconds=target.limits.timeout_seconds,
            )
        else:
            client = self._client(target)
            if target.transport == "openrouter":
                executor = OpenRouterTransport(
                    client,
                    default_max_tokens=min(1024, target.limits.max_output_tokens),
                    temperature=0.0,
                )
            elif target.transport == "openai":
                executor = OpenAITransport(
                    client,
                    default_max_tokens=min(8192, target.limits.max_output_tokens),
                    reasoning_effort="medium",
                )
            elif target.transport == "groq":
                executor = GroqTransport(
                    client,
                    default_max_tokens=min(8192, target.limits.max_output_tokens),
                    reasoning_effort="medium",
                )
            else:
                raise StrategyBuildError(
                    f"unsupported target transport: {target.transport}"
                )
        self._executors[target.target_id] = executor
        return executor

    def _client(self, target: TargetDefinition) -> httpx.AsyncClient:
        if target.credential_env is None:
            raise StrategyBuildError(
                f"target {target.target_id!r} has no credential provider"
            )
        api_key = self._env.get(target.credential_env, "")
        if not api_key.strip():
            raise StrategyBuildError(
                f"required environment variable {target.credential_env} is not set"
            )
        base_url = target.base_url or ""
        key = (target.transport, base_url, api_key)
        client = self._clients.get(key)
        if client is None:
            client = self._http_client_factory(
                base_url=base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(target.limits.timeout_seconds),
                follow_redirects=False,
            )
            if not isinstance(client, httpx.AsyncClient):
                raise TypeError("http_client_factory must return httpx.AsyncClient")
            self._clients[key] = client
        return client

    @staticmethod
    def _validate_call(target: TargetDefinition, spec: ModelCallSpec) -> None:
        parameters = spec.conversation.parameters
        max_tokens = parameters.get("max_tokens")
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
            if max_tokens > target.limits.max_output_tokens:
                raise ValueError(
                    f"profile requests {max_tokens} output tokens but target "
                    f"{target.target_id!r} allows at most "
                    f"{target.limits.max_output_tokens}"
                )
        if spec.conversation.tools and "tools" not in target.capabilities:
            raise ValueError(f"target {target.target_id!r} does not support tools")

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
        self._executors.clear()
