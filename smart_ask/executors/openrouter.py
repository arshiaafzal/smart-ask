"""OpenRouter's explicit Chat Completions executor."""

from __future__ import annotations

from collections.abc import Mapping

from .chat_completions import _ChatCompletionsExecutor


class OpenRouterExecutor(_ChatCompletionsExecutor):
    """Execute models through OpenRouter's Chat Completions dialect."""

    def __init__(
        self,
        client,
        system_prompts: Mapping[str, str] | None = None,
        max_tokens: Mapping[str, int] | None = None,
        temperatures: Mapping[str, float] | None = None,
        *,
        default_max_tokens: int,
        temperature: float,
    ):
        super().__init__(
            client,
            system_prompts=system_prompts,
            max_tokens=max_tokens,
            temperatures=temperatures,
            default_max_tokens=default_max_tokens,
            temperature=temperature,
            default_reasoning_effort=None,
            provider_name="OpenRouter",
            max_tokens_field="max_tokens",
            send_temperature=True,
            read_provider_cost=True,
        )
