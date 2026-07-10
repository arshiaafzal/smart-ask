"""First-party OpenAI Chat Completions executor."""

from __future__ import annotations

from collections.abc import Mapping

from .chat_completions import _ChatCompletionsExecutor


class OpenAIExecutor(_ChatCompletionsExecutor):
    """Execute OpenAI models with native token and reasoning controls."""

    def __init__(
        self,
        client,
        system_prompts: Mapping[str, str] | None = None,
        max_tokens: Mapping[str, int] | None = None,
        reasoning_efforts: Mapping[str, str] | None = None,
        *,
        default_max_tokens: int,
        reasoning_effort: str,
    ):
        super().__init__(
            client,
            system_prompts=system_prompts,
            max_tokens=max_tokens,
            reasoning_efforts=reasoning_efforts,
            default_max_tokens=default_max_tokens,
            temperature=None,
            default_reasoning_effort=reasoning_effort,
            provider_name="OpenAI",
            max_tokens_field="max_completion_tokens",
            send_temperature=False,
            read_provider_cost=False,
        )
