"""OpenRouter structured conversation executor."""

import httpx

from .chat_completions_conversation import _ChatCompletionsConversationExecutor


class OpenRouterConversationExecutor(_ChatCompletionsConversationExecutor):
    """Stream normalized conversations through OpenRouter."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        default_max_tokens: int,
        temperature: float,
    ):
        super().__init__(
            client,
            default_max_tokens=default_max_tokens,
            temperature=temperature,
            default_reasoning_effort=None,
            max_tokens_field="max_tokens",
            openrouter_reasoning=True,
        )
