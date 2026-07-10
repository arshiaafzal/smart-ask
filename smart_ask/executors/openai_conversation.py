"""First-party OpenAI structured conversation executor."""

import httpx

from .chat_completions_conversation import _ChatCompletionsConversationExecutor


class OpenAIConversationExecutor(_ChatCompletionsConversationExecutor):
    """Stream normalized conversations through the first-party OpenAI API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        default_max_tokens: int,
        reasoning_effort: str,
    ):
        super().__init__(
            client,
            default_max_tokens=default_max_tokens,
            temperature=None,
            default_reasoning_effort=reasoning_effort,
            max_tokens_field="max_completion_tokens",
            openrouter_reasoning=False,
        )
