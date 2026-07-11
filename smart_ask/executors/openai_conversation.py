"""First-party OpenAI Responses API conversation executor."""

from .responses_conversation import ResponsesConversationExecutor


class OpenAIConversationExecutor(ResponsesConversationExecutor):
    """Stream normalized conversations through OpenAI's Responses API."""

    _include_store = True
