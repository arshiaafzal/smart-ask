"""First-party OpenAI Responses API executor."""

from .responses import ResponsesExecutor


class OpenAIExecutor(ResponsesExecutor):
    """Execute one-shot model calls through OpenAI's Responses API."""

    _include_store = True
