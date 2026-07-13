"""First-party OpenAI Responses API transport."""

from .responses import ResponsesTransport


class OpenAITransport(ResponsesTransport):
    """Stream normalized conversations through OpenAI's Responses API."""

    _include_store = True
