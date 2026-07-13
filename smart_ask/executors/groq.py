"""Groq Responses API transport."""

from .responses import ResponsesTransport


class GroqTransport(ResponsesTransport):
    """Stream normalized conversations through Groq's Responses API."""
