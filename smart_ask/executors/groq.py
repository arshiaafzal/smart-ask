"""Groq Responses API executor."""

from .responses import ResponsesExecutor


class GroqExecutor(ResponsesExecutor):
    """Execute one-shot model calls through Groq's Responses API."""
