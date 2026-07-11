"""Groq Responses API conversation executor."""

from .responses_conversation import ResponsesConversationExecutor


class GroqConversationExecutor(ResponsesConversationExecutor):
    """Stream normalized conversations through Groq's Responses API."""
