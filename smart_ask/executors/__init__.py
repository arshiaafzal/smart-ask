"""Public model executor contracts and implementations."""

from .base import ModelExecutor
from .hermes import HermesExecutor
from .ollama import (
    OllamaConversationExecutor,
    OllamaExecutor,
    UnsupportedConversationFeature,
)
from .openai import OpenAIExecutor
from .openai_conversation import OpenAIConversationExecutor
from .openrouter import OpenRouterExecutor
from .openrouter_conversation import OpenRouterConversationExecutor

__all__ = [
    "HermesExecutor",
    "ModelExecutor",
    "OllamaConversationExecutor",
    "OllamaExecutor",
    "OpenAIConversationExecutor",
    "OpenAIExecutor",
    "OpenRouterExecutor",
    "OpenRouterConversationExecutor",
    "UnsupportedConversationFeature",
]
