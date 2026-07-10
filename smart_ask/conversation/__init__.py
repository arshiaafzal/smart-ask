"""Public harness-neutral conversation runtime API."""

from .domain import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationMessage,
    ConversationRequest,
    SessionContext,
    freeze_value,
    thaw_value,
)
from .executor import ConversationExecutor

__all__ = [
    "ConversationEvent",
    "ConversationExecutionRequest",
    "ConversationExecutor",
    "ConversationMessage",
    "ConversationMetricsStore",
    "ConversationRequest",
    "ConversationRuntime",
    "SessionContext",
    "freeze_value",
    "thaw_value",
]


def __getattr__(name):
    if name == "ConversationMetricsStore":
        from .metrics import ConversationMetricsStore

        return ConversationMetricsStore
    if name == "ConversationRuntime":
        from .runtime import ConversationRuntime

        return ConversationRuntime
    raise AttributeError(name)
