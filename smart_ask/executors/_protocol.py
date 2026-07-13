"""Private values shared by trusted provider transports."""

from __future__ import annotations

from dataclasses import dataclass

from ..conversation.model import Conversation


@dataclass(frozen=True)
class ProviderCall:
    """One resolved provider call after strategy policy has finished."""

    model: str
    role: str
    conversation: Conversation

    def __post_init__(self) -> None:
        for name in ("model", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty trimmed text")
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
