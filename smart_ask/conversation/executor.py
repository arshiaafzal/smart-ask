"""Structural contracts for harness-neutral conversation execution."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .domain import ConversationEvent, ConversationExecutionRequest


@runtime_checkable
class ConversationExecutor(Protocol):
    """Stream one complete conversation through a selected physical model."""

    async def stream(
        self,
        request: ConversationExecutionRequest,
    ) -> AsyncIterator[ConversationEvent]: ...

    async def count_tokens(self, request: ConversationExecutionRequest) -> int | None: ...

