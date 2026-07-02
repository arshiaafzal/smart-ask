"""Contract implemented by every model execution backend."""

from typing import Protocol

from ..domain import ExecutionRequest, ModelResult


class ModelExecutor(Protocol):
    """A backend-neutral interface for invoking a selected model."""

    captures_output: bool

    def execute(self, request: ExecutionRequest) -> ModelResult:
        """Run one provider-neutral request and return its response."""

        ...
