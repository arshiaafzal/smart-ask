"""Contract implemented by every routing method."""

from typing import Protocol

from ..domain import Context, RouteResult, Task


class RoutingMethod(Protocol):
    """An end-to-end method that chooses the next action for a task."""

    requires_response_text: bool

    def route(self, task: Task, context: Context = Context()) -> RouteResult:
        """Choose whether to execute a model or accept the latest response."""

        ...
