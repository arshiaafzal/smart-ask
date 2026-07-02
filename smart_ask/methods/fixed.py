"""Fixed-model routing method used by overrides and baselines."""

from typing import Literal

from ..domain import Context, RouteResult, RoutingEvent, Task


class FixedRoutingMethod:
    """Always select one configured model without classification."""

    requires_response_text = False

    def __init__(
        self,
        model: str,
        decision: Literal["easy", "hard"],
        role: str = "writer",
    ):
        if decision not in ("easy", "hard"):
            raise ValueError("FixedRoutingMethod decision must be easy or hard")
        self.model = model
        self.decision = decision
        self.role = role

    def route(self, task: Task, context: Context = Context()) -> RouteResult:
        """Execute the fixed model once, then accept its response."""

        if context.attempts:
            return RouteResult(action="accept")
        event = RoutingEvent(
            source="fixed-method",
            outcome=self.decision,
            reason=f"Configured fixed {self.decision} route",
        )
        return RouteResult(
            action="execute",
            model=self.model,
            prompt=task.prompt,
            role=self.role,
            phase="fixed",
            label="fixed",
            routing_events=(event,),
        )
