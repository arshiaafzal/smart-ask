"""Application service that connects routing methods to model executors."""

from __future__ import annotations

from collections.abc import Callable

from .domain import (
    Attempt,
    Context,
    ExecutionRequest,
    ModelResult,
    RouteResult,
    RunResult,
    Task,
)
from .executors.base import ModelExecutor
from .methods.base import RoutingMethod


RouteCallback = Callable[[RouteResult, int], None]
ResultCallback = Callable[[ModelResult, int], None]


class SmartAsk:
    """Coordinate routing and model execution for one task at a time."""

    def __init__(
        self,
        method: RoutingMethod,
        executor: ModelExecutor,
        max_attempts: int = 3,
    ):
        """Configure the routing method, execution backend, and loop guard."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if (
            getattr(method, "requires_response_text", False)
            and not getattr(executor, "captures_output", False)
        ):
            raise ValueError(
                "This routing method requires an executor that captures response text"
            )
        self.method = method
        self.executor = executor
        self.max_attempts = max_attempts

    def plan(self, task: Task, context: Context = Context()) -> RouteResult:
        """Return the method's next route without executing a model."""

        return self.method.route(task, context)

    def run(self, task: Task) -> ModelResult:
        """Route and execute a task, returning its final model response."""

        return self.run_detailed(task).final_result

    def run_detailed(
        self,
        task: Task,
        initial_route: RouteResult | None = None,
        on_route: RouteCallback | None = None,
        on_result: ResultCallback | None = None,
    ) -> RunResult:
        """Run a task and retain every routing event and model attempt."""

        context = Context()
        route = initial_route or self.method.route(task, context)

        for attempt_number in range(1, self.max_attempts + 1):
            routing_events = context.routing_events + route.routing_events
            context = Context(
                attempts=context.attempts,
                routing_events=routing_events,
            )

            if route.action == "accept":
                if not context.attempts:
                    raise RuntimeError("Routing method accepted before any model attempt")
                return RunResult(task, context.attempts, context.routing_events)

            if on_route:
                on_route(route, attempt_number)

            result = self.executor.execute(ExecutionRequest(
                model=route.model or "",
                prompt=route.prompt or "",
            ))
            attempt = Attempt(route=route, result=result)
            context = Context(
                attempts=context.attempts + (attempt,),
                routing_events=context.routing_events,
            )

            if on_result:
                on_result(result, attempt_number)

            route = self.method.route(task, context)

        if route.action == "accept":
            routing_events = context.routing_events + route.routing_events
            return RunResult(task, context.attempts, routing_events)
        raise RuntimeError(
            f"Routing method exceeded the {self.max_attempts}-attempt safety limit"
        )
