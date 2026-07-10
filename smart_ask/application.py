"""Application service that connects routing methods to model executors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType

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
from .metrics import RunStats, StatsCapture, StatsCollector
from .routing import SmartRouter


RouteCallback = Callable[[RouteResult, int], None]
ResultCallback = Callable[[ModelResult, int], None]


class SmartAsk:
    """Coordinate routing and model execution for one task at a time."""

    def __init__(
        self,
        method: RoutingMethod,
        executor: ModelExecutor,
        max_attempts: int,
        *,
        strategy_id: str | None = None,
        stats_collector: StatsCollector | None = None,
    ):
        """Configure the routing method, execution backend, and loop guard."""
        router = SmartRouter(
            method,
            max_attempts=max_attempts,
            strategy_id=strategy_id,
            stats_collector=stats_collector,
        )
        self._initialize(router, executor)

    @classmethod
    def from_router(
        cls,
        router: SmartRouter,
        executor: ModelExecutor,
    ) -> "SmartAsk":
        """Compose an already-built router with a generation backend."""

        if not isinstance(router, SmartRouter):
            raise TypeError("router must be a SmartRouter")
        application = cls.__new__(cls)
        application._initialize(router, executor)
        return application

    def _initialize(self, router: SmartRouter, executor: ModelExecutor) -> None:
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must expose a callable execute")
        captures_output = getattr(executor, "captures_output", None)
        if not isinstance(captures_output, bool):
            raise TypeError("executor.captures_output must be a boolean")
        if router.method.requires_response_text and not captures_output:
            raise ValueError(
                "This routing method requires an executor that captures response text"
            )

        # SmartAsk owns generation instrumentation. Model-backed classifiers
        # are already instrumented by the shared routing collector.
        self._router = router
        self._executor = router.stats_collector.wrap(executor, "generation")
        metrics_executors: dict[str, tuple[ModelExecutor, ...]] = {
            "generation": (self._executor,),
        }
        metrics_executors.update(router.metrics_executors)
        self._metrics_executors = MappingProxyType(metrics_executors)

    @property
    def method(self) -> RoutingMethod:
        return self._router.method

    @property
    def router(self) -> SmartRouter:
        return self._router

    @property
    def executor(self) -> ModelExecutor:
        return self._executor

    @property
    def max_attempts(self) -> int:
        return self._router.max_attempts

    @property
    def strategy_id(self) -> str | None:
        return self._router.strategy_id

    @property
    def stats_collector(self) -> StatsCollector:
        return self._router.stats_collector

    @property
    def metrics_executors(self) -> Mapping[str, tuple[ModelExecutor, ...]]:
        """Executor wrappers that provide this application's call evidence."""

        return self._metrics_executors

    def plan(self, task: Task) -> RouteResult:
        """Select a fresh task's initial route without executing generation."""
        return self._router.plan(task)

    def run(self, task: Task) -> ModelResult:
        """Route and execute a task, returning its final model response."""

        return self.run_detailed(task).final_result

    def run_with_stats(self, task: Task) -> tuple[ModelResult, RunStats]:
        """Run one task and return its final response plus immutable metrics."""

        run, stats = self.run_detailed_with_stats(task)
        return run.final_result, stats

    def run_detailed_with_stats(
        self,
        task: Task,
        on_route: RouteCallback | None = None,
        on_result: ResultCallback | None = None,
    ) -> tuple[RunResult, RunStats]:
        """Run one task and return its audit trail plus a metrics snapshot."""

        self._validate_task(task)
        with self.capture_stats(task_id=task.task_id) as capture:
            run = self.run_detailed(
                task,
                on_route=on_route,
                on_result=on_result,
            )
        return run, capture.stats.with_run_result(run)

    @contextmanager
    def capture_stats(self, *, task_id: str | None = None) -> Iterator[StatsCapture]:
        """Capture custom callback/manual workflows and partial failures."""
        with self._router.capture_stats(task_id=task_id) as capture:
            yield capture

    def run_detailed(
        self,
        task: Task,
        on_route: RouteCallback | None = None,
        on_result: ResultCallback | None = None,
    ) -> RunResult:
        """Run a task and retain every routing event and model attempt."""

        self._validate_task(task)
        for name, callback in (("on_route", on_route), ("on_result", on_result)):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")
        context = Context()
        route = self._select_route(task, context)

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

            if route.model is None or route.prompt is None or route.role is None:
                raise RuntimeError("execute route lost required execution fields")
            result = self._executor.execute(ExecutionRequest(
                model=route.model,
                prompt=route.prompt,
                role=route.role,
            ))
            if not isinstance(result, ModelResult):
                raise TypeError("executor.execute must return a ModelResult")
            attempt = Attempt(route=route, result=result)
            context = Context(
                attempts=context.attempts + (attempt,),
                routing_events=context.routing_events,
            )

            if on_result:
                on_result(result, attempt_number)

            route = self._select_route(task, context)

        if route.action == "accept":
            routing_events = context.routing_events + route.routing_events
            return RunResult(task, context.attempts, routing_events)
        raise RuntimeError(
            f"Routing method exceeded the {self.max_attempts}-attempt safety limit"
        )

    @staticmethod
    def _validate_task(task: Task) -> None:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

    def _select_route(self, task: Task, context: Context) -> RouteResult:
        return self._router.route(task, context)
