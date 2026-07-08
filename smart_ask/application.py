"""Application service that connects routing methods to model executors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Integral
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

        if not callable(getattr(method, "route", None)):
            raise TypeError("method must expose a callable route")
        requires_response_text = getattr(method, "requires_response_text", None)
        if not isinstance(requires_response_text, bool):
            raise TypeError("method.requires_response_text must be a boolean")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must expose a callable execute")
        captures_output = getattr(executor, "captures_output", None)
        if not isinstance(captures_output, bool):
            raise TypeError("executor.captures_output must be a boolean")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, Integral)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if requires_response_text and not captures_output:
            raise ValueError(
                "This routing method requires an executor that captures response text"
            )
        if strategy_id is not None and (
            not isinstance(strategy_id, str)
            or not strategy_id
            or strategy_id != strategy_id.strip()
        ):
            raise ValueError("strategy_id must be a non-empty trimmed string or None")
        if stats_collector is None:
            stats_collector = StatsCollector()
        elif not isinstance(stats_collector, StatsCollector):
            raise TypeError("stats_collector must be a StatsCollector or None")
        classifier = getattr(method, "classifier", None)
        classifier_collector = getattr(classifier, "stats_collector", None)
        if (
            classifier_collector is not None
            and classifier_collector is not stats_collector
        ):
            raise ValueError(
                "SmartAsk and its model-backed classifier must share one "
                "StatsCollector"
            )
        # SmartAsk is the sole owner of generation instrumentation. Classifier
        # collaborators own instrumentation of their hidden executor calls.
        self._method = method
        self._executor = stats_collector.wrap(executor, "generation")
        metrics_executors: dict[str, tuple[ModelExecutor, ...]] = {
            "generation": (self._executor,),
        }
        if classifier_collector is not None:
            classifier_executor = getattr(classifier, "executor", None)
            if not stats_collector.is_instrumented(
                classifier_executor,
                channel="classifier",
            ):
                raise ValueError(
                    "a metrics-aware classifier must expose its classifier "
                    "executor instrumented by the shared StatsCollector"
                )
            metrics_executors["classifier"] = (classifier_executor,)
        self._max_attempts = int(max_attempts)
        self._strategy_id = strategy_id
        self._stats_collector = stats_collector
        self._metrics_executors = MappingProxyType(metrics_executors)

    @property
    def method(self) -> RoutingMethod:
        return self._method

    @property
    def executor(self) -> ModelExecutor:
        return self._executor

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def strategy_id(self) -> str | None:
        return self._strategy_id

    @property
    def stats_collector(self) -> StatsCollector:
        return self._stats_collector

    @property
    def metrics_executors(self) -> Mapping[str, tuple[ModelExecutor, ...]]:
        """Executor wrappers that provide this application's call evidence."""

        return self._metrics_executors

    def plan(self, task: Task) -> RouteResult:
        """Select a fresh task's initial route without executing generation."""

        self._validate_task(task)
        return self._select_route(task, Context())

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

        with self._stats_collector.capture(
            strategy_id=self._strategy_id,
            task_id=task_id,
        ) as capture:
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

        for attempt_number in range(1, self._max_attempts + 1):
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
            f"Routing method exceeded the {self._max_attempts}-attempt safety limit"
        )

    @staticmethod
    def _validate_task(task: Task) -> None:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

    def _select_route(self, task: Task, context: Context) -> RouteResult:
        route = self._method.route(task, context)
        if not isinstance(route, RouteResult):
            raise TypeError("method.route must return a RouteResult")
        return route
