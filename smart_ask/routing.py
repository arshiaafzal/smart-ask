"""Routing-only service shared by task and conversation runtimes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from numbers import Integral
from types import MappingProxyType

from .domain import Context, RouteResult, Task
from .methods.base import RoutingMethod
from .metrics import RunStats, StatsCapture, StatsCollector


class SmartRouter:
    """Select routes without owning or invoking a generation backend."""

    def __init__(
        self,
        method: RoutingMethod,
        *,
        max_attempts: int,
        strategy_id: str | None = None,
        stats_collector: StatsCollector | None = None,
    ):
        if not callable(getattr(method, "route", None)):
            raise TypeError("method must expose a callable route")
        if not isinstance(getattr(method, "requires_response_text", None), bool):
            raise TypeError("method.requires_response_text must be a boolean")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, Integral)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
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
                "SmartRouter and its model-backed classifier must share one "
                "StatsCollector"
            )

        metrics_executors = {}
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

        self._method = method
        self._max_attempts = int(max_attempts)
        self._strategy_id = strategy_id
        self._stats_collector = stats_collector
        self._metrics_executors = MappingProxyType(metrics_executors)

    @property
    def method(self) -> RoutingMethod:
        return self._method

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
    def metrics_executors(self) -> Mapping[str, tuple[object, ...]]:
        return self._metrics_executors

    def route(self, task: Task, context: Context) -> RouteResult:
        self._validate_task(task)
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        route = self._method.route(task, context)
        if not isinstance(route, RouteResult):
            raise TypeError("method.route must return a RouteResult")
        return route

    def plan(self, task: Task) -> RouteResult:
        """Select the initial route for a fresh task without generation."""

        return self.route(task, Context())

    def plan_with_stats(self, task: Task) -> tuple[RouteResult, RunStats]:
        """Plan one fresh task and return classifier/routing metrics."""

        with self.capture_stats(task_id=task.task_id) as capture:
            route = self.plan(task)
        stats = capture.stats.with_routing_counts(
            generation_attempts=0,
            routing_events=len(route.routing_events),
        )
        return route, stats

    @contextmanager
    def capture_stats(self, *, task_id: str | None = None) -> Iterator[StatsCapture]:
        with self._stats_collector.capture(
            strategy_id=self._strategy_id,
            task_id=task_id,
        ) as capture:
            yield capture

    @staticmethod
    def _validate_task(task: Task) -> None:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
