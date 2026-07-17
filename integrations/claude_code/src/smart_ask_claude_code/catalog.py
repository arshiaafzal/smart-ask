"""One external Claude model alias per compiled SmartAsk strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType

from smart_ask.conversation.engine import RunObserver, StrategyEngine
from smart_ask.conversation.metrics import RunMetricsStore
from smart_ask.conversation.model import RunRecord
from smart_ask.strategy import StrategyBuilder, load_strategy
from smart_ask.strategy.errors import StrategyConfigError
from smart_ask.strategy.loader import BUILTIN_STRATEGY_PREFIX, LoadedStrategy

from .config import AdapterConfig, AdapterConfigError


_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def _slug(reference: str) -> str:
    value = (
        reference.removeprefix(BUILTIN_STRATEGY_PREFIX)
        if reference.startswith(BUILTIN_STRATEGY_PREFIX)
        else Path(reference).stem
    )
    if value.endswith(".yaml"):
        value = value[:-5]
    if not _SLUG.fullmatch(value):
        raise AdapterConfigError(
            f"strategy {reference!r} needs a lowercase hyphenated filename"
        )
    return value


@dataclass(frozen=True)
class CatalogEntry:
    model_id: str
    display_name: str
    reference: str
    loaded: LoadedStrategy
    engine: StrategyEngine
    # Forced-hard engine used when the request contains tool definitions.
    # Prevents easy models (e.g. Gemini) from hallucinating invalid tool calls
    # against Claude Code's rich tool schema.
    engine_hard: StrategyEngine | None = None


class StrategyCatalog:
    """Adapter registry containing compiled engines, never routing policy."""

    def __init__(
        self,
        entries: tuple[CatalogEntry, ...],
        *,
        metrics: RunMetricsStore | None = None,
        resource_owners: tuple[object, ...] = (),
    ) -> None:
        by_model = {}
        for entry in entries:
            if entry.model_id in by_model:
                raise AdapterConfigError(f"duplicate model alias: {entry.model_id}")
            by_model[entry.model_id] = entry
        if not by_model:
            raise AdapterConfigError("at least one strategy is required")
        self._entries: Mapping[str, CatalogEntry] = MappingProxyType(by_model)
        self.metrics = metrics or RunMetricsStore()
        self._resource_owners = resource_owners

    @classmethod
    def from_config(
        cls,
        config: AdapterConfig,
        *,
        env: Mapping[str, str],
        loader: Callable[..., LoadedStrategy] = load_strategy,
        engine_builder: Callable[
            [LoadedStrategy, RunObserver | None], StrategyEngine
        ] | None = None,
        metrics: RunMetricsStore | None = None,
        trace_observer: RunObserver | None = None,
    ) -> "StrategyCatalog":
        resource_owners: tuple[object, ...] = ()
        hard_engine_builder: Callable[
            [LoadedStrategy, RunObserver | None], StrategyEngine
        ] | None = None
        if engine_builder is None:
            builder = StrategyBuilder(env=env)
            resource_owners = (builder,)
            engine_builder = lambda loaded, observer: builder.build_engine(
                loaded,
                observer=observer,
            )
            hard_engine_builder = lambda loaded, observer: builder.build_engine(
                loaded,
                force="hard",
                observer=observer,
            )
        entries = []
        for reference in config.strategies:
            if loader is load_strategy and not reference.startswith(
                BUILTIN_STRATEGY_PREFIX
            ):
                roots = config.security.allowed_strategy_roots
                if not roots:
                    raise AdapterConfigError(
                        f"custom strategy {reference!r} is not allowed"
                    )
                try:
                    loaded = loader(reference, allowed_roots=roots)
                except StrategyConfigError as exc:
                    raise AdapterConfigError(str(exc)) from exc
            else:
                loaded = loader(reference)
            slug = _slug(reference)
            # Build forced-hard engine; some fixed strategies don't support
            # force overrides, so fall back gracefully.
            engine_hard: StrategyEngine | None = None
            if hard_engine_builder is not None:
                try:
                    engine_hard = hard_engine_builder(loaded, trace_observer)
                except Exception:
                    engine_hard = None
            entries.append(CatalogEntry(
                model_id=f"claude-smart-ask-{slug}",
                display_name="SmartAsk: " + " ".join(
                    part.capitalize() for part in slug.split("-")
                ),
                reference=reference,
                loaded=loaded,
                engine=engine_builder(loaded, trace_observer),
                engine_hard=engine_hard,
            ))
        return cls(
            tuple(entries),
            metrics=metrics,
            resource_owners=resource_owners,
        )

    def resolve(self, model_id: str) -> CatalogEntry:
        try:
            return self._entries[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model alias: {model_id}") from exc

    def __iter__(self):
        return iter(self._entries.values())

    def record(self, record: RunRecord) -> None:
        """Persist content-free metrics for one completed invocation."""
        self.metrics.record(record)

    async def aclose(self) -> None:
        seen = set()
        for entry in self:
            for engine in (entry.engine, entry.engine_hard):
                if engine is None or id(engine) in seen:
                    continue
                seen.add(id(engine))
                await engine.aclose()
        for owner in self._resource_owners:
            closer = getattr(owner, "aclose", None)
            if callable(closer):
                await closer()

    def discovery_payload(self) -> dict[str, list[dict[str, str]]]:
        return {
            "data": [
                {"id": entry.model_id, "display_name": entry.display_name}
                for entry in self
            ]
        }
