"""One external Claude model alias per SmartAsk strategy YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping

from smart_ask.conversation import ConversationMetricsStore, ConversationRuntime
from smart_ask.strategy import StrategyBuilder, load_strategy
from smart_ask.strategy.loader import BUILTIN_STRATEGY_PREFIX, LoadedStrategy

from .config import AdapterConfig, AdapterConfigError


_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def _slug(reference: str) -> str:
    if reference.startswith(BUILTIN_STRATEGY_PREFIX):
        value = reference.removeprefix(BUILTIN_STRATEGY_PREFIX)
    else:
        value = Path(reference).stem
    if value.endswith(".yaml"):
        value = value[:-5]
    if not _SLUG.fullmatch(value):
        raise AdapterConfigError(
            f"strategy {reference!r} needs a lowercase hyphenated filename"
        )
    return value


def _inside(path: Path, roots: tuple[str, ...]) -> bool:
    resolved = path.resolve()
    return any(
        resolved == Path(root) or resolved.is_relative_to(Path(root))
        for root in roots
    )


def _all_strategy_files_inside(
    loaded: LoadedStrategy,
    roots: tuple[str, ...],
) -> bool:
    declared_prompts = loaded.manifest()["prompts"]
    files = [loaded.path]
    files.extend(
        (loaded.path.parent / item["declared_path"]).resolve()
        for item in declared_prompts
    )
    return all(_inside(path, roots) for path in files)


@dataclass(frozen=True)
class CatalogEntry:
    model_id: str
    display_name: str
    reference: str
    loaded: LoadedStrategy
    runtime: ConversationRuntime


class StrategyCatalog:
    """Adapter registry that delegates every strategy operation to SmartAsk."""

    def __init__(self, entries: tuple[CatalogEntry, ...]):
        by_model = {}
        for entry in entries:
            if entry.model_id in by_model:
                raise AdapterConfigError(f"duplicate model alias: {entry.model_id}")
            by_model[entry.model_id] = entry
        if not by_model:
            raise AdapterConfigError("at least one strategy is required")
        self._entries: Mapping[str, CatalogEntry] = MappingProxyType(by_model)

    @classmethod
    def from_config(
        cls,
        config: AdapterConfig,
        *,
        env: Mapping[str, str],
        loader: Callable[[str], LoadedStrategy] = load_strategy,
        runtime_builder: Callable[[LoadedStrategy], ConversationRuntime] | None = None,
        metrics: ConversationMetricsStore | None = None,
    ) -> "StrategyCatalog":
        if runtime_builder is None:
            builder = StrategyBuilder(env=env)
            runtime_builder = lambda loaded: builder.build_conversation_runtime(
                loaded,
                metrics=metrics,
            )
        elif metrics is not None:
            raise ValueError("metrics cannot be combined with a custom runtime_builder")
        entries = []
        for reference in config.strategies:
            loaded = loader(reference)
            if not reference.startswith(BUILTIN_STRATEGY_PREFIX):
                roots = config.security.allowed_strategy_roots
                if not roots or not _all_strategy_files_inside(loaded, roots):
                    raise AdapterConfigError(
                        f"custom strategy {reference!r} or one of its prompts "
                        "is outside allowed roots"
                    )
            slug = _slug(reference)
            entries.append(CatalogEntry(
                model_id=f"claude-smart-ask-{slug}",
                display_name="SmartAsk: " + " ".join(
                    part.capitalize() for part in slug.split("-")
                ),
                reference=reference,
                loaded=loaded,
                runtime=runtime_builder(loaded),
            ))
        return cls(tuple(entries))

    def resolve(self, model_id: str) -> CatalogEntry:
        try:
            return self._entries[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model alias: {model_id}") from exc

    def __iter__(self):
        return iter(self._entries.values())

    async def aclose(self) -> None:
        seen = set()
        for entry in self:
            runtime = entry.runtime
            if id(runtime) in seen:
                continue
            seen.add(id(runtime))
            closer = getattr(runtime, "aclose", None)
            if callable(closer):
                await closer()

    def discovery_payload(self) -> dict[str, list[dict[str, str]]]:
        return {
            "data": [
                {"id": entry.model_id, "display_name": entry.display_name}
                for entry in self
            ]
        }
