"""Safe YAML loading and prompt resolution for strategy configurations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from pydantic import BaseModel, ValidationError

from .errors import StrategyConfigError
from .schema import FilePromptConfig, InlinePromptConfig, PromptConfig, StrategyConfig


class _UniqueKeyLoader:
    """Namespace populated lazily with a duplicate-rejecting SafeLoader."""

    loader = None


def _yaml_loader():
    try:
        import yaml
    except ImportError as exc:
        raise StrategyConfigError(
            "PyYAML is required to load strategy files; install the project dependencies"
        ) from exc

    if _UniqueKeyLoader.loader is None:
        class UniqueKeySafeLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise StrategyConfigError(f"duplicate YAML key: {key!r}")
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        UniqueKeySafeLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping,
        )
        _UniqueKeyLoader.loader = UniqueKeySafeLoader
    return yaml, _UniqueKeyLoader.loader


def _iter_prompt_configs(value: Any) -> Iterator[PromptConfig]:
    if isinstance(value, (InlinePromptConfig, FilePromptConfig)):
        yield value
        return
    if isinstance(value, BaseModel):
        for field_name in value.__class__.model_fields:
            yield from _iter_prompt_configs(getattr(value, field_name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_prompt_configs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_prompt_configs(item)


@dataclass(frozen=True)
class LoadedStrategy:
    """A validated strategy plus its resolved prompt contents and identity."""

    config: StrategyConfig
    path: Path
    digest: str
    _prompt_texts: Mapping[str, str]

    def resolve_prompt(self, prompt: PromptConfig) -> str:
        if isinstance(prompt, InlinePromptConfig):
            return prompt.text
        resolved = str(_resolve_prompt_path(self.path, prompt.path))
        return self._prompt_texts[resolved]

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot including prompt hashes."""

        prompts = []
        for path, text in sorted(self._prompt_texts.items()):
            prompts.append({
                "path": path,
                "sha256": sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            })
        return {
            "name": self.config.name,
            "path": str(self.path),
            "digest": self.digest,
            "config": self.config.model_dump(mode="json"),
            "prompts": prompts,
        }


def _resolve_prompt_path(config_path: Path, prompt_path: str) -> Path:
    prompt_path = Path(prompt_path).expanduser()
    path = prompt_path if prompt_path.is_absolute() else config_path.parent / prompt_path
    return path.resolve()


def _format_validation_error(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def load_strategy(path: str | Path) -> LoadedStrategy:
    """Load, strictly validate, and resolve one YAML strategy file."""

    source_path = Path(path).expanduser().resolve()
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrategyConfigError(f"cannot read strategy {source_path}: {exc}") from exc

    yaml, loader = _yaml_loader()
    try:
        documents = list(yaml.load_all(source_text, Loader=loader))
    except StrategyConfigError:
        raise
    except yaml.YAMLError as exc:
        raise StrategyConfigError(f"invalid YAML in {source_path}: {exc}") from exc
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise StrategyConfigError("a strategy file must contain exactly one mapping document")

    try:
        config = StrategyConfig.model_validate(documents[0])
    except ValidationError as exc:
        raise StrategyConfigError(_format_validation_error(exc)) from exc

    prompt_texts: dict[str, str] = {}
    for prompt in _iter_prompt_configs(config):
        if isinstance(prompt, InlinePromptConfig):
            continue
        prompt_path = _resolve_prompt_path(source_path, prompt.path)
        if not prompt_path.is_file():
            raise StrategyConfigError(f"prompt file does not exist: {prompt_path}")
        try:
            text = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise StrategyConfigError(f"cannot read prompt {prompt_path}: {exc}") from exc
        if not text.strip():
            raise StrategyConfigError(f"prompt file is empty: {prompt_path}")
        prompt_texts[str(prompt_path)] = text

    method = config.method
    if getattr(method, "type", None) == "cascade":
        suffix = (
            method.escalation.self_check_suffix.text
            if isinstance(method.escalation.self_check_suffix, InlinePromptConfig)
            else prompt_texts[str(_resolve_prompt_path(
                source_path,
                method.escalation.self_check_suffix.path,
            ))]
        )
        if method.escalation.marker not in suffix:
            raise StrategyConfigError(
                "method.escalation.self_check_suffix must contain the configured marker"
            )

    prompt_identity = []
    for prompt in _iter_prompt_configs(config):
        if isinstance(prompt, FilePromptConfig):
            resolved = str(_resolve_prompt_path(source_path, prompt.path))
            prompt_identity.append({
                "declared_path": prompt.path,
                "sha256": sha256(prompt_texts[resolved].encode("utf-8")).hexdigest(),
            })
    identity = {
        "config": config.model_dump(mode="json"),
        "prompts": sorted(prompt_identity, key=lambda item: item["declared_path"]),
    }
    digest = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LoadedStrategy(
        config=config,
        path=source_path,
        digest=digest,
        _prompt_texts=MappingProxyType(prompt_texts),
    )
