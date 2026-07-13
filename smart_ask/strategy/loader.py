"""Safe YAML loading and prompt resolution for strategy configurations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from .errors import StrategyConfigError
from .schema import (
    CascadeMethodConfig,
    FilePromptConfig,
    InlinePromptConfig,
    PromptConfig,
    StrategyConfig,
)


BUILTIN_STRATEGY_PREFIX = "builtin:"
MAX_STRATEGY_BYTES = 1_048_576
MAX_PROMPT_BYTES = 1_048_576


class _UniqueKeyLoader:
    """Namespace populated lazily with a duplicate-rejecting SafeLoader."""

    loader = None


def compute_strategy_digest(
    config: StrategyConfig,
    prompt_hashes: Mapping[str, str],
) -> str:
    """Hash a validated config and its declared file-prompt identities."""

    if not isinstance(config, StrategyConfig):
        raise TypeError("config must be a StrategyConfig")
    declared_paths = {
        prompt.path
        for prompt in _iter_prompt_configs(config)
        if isinstance(prompt, FilePromptConfig)
    }
    if set(prompt_hashes) != declared_paths:
        raise ValueError("prompt hashes must exactly match declared file prompts")
    prompt_identity = []
    for declared_path in sorted(declared_paths):
        digest = prompt_hashes[declared_path]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("prompt hashes must be lowercase SHA-256 digests")
        prompt_identity.append({
            "declared_path": declared_path,
            "sha256": digest,
        })
    identity = {
        "config": config.model_dump(mode="json"),
        "prompts": prompt_identity,
    }
    return sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
                try:
                    duplicate = key in mapping
                except TypeError as exc:
                    raise StrategyConfigError(
                        "YAML mapping keys must be scalar values"
                    ) from exc
                if duplicate:
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

    def __post_init__(self) -> None:
        if not isinstance(self.config, StrategyConfig):
            raise TypeError("config must be a StrategyConfig")
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("path must be absolute")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("digest must be a lowercase SHA-256 digest")
        if not isinstance(self._prompt_texts, Mapping):
            raise TypeError("prompt texts must be a mapping")
        prompt_texts = dict(self._prompt_texts)
        declared_paths = {
            prompt.path
            for prompt in _iter_prompt_configs(self.config)
            if isinstance(prompt, FilePromptConfig)
        }
        if set(prompt_texts) != declared_paths:
            raise ValueError("prompt texts must exactly match declared file prompts")
        if any(
            not isinstance(text, str) or not text.strip()
            for text in prompt_texts.values()
        ):
            raise ValueError("resolved prompt texts must be non-empty strings")
        prompt_hashes = {
            declared_path: sha256(text.encode("utf-8")).hexdigest()
            for declared_path, text in prompt_texts.items()
        }
        if self.digest != compute_strategy_digest(self.config, prompt_hashes):
            raise ValueError("digest does not match the strategy config and prompts")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "_prompt_texts",
            MappingProxyType(prompt_texts),
        )

    def resolve_prompt(self, prompt: PromptConfig) -> str:
        if not isinstance(prompt, (InlinePromptConfig, FilePromptConfig)):
            raise TypeError("prompt must be an inline or file prompt configuration")
        if prompt not in tuple(_iter_prompt_configs(self.config)):
            path = getattr(prompt, "path", "<inline>")
            raise ValueError(f"prompt is not declared by this strategy: {path}")
        if isinstance(prompt, InlinePromptConfig):
            return prompt.text
        return self._prompt_texts[prompt.path]

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot including prompt hashes."""

        prompts_by_declared_path = {}
        for prompt in _iter_prompt_configs(self.config):
            if not isinstance(prompt, FilePromptConfig):
                continue
            text = self._prompt_texts[prompt.path]
            prompts_by_declared_path[prompt.path] = {
                "declared_path": prompt.path,
                "sha256": sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        return {
            "name": self.config.name,
            "digest": self.digest,
            "config": self.config.model_dump(mode="json"),
            "prompts": [
                prompts_by_declared_path[path]
                for path in sorted(prompts_by_declared_path)
            ],
        }


def _resolve_prompt_path(config_path: Path, prompt_path: str) -> Path:
    return (config_path.parent / prompt_path).resolve()


def _format_validation_error(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def _inside_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _checked_text(path: Path, *, limit: int, label: str) -> str:
    try:
        size = path.stat().st_size
        if size > limit:
            raise StrategyConfigError(
                f"{label} exceeds the {limit}-byte deployment limit: {path}"
            )
        return path.read_text(encoding="utf-8")
    except StrategyConfigError:
        raise
    except (OSError, UnicodeError) as exc:
        raise StrategyConfigError(f"cannot read {label} {path}: {exc}") from exc


def load_strategy(
    path: str | Path,
    *,
    allowed_roots: Sequence[str | Path] | None = None,
) -> LoadedStrategy:
    """Load a filesystem YAML or a ``builtin:<name>`` package resource."""

    builtin = isinstance(path, str) and path.startswith(BUILTIN_STRATEGY_PREFIX)
    source_path = _strategy_path(path).expanduser().resolve()
    roots = tuple(
        Path(root).expanduser().resolve()
        for root in (() if allowed_roots is None else allowed_roots)
    )
    if not builtin and roots and not _inside_roots(source_path, roots):
        raise StrategyConfigError(
            f"strategy is outside the deployment's allowed roots: {source_path}"
        )
    source_text = _checked_text(
        source_path,
        limit=MAX_STRATEGY_BYTES,
        label="strategy",
    )

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
        if prompt.path in prompt_texts:
            continue
        prompt_path = _resolve_prompt_path(source_path, prompt.path)
        if not builtin and roots and not _inside_roots(prompt_path, roots):
            raise StrategyConfigError(
                f"prompt is outside the deployment's allowed roots: {prompt_path}"
            )
        if not prompt_path.is_file():
            raise StrategyConfigError(f"prompt file does not exist: {prompt_path}")
        text = _checked_text(
            prompt_path,
            limit=MAX_PROMPT_BYTES,
            label="prompt",
        )
        if not text.strip():
            raise StrategyConfigError(f"prompt file is empty: {prompt_path}")
        prompt_texts[prompt.path] = text

    method = config.method
    if isinstance(method, CascadeMethodConfig):
        suffix = (
            method.escalation.self_check_suffix.text
            if isinstance(method.escalation.self_check_suffix, InlinePromptConfig)
            else prompt_texts[method.escalation.self_check_suffix.path]
        )
        if method.escalation.marker not in suffix:
            raise StrategyConfigError(
                "method.escalation.self_check_suffix must contain the configured marker"
            )

    prompt_hashes: dict[str, str] = {}
    for prompt in _iter_prompt_configs(config):
        if isinstance(prompt, FilePromptConfig):
            prompt_hashes[prompt.path] = sha256(
                prompt_texts[prompt.path].encode("utf-8")
            ).hexdigest()
    digest = compute_strategy_digest(config, prompt_hashes)
    return LoadedStrategy(
        config=config,
        path=source_path,
        digest=digest,
        _prompt_texts=prompt_texts,
    )


def _strategy_path(value: str | Path) -> Path:
    if not isinstance(value, str) or not value.startswith(BUILTIN_STRATEGY_PREFIX):
        return Path(value)
    name = value.removeprefix(BUILTIN_STRATEGY_PREFIX)
    if name.endswith(".yaml"):
        name = name[:-5]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise StrategyConfigError(
            "bundled strategy must use builtin:<name> without path separators"
        )
    resource = files("smart_ask").joinpath(
        "resources",
        "strategies",
        f"{name}.yaml",
    )
    path = Path(str(resource))
    if not path.is_file():
        raise StrategyConfigError(f"unknown bundled strategy: {name!r}")
    return path
