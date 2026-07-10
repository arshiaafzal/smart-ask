"""Strict deployment configuration for the external Claude Code adapter."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AdapterConfigError(ValueError):
    """Raised when external adapter configuration is invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ListenConfig(_Model):
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)

    @property
    def is_loopback(self) -> bool:
        if self.host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False


class AuthConfig(_Model):
    token_env: str = "SMART_ASK_CLAUDE_CODE_TOKEN"
    required_on_loopback: bool = True

    @field_validator("token_env")
    @classmethod
    def valid_env_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("token_env must be an environment-variable name")
        return value


class LimitsConfig(_Model):
    max_request_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)
    max_concurrent_requests: int = Field(default=32, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)


class SecurityConfig(_Model):
    allowed_strategy_roots: tuple[str, ...] = ()

    @field_validator("allowed_strategy_roots", mode="before")
    @classmethod
    def list_to_tuple(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("allowed_strategy_roots")
    @classmethod
    def absolute_unique_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        roots = []
        for raw in value:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                raise ValueError("allowed_strategy_roots must be absolute")
            roots.append(str(path.resolve()))
        if len(set(roots)) != len(roots):
            raise ValueError("allowed_strategy_roots must be unique")
        return tuple(roots)


class MetricsConfig(_Model):
    jsonl_path: str | None = None

    @field_validator("jsonl_path")
    @classmethod
    def normalize_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("jsonl_path must be non-empty text or null")
        return str(Path(value).expanduser().resolve())


class AdapterConfig(_Model):
    schema_version: Literal[1]
    listen: ListenConfig = Field(default_factory=ListenConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    strategies: tuple[str, ...]
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @field_validator("strategies", mode="before")
    @classmethod
    def strategy_list_to_tuple(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("strategies")
    @classmethod
    def nonempty_unique_strategies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one strategy is required")
        if any(not item or item != item.strip() for item in value):
            raise ValueError("strategy references must be non-empty and trimmed")
        if len(set(value)) != len(value):
            raise ValueError("strategy references must be unique")
        return value


def _yaml_loader():
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        """Safe loader with duplicate-key rejection."""

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise AdapterConfigError(f"duplicate adapter YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml, UniqueKeyLoader


def load_adapter_config(path: str | Path) -> AdapterConfig:
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AdapterConfigError(f"cannot read adapter config {source}: {exc}") from exc
    yaml, loader = _yaml_loader()
    try:
        documents = list(yaml.load_all(text, Loader=loader))
    except AdapterConfigError:
        raise
    except yaml.YAMLError as exc:
        raise AdapterConfigError(f"invalid adapter YAML: {exc}") from exc
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise AdapterConfigError("adapter config must contain exactly one mapping")
    try:
        return AdapterConfig.model_validate(documents[0])
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        )
        raise AdapterConfigError(details) from exc
