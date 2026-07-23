"""Trusted deployment targets available to declarative strategies.

Strategy YAML deliberately contains only target identifiers.  Network
endpoints, credential names, executable commands, and hard operational limits
live here (or in another deployment-owned registry supplied by the caller).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from numbers import Integral, Real
import re
import os
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlparse

from .._numeric import is_finite_real


TransportKind = Literal["anthropic", "openrouter", "openai", "groq", "ollama"]
_TARGET_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


@dataclass(frozen=True)
class TargetLimits:
    """Deployment-enforced ceilings that strategy parameters cannot exceed."""

    max_output_tokens: int = 32768
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, Integral)
            or self.max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, Real)
            or not is_finite_real(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "max_output_tokens", int(self.max_output_tokens))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True)
class TargetDefinition:
    """One trusted physical model target.

    This value is deployment configuration, not strategy configuration.
    Credential handles therefore never originate in an untrusted strategy
    file.
    """

    target_id: str
    transport: TransportKind
    model: str
    base_url: str | None = None
    credential_env: str | None = None
    think: bool = False
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({"text", "streaming"})
    )
    limits: TargetLimits = field(default_factory=TargetLimits)

    def __post_init__(self) -> None:
        for name in ("target_id", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty trimmed text")
        if not _TARGET_ID.fullmatch(self.target_id):
            raise ValueError("target_id must be a lowercase hyphenated identifier")
        if self.transport not in (
            "anthropic",
            "openrouter",
            "openai",
            "groq",
            "ollama",
        ):
            raise ValueError(f"unsupported target transport: {self.transport!r}")
        if self.base_url is not None:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("base_url must be an absolute HTTP(S) URL")
        for name in ("credential_env",):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be non-empty trimmed text or None")
        if not isinstance(self.think, bool):
            raise TypeError("think must be a boolean")
        capabilities = frozenset(self.capabilities)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in capabilities
        ):
            raise ValueError("capabilities must contain non-empty trimmed strings")
        object.__setattr__(self, "capabilities", capabilities)
        if not isinstance(self.limits, TargetLimits):
            raise TypeError("limits must be TargetLimits")


class TargetRegistry(Mapping[str, TargetDefinition]):
    """Immutable set of deployment-approved model targets."""

    def __init__(self, targets: Iterable[TargetDefinition]):
        values: dict[str, TargetDefinition] = {}
        for target in targets:
            if not isinstance(target, TargetDefinition):
                raise TypeError("targets must contain TargetDefinition values")
            if target.target_id in values:
                raise ValueError(f"duplicate target ID: {target.target_id}")
            values[target.target_id] = target
        if not values:
            raise ValueError("a target registry cannot be empty")
        self._targets = MappingProxyType(values)

    def __getitem__(self, key: str) -> TargetDefinition:
        try:
            return self._targets[key]
        except KeyError as exc:
            raise KeyError(f"unknown deployment target: {key}") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._targets)

    def __len__(self) -> int:
        return len(self._targets)

    def resolve(self, target_id: str) -> TargetDefinition:
        return self[target_id]

    def required_secret_envs(self, target_ids: Iterable[str]) -> frozenset[str]:
        return frozenset(
            target.credential_env
            for target in (self.resolve(value) for value in target_ids)
            if target.credential_env is not None
        )

    def snapshot(self, target_ids: Iterable[str]) -> tuple[dict[str, object], ...]:
        """Return reproducibility metadata without endpoints or secret handles."""

        values = []
        for target_id in sorted(set(target_ids)):
            target = self.resolve(target_id)
            values.append({
                "target_id": target.target_id,
                "transport": target.transport,
                "model": target.model,
                "capabilities": sorted(target.capabilities),
                "limits": {
                    "max_output_tokens": target.limits.max_output_tokens,
                    "timeout_seconds": target.limits.timeout_seconds,
                },
                "configuration_digest": self._target_digest(target),
            })
        return tuple(values)

    def digest(self, target_ids: Iterable[str]) -> str:
        return sha256(json.dumps(
            self.snapshot(target_ids),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def _target_digest(target: TargetDefinition) -> str:
        # Endpoint and credential-handle changes must affect reproducibility,
        # but their raw values are never exposed in the public snapshot.
        value = {
            "target_id": target.target_id,
            "transport": target.transport,
            "model": target.model,
            "base_url": target.base_url,
            "credential_env": target.credential_env,
            "think": target.think,
            "capabilities": sorted(target.capabilities),
            "limits": {
                "max_output_tokens": target.limits.max_output_tokens,
                "timeout_seconds": target.limits.timeout_seconds,
            },
        }
        return sha256(json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


_OPENROUTER = "https://openrouter.ai/api/v1"
_ANTHROPIC = "https://api.anthropic.com"
_OPENAI = "https://api.openai.com/v1"
_GROQ = "https://api.groq.com/openai/v1"
_OLLAMA = "http://127.0.0.1:11434/api"
_STRUCTURED = frozenset({"text", "streaming", "tools", "images", "reasoning"})


DEFAULT_TARGET_REGISTRY = TargetRegistry((
    TargetDefinition(
        "anthropic-claude-sonnet",
        "anthropic",
        "claude-sonnet-4-6",
        base_url=_ANTHROPIC,
        credential_env="ANTHROPIC_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "anthropic-claude-opus",
        "anthropic",
        "claude-opus-4-6",
        base_url=_ANTHROPIC,
        credential_env="ANTHROPIC_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "openrouter-gemini-flash-lite",
        "openrouter",
        "google/gemini-2.5-flash-lite",
        base_url=_OPENROUTER,
        credential_env="OPENROUTER_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "openrouter-claude-opus-4-8",
        "openrouter",
        "anthropic/claude-opus-4.8",
        base_url=_OPENROUTER,
        credential_env="OPENROUTER_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "openai-codex-mini",
        "openai",
        "gpt-5.1-codex-mini",
        base_url=_OPENAI,
        credential_env="OPENAI_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "openai-codex",
        "openai",
        "gpt-5.3-codex",
        base_url=_OPENAI,
        credential_env="OPENAI_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "groq-oss-20b",
        "groq",
        "openai/gpt-oss-20b",
        base_url=_GROQ,
        credential_env="GROQ_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "groq-oss-120b",
        "groq",
        "openai/gpt-oss-120b",
        base_url=_GROQ,
        credential_env="GROQ_API_KEY",
        capabilities=_STRUCTURED,
    ),
    TargetDefinition(
        "local-qwen3-14b",
        "ollama",
        "qwen3:14b",
        base_url=_OLLAMA,
        capabilities=_STRUCTURED,
    ),
))


def default_target_registry(
    env: Mapping[str, str] | None = None,
) -> TargetRegistry:
    """Build the process-owned defaults with explicit operator overrides."""

    values = os.environ if env is None else env
    ollama_url = values.get("SMART_ASK_OLLAMA_URL")
    foundry_url = values.get("ANTHROPIC_FOUNDRY_BASE_URL")
    use_foundry = bool(foundry_url and values.get("ANTHROPIC_FOUNDRY_API_KEY"))
    targets = []
    for target in DEFAULT_TARGET_REGISTRY.values():
        if target.target_id == "local-qwen3-14b" and ollama_url:
            target = replace(target, base_url=ollama_url)
        elif target.target_id == "anthropic-claude-sonnet":
            target = replace(
                target,
                model=values.get("ANTHROPIC_DEFAULT_SONNET_MODEL", target.model),
                base_url=foundry_url if use_foundry else target.base_url,
                credential_env=(
                    "ANTHROPIC_FOUNDRY_API_KEY"
                    if use_foundry
                    else target.credential_env
                ),
            )
        elif target.target_id == "anthropic-claude-opus":
            target = replace(
                target,
                model=values.get("ANTHROPIC_DEFAULT_OPUS_MODEL", target.model),
                base_url=foundry_url if use_foundry else target.base_url,
                credential_env=(
                    "ANTHROPIC_FOUNDRY_API_KEY"
                    if use_foundry
                    else target.credential_env
                ),
            )
        targets.append(target)
    return TargetRegistry(targets)
