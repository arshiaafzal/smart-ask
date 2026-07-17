"""Strict declarative schema for SmartAsk strategy policy.

Provider endpoints, credentials, and executable commands are intentionally not
part of this schema.  Profiles reference trusted deployment targets instead.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InlinePromptConfig(ConfigModel):
    type: Literal["inline"]
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt text must not be empty")
        return value


class FilePromptConfig(ConfigModel):
    type: Literal["file"]
    path: str

    @field_validator("path")
    @classmethod
    def path_must_be_portable_and_relative(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("prompt path must be non-empty and trimmed")
        if (
            value.startswith("~")
            or "\\" in value
            or Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or bool(PureWindowsPath(value).drive)
        ):
            raise ValueError(
                "prompt path must be a portable path relative to the strategy file"
            )
        return value


PromptConfig = Annotated[
    InlinePromptConfig | FilePromptConfig,
    Field(discriminator="type"),
]


class ModelParametersConfig(ConfigModel):
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None


class ModelProfileConfig(ConfigModel):
    """Strategy-owned transforms applied to a trusted deployment target."""

    target: str
    system_prompt: PromptConfig | None = None
    parameters: ModelParametersConfig = Field(default_factory=ModelParametersConfig)
    context_messages: int | None = Field(default=None, gt=0)

    @field_validator("target")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("target must be non-empty and trimmed")
        return value


class ClassifierParametersConfig(ConfigModel):
    max_tokens: int = Field(default=20, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None


class LLMClassifierConfig(ConfigModel):
    type: Literal["llm"]
    target: str
    prompt: PromptConfig
    fallback: Literal["easy", "hard", "raise"]
    missing_input: Literal["easy", "hard", "raise", "classify-tool-result"] = "hard"
    projection: Literal["latest-user-text", "full-conversation"] = (
        "latest-user-text"
    )
    max_prompt_chars: int = Field(default=1200, gt=0)
    parameters: ClassifierParametersConfig = Field(
        default_factory=ClassifierParametersConfig
    )

    @field_validator("target")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("classifier target must be non-empty and trimmed")
        return value


ClassifierConfig = Annotated[LLMClassifierConfig, Field(discriminator="type")]


class MarkerEscalationConfig(ConfigModel):
    type: Literal["marker"]
    marker: str
    self_check_suffix: PromptConfig
    escalation_prefix: PromptConfig
    tool_output: Literal["accept-and-pin", "escalate", "fail"] = (
        "accept-and-pin"
    )

    @field_validator("marker")
    @classmethod
    def marker_must_be_one_line(cls, value: str) -> str:
        if (
            not value.strip()
            or value != value.strip()
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError("marker must be nonempty, trimmed, and single-line")
        return value


EscalationConfig = Annotated[
    MarkerEscalationConfig,
    Field(discriminator="type"),
]


class RouteMemoryConfig(ConfigModel):
    enabled: bool = False
    ttl_seconds: float = Field(default=3600.0, gt=0)
    max_entries: int = Field(default=10000, gt=0)


class DifficultyMethodConfig(ConfigModel):
    type: Literal["difficulty"]
    classifier: ClassifierConfig
    easy: str
    hard: str
    route_memory: RouteMemoryConfig = Field(default_factory=RouteMemoryConfig)


class CascadeMethodConfig(ConfigModel):
    type: Literal["cascade"]
    classifier: ClassifierConfig
    escalation: EscalationConfig
    easy: str
    hard: str
    route_memory: RouteMemoryConfig = Field(default_factory=RouteMemoryConfig)


class FixedMethodConfig(ConfigModel):
    type: Literal["fixed"]
    role: Literal["generator", "writer", "fixer"]
    profile: str
    prompt_prefix: PromptConfig | None = None
    prompt_suffix: PromptConfig | None = None


MethodConfig = Annotated[
    DifficultyMethodConfig | CascadeMethodConfig | FixedMethodConfig,
    Field(discriminator="type"),
]


class RunLimitsConfig(ConfigModel):
    max_model_calls: int = Field(default=4, gt=0)
    max_buffered_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    deadline_seconds: float = Field(default=600.0, gt=0)


class StrategyConfig(ConfigModel):
    """One reproducible method over deployment-approved model targets."""

    schema_version: Literal[3]
    name: str
    profiles: dict[str, ModelProfileConfig]
    method: MethodConfig
    limits: RunLimitsConfig = Field(default_factory=RunLimitsConfig)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("name must be non-empty and trimmed")
        return value

    @field_validator("profiles")
    @classmethod
    def profiles_must_be_named(cls, value: dict[str, ModelProfileConfig]):
        if not value:
            raise ValueError("profiles must not be empty")
        if any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            for name in value
        ):
            raise ValueError("profile names must be non-empty and trimmed")
        return value

    @property
    def model_profiles(self) -> tuple[ModelProfileConfig, ...]:
        return tuple(self.profiles.values())

    @property
    def target_ids(self) -> frozenset[str]:
        values = {profile.target for profile in self.profiles.values()}
        classifier = getattr(self.method, "classifier", None)
        if classifier is not None:
            values.add(classifier.target)
        return frozenset(values)

    @model_validator(mode="after")
    def validate_profile_references(self) -> "StrategyConfig":
        if isinstance(self.method, FixedMethodConfig):
            references = (self.method.profile,)
        else:
            references = (self.method.easy, self.method.hard)
        missing = sorted(set(references) - set(self.profiles))
        if missing:
            raise ValueError(
                "method references undefined profiles: " + ", ".join(missing)
            )
        if (
            isinstance(self.method, CascadeMethodConfig)
            and self.method.escalation.tool_output == "accept-and-pin"
            and not self.method.route_memory.enabled
        ):
            raise ValueError(
                "cascade accept-and-pin requires route_memory.enabled"
            )
        return self
