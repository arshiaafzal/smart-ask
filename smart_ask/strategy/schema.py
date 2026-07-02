"""Strict typed schema for complete smart-ask strategy configurations."""

from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import OR_BASE


class ConfigModel(BaseModel):
    """Base for immutable strategy configuration values."""

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
    def path_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt path must not be blank")
        return value


PromptConfig = Annotated[
    InlinePromptConfig | FilePromptConfig,
    Field(discriminator="type"),
]


class ModelParametersConfig(ConfigModel):
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ModelProfileConfig(ConfigModel):
    model: str
    system_prompt: PromptConfig | None = None
    parameters: ModelParametersConfig = Field(default_factory=ModelParametersConfig)

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value


class OpenRouterExecutorConfig(ConfigModel):
    type: Literal["openrouter"]
    base_url: str = OR_BASE
    api_key_env: str = "OPENROUTER_API_KEY"
    defaults: ModelParametersConfig = Field(
        default_factory=lambda: ModelParametersConfig(
            max_tokens=1024,
            temperature=0.0,
        )
    )

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return value

    @field_validator("api_key_env")
    @classmethod
    def env_name_must_be_valid(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must be a valid environment-variable name")
        return value


class HermesExecutorConfig(ConfigModel):
    type: Literal["hermes"]
    command: str = "hermes"
    provider: str = "openrouter"

    @field_validator("command", "provider")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


ExecutorConfig = Annotated[
    OpenRouterExecutorConfig | HermesExecutorConfig,
    Field(discriminator="type"),
]


class LLMClassifierConfig(ConfigModel):
    type: Literal["llm"]
    model: str
    executor: ExecutorConfig
    prompt: PromptConfig
    max_prompt_chars: int = Field(default=1200, gt=0)
    parameters: ModelParametersConfig = Field(
        default_factory=lambda: ModelParametersConfig(
            max_tokens=20,
            temperature=0.0,
        )
    )

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @model_validator(mode="after")
    def executor_must_capture_output(self) -> "LLMClassifierConfig":
        if isinstance(self.executor, HermesExecutorConfig):
            raise ValueError("an LLM classifier requires a capturing executor")
        return self


ClassifierConfig = Annotated[LLMClassifierConfig, Field(discriminator="type")]


class MarkerEscalationConfig(ConfigModel):
    type: Literal["marker"]
    marker: str = "ESCALATE_NOW"
    self_check_suffix: PromptConfig
    escalation_prefix: PromptConfig

    @field_validator("marker")
    @classmethod
    def marker_must_be_one_line(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or "\n" in value or "\r" in value:
            raise ValueError("marker must be a nonempty, trimmed, single-line value")
        return value


EscalationConfig = Annotated[
    MarkerEscalationConfig,
    Field(discriminator="type"),
]


class DifficultyMethodConfig(ConfigModel):
    type: Literal["difficulty"]
    classifier: ClassifierConfig
    easy: ModelProfileConfig
    hard: ModelProfileConfig


class CascadeMethodConfig(ConfigModel):
    type: Literal["cascade"]
    classifier: ClassifierConfig
    escalation: EscalationConfig
    easy: ModelProfileConfig
    hard: ModelProfileConfig


class FixedMethodConfig(ConfigModel):
    type: Literal["fixed"]
    decision: Literal["easy", "hard"]
    role: Literal["generator", "writer", "fixer"] | None = None
    model: ModelProfileConfig


MethodConfig = Annotated[
    DifficultyMethodConfig | CascadeMethodConfig | FixedMethodConfig,
    Field(discriminator="type"),
]


class StrategyConfig(ConfigModel):
    """One complete, reproducible smart-ask strategy."""

    schema_version: Literal[1]
    name: str
    method: MethodConfig
    generation: ExecutorConfig
    max_attempts: int | None = Field(default=None, gt=0)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value

    @property
    def resolved_max_attempts(self) -> int:
        if self.max_attempts is not None:
            return self.max_attempts
        return 2 if isinstance(self.method, CascadeMethodConfig) else 1

    @property
    def model_profiles(self) -> tuple[ModelProfileConfig, ...]:
        if isinstance(self.method, FixedMethodConfig):
            return (self.method.model,)
        return (self.method.easy, self.method.hard)

    @model_validator(mode="after")
    def validate_composition(self) -> "StrategyConfig":
        if isinstance(self.method, CascadeMethodConfig):
            if isinstance(self.generation, HermesExecutorConfig):
                raise ValueError("cascade requires a generation executor that captures output")
            if self.max_attempts is not None and self.max_attempts < 2:
                raise ValueError("cascade requires max_attempts >= 2")

        if isinstance(self.generation, HermesExecutorConfig):
            for profile in self.model_profiles:
                if profile.system_prompt is not None:
                    raise ValueError("Hermes generation does not support system prompts")
                if (
                    profile.parameters.max_tokens is not None
                    or profile.parameters.temperature is not None
                ):
                    raise ValueError("Hermes generation does not support model tuning parameters")

        profiles = self.model_profiles
        if len(profiles) == 2 and profiles[0].model == profiles[1].model:
            if profiles[0] != profiles[1]:
                raise ValueError(
                    "profiles sharing a model ID must have identical prompts and parameters"
                )
        return self
