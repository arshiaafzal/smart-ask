"""Strict typed schema for complete smart-ask strategy configurations."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


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


class OpenRouterDefaultsConfig(ConfigModel):
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ClassifierParametersConfig(ConfigModel):
    max_tokens: int = Field(default=20, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None


class ModelProfileConfig(ConfigModel):
    model: str
    system_prompt: PromptConfig | None = None
    parameters: ModelParametersConfig = Field(default_factory=ModelParametersConfig)

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("model must be non-empty and trimmed")
        return value


class OpenRouterConnectionConfig(ConfigModel):
    """Connection settings shared by OpenRouter call sites."""

    type: Literal["openrouter"]
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("base_url must be non-empty and trimmed")
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


class OpenRouterExecutorConfig(OpenRouterConnectionConfig):
    """OpenRouter generation settings, including request fallbacks."""

    defaults: OpenRouterDefaultsConfig = Field(
        default_factory=OpenRouterDefaultsConfig
    )


class OpenAIConnectionConfig(ConfigModel):
    """Connection settings for the first-party OpenAI API."""

    type: Literal["openai"]
    base_url: str = DEFAULT_OPENAI_BASE_URL
    api_key_env: str = "OPENAI_API_KEY"

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("base_url must be non-empty and trimmed")
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


class OpenAIDefaultsConfig(ConfigModel):
    max_tokens: int = Field(default=8192, gt=0)
    reasoning_effort: ReasoningEffort = "medium"


class OpenAIExecutorConfig(OpenAIConnectionConfig):
    """First-party OpenAI generation settings."""

    defaults: OpenAIDefaultsConfig = Field(default_factory=OpenAIDefaultsConfig)


class OllamaExecutorConfig(ConfigModel):
    """Native local Ollama chat transport with no provider credential."""

    type: Literal["ollama"]
    base_url: str = "http://127.0.0.1:11434/api"
    think: bool = False
    timeout_seconds: float = Field(default=300.0, gt=0)
    defaults: OpenRouterDefaultsConfig = Field(
        default_factory=OpenRouterDefaultsConfig
    )

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("base_url must be non-empty and trimmed")
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return value


class HermesExecutorConfig(ConfigModel):
    type: Literal["hermes"]
    command: str = "hermes"
    provider: str = "openrouter"

    @field_validator("command", "provider")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("value must be non-empty and trimmed")
        return value


ExecutorConfig = Annotated[
    OpenRouterExecutorConfig
    | OpenAIExecutorConfig
    | OllamaExecutorConfig
    | HermesExecutorConfig,
    Field(discriminator="type"),
]

ModelConnectionConfig = Annotated[
    OpenRouterConnectionConfig | OpenAIConnectionConfig,
    Field(discriminator="type"),
]


class LLMClassifierConfig(ConfigModel):
    type: Literal["llm"]
    model: str
    executor: ModelConnectionConfig
    prompt: PromptConfig
    fallback: Literal["easy", "hard", "raise"]
    max_prompt_chars: int = Field(default=1200, gt=0)
    parameters: ClassifierParametersConfig = Field(
        default_factory=ClassifierParametersConfig
    )

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("model must be non-empty and trimmed")
        return value

ClassifierConfig = Annotated[LLMClassifierConfig, Field(discriminator="type")]


class MarkerEscalationConfig(ConfigModel):
    type: Literal["marker"]
    marker: str
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
    role: Literal["generator", "writer", "fixer"]
    model: ModelProfileConfig
    prompt_prefix: PromptConfig | None = None
    prompt_suffix: PromptConfig | None = None


MethodConfig = Annotated[
    DifficultyMethodConfig | CascadeMethodConfig | FixedMethodConfig,
    Field(discriminator="type"),
]


class StrategyConfig(ConfigModel):
    """One complete, reproducible smart-ask strategy."""

    schema_version: Literal[2]
    name: str
    method: MethodConfig
    generation: ExecutorConfig

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("name must be non-empty and trimmed")
        return value

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

        if isinstance(self.generation, HermesExecutorConfig):
            for profile in self.model_profiles:
                if profile.system_prompt is not None:
                    raise ValueError("Hermes generation does not support system prompts")
                if (
                    profile.parameters.max_tokens is not None
                    or profile.parameters.temperature is not None
                    or profile.parameters.reasoning_effort is not None
                ):
                    raise ValueError("Hermes generation does not support model tuning parameters")

        if not isinstance(self.generation, OpenAIExecutorConfig):
            for profile in self.model_profiles:
                if profile.parameters.reasoning_effort is not None:
                    raise ValueError(
                        "reasoning_effort requires an OpenAI generation executor"
                    )

        classifier = getattr(self.method, "classifier", None)
        if (
            classifier is not None
            and classifier.parameters.reasoning_effort is not None
            and not isinstance(classifier.executor, OpenAIConnectionConfig)
        ):
            raise ValueError(
                "classifier reasoning_effort requires an OpenAI executor"
            )

        profiles = self.model_profiles
        if len(profiles) == 2 and profiles[0].model == profiles[1].model:
            if profiles[0] != profiles[1]:
                raise ValueError(
                    "profiles sharing a model ID must have identical prompts and parameters"
                )
        return self
