"""Build runtime applications from validated strategy configurations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import subprocess
from types import MappingProxyType
from typing import Any, Literal

from ..application import SmartAsk
from ..executors import HermesExecutor, ModelExecutor, OpenRouterExecutor
from ..metrics import DEFAULT_PRICE_CATALOG, StatsCollector
from ..methods import (
    CascadeRoutingMethod,
    DifficultyRoutingMethod,
    FixedRoutingMethod,
    LLMDifficultyClassifier,
    MarkerEscalationPolicy,
)
from .errors import StrategyBuildError
from .loader import LoadedStrategy
from .schema import (
    CascadeMethodConfig,
    DifficultyMethodConfig,
    ExecutorConfig,
    FixedMethodConfig,
    HermesExecutorConfig,
    LLMClassifierConfig,
    ModelProfileConfig,
    OpenRouterConnectionConfig,
    OpenRouterExecutorConfig,
)

OpenRouterClientFactory = Callable[[str, str], Any]


class StrategyBuilder:
    """Construct SmartAsk applications while keeping runtime dependencies injectable."""

    __slots__ = (
        "_env",
        "_openrouter_client_factory",
        "_hermes_runner",
        "_stats_collector",
        "_clients",
    )

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        openrouter_client_factory: OpenRouterClientFactory | None = None,
        hermes_runner: Callable[..., Any] | None = None,
        stats_collector: StatsCollector | None = None,
    ):
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("env must be a mapping or None")
        if openrouter_client_factory is not None and not callable(
            openrouter_client_factory
        ):
            raise TypeError("openrouter_client_factory must be callable or None")
        if hermes_runner is not None and not callable(hermes_runner):
            raise TypeError("hermes_runner must be callable or None")
        if stats_collector is not None and not isinstance(
            stats_collector,
            StatsCollector,
        ):
            raise TypeError("stats_collector must be a StatsCollector or None")
        resolved_env = dict(os.environ if env is None else env)
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in resolved_env.items()
        ):
            raise TypeError("env keys and values must be strings")
        self._env = MappingProxyType(resolved_env)
        self._openrouter_client_factory = (
            self._default_openrouter_client
            if openrouter_client_factory is None
            else openrouter_client_factory
        )
        self._hermes_runner = (
            subprocess.run if hermes_runner is None else hermes_runner
        )
        self._stats_collector = (
            stats_collector
            if stats_collector is not None
            else StatsCollector(price_catalog=DEFAULT_PRICE_CATALOG)
        )
        self._clients: dict[tuple[str, str], Any] = {}

    @property
    def stats_collector(self) -> StatsCollector:
        return self._stats_collector

    @staticmethod
    def _default_openrouter_client(base_url: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise StrategyBuildError(
                "the openai package is required for OpenRouter strategies"
            ) from exc
        return OpenAI(base_url=base_url, api_key=api_key)

    def _openrouter_client(
        self,
        config: OpenRouterConnectionConfig,
    ):
        api_key = self._env.get(config.api_key_env, "")
        if not api_key.strip():
            raise StrategyBuildError(
                f"required environment variable {config.api_key_env} is not set"
            )
        key = (config.base_url, api_key)
        if key not in self._clients:
            self._clients[key] = self._openrouter_client_factory(
                config.base_url,
                api_key,
            )
        return self._clients[key]

    def _build_executor(
        self,
        config: ExecutorConfig,
    ) -> ModelExecutor:
        if isinstance(config, HermesExecutorConfig):
            return HermesExecutor(
                provider=config.provider,
                command=config.command,
                runner=self._hermes_runner,
            )

        return OpenRouterExecutor(
            self._openrouter_client(config),
            default_max_tokens=config.defaults.max_tokens,
            temperature=config.defaults.temperature,
        )

    def _build_generation_executor(
        self,
        loaded: LoadedStrategy,
        profiles: tuple[ModelProfileConfig, ...],
    ) -> ModelExecutor:
        config = loaded.config.generation
        if isinstance(config, HermesExecutorConfig):
            return self._build_executor(config)

        system_prompts = {}
        max_tokens = {}
        temperatures = {}
        for profile in profiles:
            if profile.system_prompt is not None:
                system_prompts[profile.model] = loaded.resolve_prompt(profile.system_prompt)
            if profile.parameters.max_tokens is not None:
                max_tokens[profile.model] = profile.parameters.max_tokens
            if profile.parameters.temperature is not None:
                temperatures[profile.model] = profile.parameters.temperature
        return OpenRouterExecutor(
            self._openrouter_client(config),
            system_prompts=system_prompts,
            max_tokens=max_tokens,
            temperatures=temperatures,
            default_max_tokens=config.defaults.max_tokens,
            temperature=config.defaults.temperature,
        )

    def _build_classifier(
        self,
        loaded: LoadedStrategy,
        config: LLMClassifierConfig,
    ) -> LLMDifficultyClassifier:
        executor = OpenRouterExecutor(
            self._openrouter_client(config.executor),
            default_max_tokens=config.parameters.max_tokens,
            temperature=config.parameters.temperature,
        )
        return LLMDifficultyClassifier(
            executor,
            stats_collector=self._stats_collector,
            model=config.model,
            prompt_prefix=loaded.resolve_prompt(config.prompt),
            fallback=config.fallback,
            max_prompt_chars=config.max_prompt_chars,
            max_tokens=config.parameters.max_tokens,
            temperature=config.parameters.temperature,
        )

    def build(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None = None,
    ) -> SmartAsk:
        """Build the configured application, optionally forcing an easy/hard profile."""

        if not isinstance(loaded, LoadedStrategy):
            raise TypeError("loaded must be a LoadedStrategy")
        if force not in (None, "easy", "hard"):
            raise StrategyBuildError("force must be 'easy', 'hard', or None")

        config = loaded.config
        method_config = config.method

        if force is not None:
            if isinstance(method_config, FixedMethodConfig):
                raise StrategyBuildError(
                    f"fixed strategy {config.name!r} does not support force overrides"
                )
            profile = method_config.easy if force == "easy" else method_config.hard
            method = FixedRoutingMethod(
                profile.model,
                "generator" if force == "easy" else "writer",
                label=f"forced-{force}",
            )
            executor = self._build_generation_executor(loaded, (profile,))
            return SmartAsk(
                method,
                executor,
                max_attempts=1,
                strategy_id=config.name,
                stats_collector=self._stats_collector,
            )

        if isinstance(method_config, FixedMethodConfig):
            method = FixedRoutingMethod(
                method_config.model.model,
                role=method_config.role,
                prompt_prefix=(
                    loaded.resolve_prompt(method_config.prompt_prefix)
                    if method_config.prompt_prefix is not None
                    else ""
                ),
                prompt_suffix=(
                    loaded.resolve_prompt(method_config.prompt_suffix)
                    if method_config.prompt_suffix is not None
                    else ""
                ),
            )
            profiles = (method_config.model,)
        elif isinstance(method_config, DifficultyMethodConfig):
            classifier = self._build_classifier(loaded, method_config.classifier)
            method = DifficultyRoutingMethod(
                classifier,
                easy_model=method_config.easy.model,
                hard_model=method_config.hard.model,
            )
            profiles = (method_config.easy, method_config.hard)
        elif isinstance(method_config, CascadeMethodConfig):
            classifier = self._build_classifier(loaded, method_config.classifier)
            escalation = MarkerEscalationPolicy(
                marker=method_config.escalation.marker,
                self_check_suffix=loaded.resolve_prompt(
                    method_config.escalation.self_check_suffix
                ),
                escalation_prefix=loaded.resolve_prompt(
                    method_config.escalation.escalation_prefix
                ),
            )
            method = CascadeRoutingMethod(
                classifier,
                escalation,
                easy_model=method_config.easy.model,
                hard_model=method_config.hard.model,
            )
            profiles = (method_config.easy, method_config.hard)
        else:
            raise StrategyBuildError(f"unsupported method configuration: {method_config}")

        executor = self._build_generation_executor(loaded, profiles)
        return SmartAsk(
            method,
            executor,
            max_attempts=2 if isinstance(method_config, CascadeMethodConfig) else 1,
            strategy_id=config.name,
            stats_collector=self._stats_collector,
        )
