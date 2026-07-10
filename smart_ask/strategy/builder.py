"""Build runtime applications from validated strategy configurations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import subprocess
from types import MappingProxyType
from typing import Any, Literal

import httpx

from ..application import SmartAsk
from ..conversation.metrics import ConversationMetricsStore
from ..conversation.runtime import ConversationRuntime
from ..executors import (
    HermesExecutor,
    ModelExecutor,
    OllamaExecutor,
    OllamaConversationExecutor,
    OpenAIExecutor,
    OpenAIConversationExecutor,
    OpenRouterExecutor,
    OpenRouterConversationExecutor,
)
from ..metrics import DEFAULT_PRICE_CATALOG, StatsCollector
from ..methods import (
    CascadeRoutingMethod,
    DifficultyRoutingMethod,
    FixedRoutingMethod,
    LLMDifficultyClassifier,
    MarkerEscalationPolicy,
)
from ..routing import SmartRouter
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
    OllamaExecutorConfig,
    OpenAIConnectionConfig,
    OpenAIExecutorConfig,
    OpenRouterConnectionConfig,
    OpenRouterExecutorConfig,
)

OpenRouterClientFactory = Callable[[str, str], Any]
OpenRouterConversationClientFactory = Callable[[str, str], httpx.AsyncClient]
OpenAIClientFactory = Callable[[str, str], Any]
OpenAIConversationClientFactory = Callable[[str, str], httpx.AsyncClient]


class StrategyBuilder:
    """Construct SmartAsk applications while keeping runtime dependencies injectable."""

    __slots__ = (
        "_env",
        "_openrouter_client_factory",
        "_openrouter_conversation_client_factory",
        "_openai_client_factory",
        "_openai_conversation_client_factory",
        "_hermes_runner",
        "_stats_collector",
        "_clients",
        "_conversation_clients",
    )

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        openrouter_client_factory: OpenRouterClientFactory | None = None,
        openrouter_conversation_client_factory: (
            OpenRouterConversationClientFactory | None
        ) = None,
        openai_client_factory: OpenAIClientFactory | None = None,
        openai_conversation_client_factory: (
            OpenAIConversationClientFactory | None
        ) = None,
        hermes_runner: Callable[..., Any] | None = None,
        stats_collector: StatsCollector | None = None,
    ):
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("env must be a mapping or None")
        if openrouter_client_factory is not None and not callable(
            openrouter_client_factory
        ):
            raise TypeError("openrouter_client_factory must be callable or None")
        if (
            openrouter_conversation_client_factory is not None
            and not callable(openrouter_conversation_client_factory)
        ):
            raise TypeError(
                "openrouter_conversation_client_factory must be callable or None"
            )
        if openai_client_factory is not None and not callable(openai_client_factory):
            raise TypeError("openai_client_factory must be callable or None")
        if (
            openai_conversation_client_factory is not None
            and not callable(openai_conversation_client_factory)
        ):
            raise TypeError(
                "openai_conversation_client_factory must be callable or None"
            )
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
        self._openrouter_conversation_client_factory = (
            self._default_openrouter_conversation_client
            if openrouter_conversation_client_factory is None
            else openrouter_conversation_client_factory
        )
        self._openai_client_factory = (
            self._default_openai_client
            if openai_client_factory is None
            else openai_client_factory
        )
        self._openai_conversation_client_factory = (
            self._default_openai_conversation_client
            if openai_conversation_client_factory is None
            else openai_conversation_client_factory
        )
        self._hermes_runner = (
            subprocess.run if hermes_runner is None else hermes_runner
        )
        self._stats_collector = (
            stats_collector
            if stats_collector is not None
            else StatsCollector(price_catalog=DEFAULT_PRICE_CATALOG)
        )
        self._clients: dict[tuple[str, ...], Any] = {}
        self._conversation_clients: dict[tuple[str, ...], httpx.AsyncClient] = {}

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

    @staticmethod
    def _default_openrouter_conversation_client(
        base_url: str,
        api_key: str,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300.0),
        )

    @staticmethod
    def _default_openai_client(base_url: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise StrategyBuildError(
                "the openai package is required for OpenAI strategies"
            ) from exc
        return OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _default_openai_conversation_client(
        base_url: str,
        api_key: str,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300.0),
        )

    def _openrouter_client(
        self,
        config: OpenRouterConnectionConfig,
    ):
        api_key = self._env.get(config.api_key_env, "")
        if not api_key.strip():
            raise StrategyBuildError(
                f"required environment variable {config.api_key_env} is not set"
            )
        key = ("openrouter", config.base_url, api_key)
        if key not in self._clients:
            self._clients[key] = self._openrouter_client_factory(
                config.base_url,
                api_key,
            )
        return self._clients[key]

    def _openrouter_conversation_client(
        self,
        config: OpenRouterConnectionConfig,
    ) -> httpx.AsyncClient:
        api_key = self._env.get(config.api_key_env, "")
        if not api_key.strip():
            raise StrategyBuildError(
                f"required environment variable {config.api_key_env} is not set"
            )
        key = ("openrouter", config.base_url, api_key)
        if key not in self._conversation_clients:
            client = self._openrouter_conversation_client_factory(
                config.base_url,
                api_key,
            )
            if not isinstance(client, httpx.AsyncClient):
                raise TypeError(
                    "openrouter conversation client factory must return "
                    "httpx.AsyncClient"
                )
            self._conversation_clients[key] = client
        return self._conversation_clients[key]

    def _openai_client(self, config: OpenAIConnectionConfig):
        api_key = self._env.get(config.api_key_env, "")
        if not api_key.strip():
            raise StrategyBuildError(
                f"required environment variable {config.api_key_env} is not set"
            )
        key = ("openai", config.base_url, api_key)
        if key not in self._clients:
            self._clients[key] = self._openai_client_factory(
                config.base_url,
                api_key,
            )
        return self._clients[key]

    def _openai_conversation_client(
        self,
        config: OpenAIConnectionConfig,
    ) -> httpx.AsyncClient:
        api_key = self._env.get(config.api_key_env, "")
        if not api_key.strip():
            raise StrategyBuildError(
                f"required environment variable {config.api_key_env} is not set"
            )
        key = ("openai", config.base_url, api_key)
        if key not in self._conversation_clients:
            client = self._openai_conversation_client_factory(
                config.base_url,
                api_key,
            )
            if not isinstance(client, httpx.AsyncClient):
                raise TypeError(
                    "openai conversation client factory must return httpx.AsyncClient"
                )
            self._conversation_clients[key] = client
        return self._conversation_clients[key]

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

        if isinstance(config, OllamaExecutorConfig):
            return OllamaExecutor(
                base_url=config.base_url,
                default_max_tokens=config.defaults.max_tokens,
                temperature=config.defaults.temperature,
                think=config.think,
                timeout_seconds=config.timeout_seconds,
            )

        if isinstance(config, OpenRouterExecutorConfig):
            return OpenRouterExecutor(
                self._openrouter_client(config),
                default_max_tokens=config.defaults.max_tokens,
                temperature=config.defaults.temperature,
            )
        if isinstance(config, OpenAIExecutorConfig):
            return OpenAIExecutor(
                self._openai_client(config),
                default_max_tokens=config.defaults.max_tokens,
                reasoning_effort=config.defaults.reasoning_effort,
            )
        raise StrategyBuildError(f"unsupported executor configuration: {config}")

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
        reasoning_efforts = {}
        for profile in profiles:
            if profile.system_prompt is not None:
                system_prompts[profile.model] = loaded.resolve_prompt(profile.system_prompt)
            if profile.parameters.max_tokens is not None:
                max_tokens[profile.model] = profile.parameters.max_tokens
            if profile.parameters.temperature is not None:
                temperatures[profile.model] = profile.parameters.temperature
            if profile.parameters.reasoning_effort is not None:
                reasoning_efforts[profile.model] = (
                    profile.parameters.reasoning_effort
                )
        if isinstance(config, OllamaExecutorConfig):
            return OllamaExecutor(
                base_url=config.base_url,
                system_prompts=system_prompts,
                max_tokens=max_tokens,
                temperatures=temperatures,
                default_max_tokens=config.defaults.max_tokens,
                temperature=config.defaults.temperature,
                think=config.think,
                timeout_seconds=config.timeout_seconds,
            )
        if isinstance(config, OpenRouterExecutorConfig):
            return OpenRouterExecutor(
                self._openrouter_client(config),
                system_prompts=system_prompts,
                max_tokens=max_tokens,
                temperatures=temperatures,
                default_max_tokens=config.defaults.max_tokens,
                temperature=config.defaults.temperature,
            )
        if isinstance(config, OpenAIExecutorConfig):
            return OpenAIExecutor(
                self._openai_client(config),
                system_prompts=system_prompts,
                max_tokens=max_tokens,
                reasoning_efforts=reasoning_efforts,
                default_max_tokens=config.defaults.max_tokens,
                reasoning_effort=config.defaults.reasoning_effort,
            )
        raise StrategyBuildError(f"unsupported generation configuration: {config}")

    def _build_classifier(
        self,
        loaded: LoadedStrategy,
        config: LLMClassifierConfig,
    ) -> LLMDifficultyClassifier:
        if isinstance(config.executor, OpenRouterConnectionConfig):
            executor = OpenRouterExecutor(
                self._openrouter_client(config.executor),
                default_max_tokens=config.parameters.max_tokens,
                temperature=config.parameters.temperature,
            )
        elif isinstance(config.executor, OpenAIConnectionConfig):
            executor = OpenAIExecutor(
                self._openai_client(config.executor),
                default_max_tokens=config.parameters.max_tokens,
                reasoning_effort=(
                    config.parameters.reasoning_effort or "low"
                ),
            )
        else:
            raise StrategyBuildError(
                f"unsupported classifier executor: {config.executor}"
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

        router, profiles = self._build_router_and_profiles(loaded, force)
        executor = self._build_generation_executor(loaded, profiles)
        return SmartAsk.from_router(router, executor)

    def build_router(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None = None,
    ) -> SmartRouter:
        """Build only the strategy's routing policy and collaborators."""

        router, _profiles = self._build_router_and_profiles(loaded, force)
        return router

    def build_conversation_runtime(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None = None,
        *,
        metrics: ConversationMetricsStore | None = None,
    ) -> ConversationRuntime:
        """Build the complete harness-neutral structured conversation runtime."""

        router, _profiles = self._build_router_and_profiles(loaded, force)
        generation = loaded.config.generation
        if isinstance(generation, OllamaExecutorConfig):
            executor = OllamaConversationExecutor(
                base_url=generation.base_url,
                default_max_tokens=generation.defaults.max_tokens,
                temperature=generation.defaults.temperature,
                think=generation.think,
                timeout_seconds=generation.timeout_seconds,
            )
        elif isinstance(generation, OpenRouterExecutorConfig):
            executor = OpenRouterConversationExecutor(
                self._openrouter_conversation_client(generation),
                default_max_tokens=generation.defaults.max_tokens,
                temperature=generation.defaults.temperature,
            )
        elif isinstance(generation, OpenAIExecutorConfig):
            executor = OpenAIConversationExecutor(
                self._openai_conversation_client(generation),
                default_max_tokens=generation.defaults.max_tokens,
                reasoning_effort=generation.defaults.reasoning_effort,
            )
        else:
            raise StrategyBuildError(
                f"generation transport {generation.type!r} does not yet expose "
                "structured conversation execution"
            )
        return ConversationRuntime(
            loaded_strategy=loaded,
            router=router,
            executor=executor,
            metrics=metrics,
        )

    def _build_router_and_profiles(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None,
    ) -> tuple[SmartRouter, tuple[ModelProfileConfig, ...]]:

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
            profiles = (profile,)
            router = SmartRouter(
                method,
                max_attempts=1,
                strategy_id=config.name,
                stats_collector=self._stats_collector,
            )
            return router, profiles

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

        router = SmartRouter(
            method,
            max_attempts=2 if isinstance(method_config, CascadeMethodConfig) else 1,
            strategy_id=config.name,
            stats_collector=self._stats_collector,
        )
        return router, profiles
