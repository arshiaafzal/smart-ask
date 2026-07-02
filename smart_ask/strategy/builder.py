"""Build runtime applications from validated strategy configurations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import subprocess
from typing import Any, Literal

from ..application import SmartAsk
from ..executors import HermesExecutor, ModelExecutor, OpenRouterExecutor
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
    ModelProfileConfig,
    OpenRouterExecutorConfig,
)


ExecutorWrapper = Callable[[ModelExecutor, str], ModelExecutor]
OpenRouterClientFactory = Callable[[str, str], Any]


class StrategyBuilder:
    """Construct SmartAsk applications while keeping runtime dependencies injectable."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        openrouter_client_factory: OpenRouterClientFactory | None = None,
        hermes_runner: Callable[..., Any] | None = None,
        executor_wrapper: ExecutorWrapper | None = None,
    ):
        self.env = dict(os.environ if env is None else env)
        self.openrouter_client_factory = (
            openrouter_client_factory or self._default_openrouter_client
        )
        self.hermes_runner = hermes_runner or subprocess.run
        self.executor_wrapper = executor_wrapper
        self._clients: dict[tuple[str, str], Any] = {}

    @staticmethod
    def _default_openrouter_client(base_url: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise StrategyBuildError(
                "the openai package is required for OpenRouter strategies"
            ) from exc
        return OpenAI(base_url=base_url, api_key=api_key)

    def _openrouter_client(self, config: OpenRouterExecutorConfig):
        api_key = self.env.get(config.api_key_env, "")
        if not api_key:
            raise StrategyBuildError(
                f"required environment variable {config.api_key_env} is not set"
            )
        key = (config.base_url, config.api_key_env)
        if key not in self._clients:
            self._clients[key] = self.openrouter_client_factory(config.base_url, api_key)
        return self._clients[key]

    def _wrap(self, executor: ModelExecutor, role: str) -> ModelExecutor:
        if self.executor_wrapper is None:
            return executor
        return self.executor_wrapper(executor, role)

    def _build_executor(
        self,
        config: ExecutorConfig,
        profiles: tuple[ModelProfileConfig, ...] = (),
        *,
        role: str,
    ) -> ModelExecutor:
        if isinstance(config, HermesExecutorConfig):
            executor: ModelExecutor = HermesExecutor(
                provider=config.provider,
                command=config.command,
                runner=self.hermes_runner,
            )
            return self._wrap(executor, role)

        system_prompts = {}
        max_tokens = {}
        temperatures = {}
        for profile in profiles:
            if profile.system_prompt is not None:
                raise AssertionError("prompt resolution requires the loaded strategy")
            if profile.parameters.max_tokens is not None:
                max_tokens[profile.model] = profile.parameters.max_tokens
            if profile.parameters.temperature is not None:
                temperatures[profile.model] = profile.parameters.temperature
        executor = OpenRouterExecutor(
            self._openrouter_client(config),
            system_prompts=system_prompts,
            max_tokens=max_tokens,
            temperatures=temperatures,
            default_max_tokens=config.defaults.max_tokens or 1024,
            temperature=config.defaults.temperature or 0.0,
        )
        return self._wrap(executor, role)

    def _build_generation_executor(
        self,
        loaded: LoadedStrategy,
        profiles: tuple[ModelProfileConfig, ...],
    ) -> ModelExecutor:
        config = loaded.config.generation
        if isinstance(config, HermesExecutorConfig):
            return self._build_executor(config, role="generation")

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
        executor: ModelExecutor = OpenRouterExecutor(
            self._openrouter_client(config),
            system_prompts=system_prompts,
            max_tokens=max_tokens,
            temperatures=temperatures,
            default_max_tokens=config.defaults.max_tokens or 1024,
            temperature=(
                config.defaults.temperature
                if config.defaults.temperature is not None
                else 0.0
            ),
        )
        return self._wrap(executor, "generation")

    def _build_classifier(self, loaded: LoadedStrategy, config):
        executor = self._build_executor(config.executor, role="classifier")
        return LLMDifficultyClassifier(
            executor,
            model=config.model,
            prompt_prefix=loaded.resolve_prompt(config.prompt),
            max_prompt_chars=config.max_prompt_chars,
            max_tokens=config.parameters.max_tokens or 20,
            temperature=(
                config.parameters.temperature
                if config.parameters.temperature is not None
                else 0.0
            ),
        )

    def build(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None = None,
    ) -> SmartAsk:
        """Build the configured application, optionally forcing an easy/hard profile."""

        config = loaded.config
        method_config = config.method

        if force is not None:
            if isinstance(method_config, FixedMethodConfig):
                if force != method_config.decision:
                    raise StrategyBuildError(
                        f"strategy {config.name!r} has no {force} model profile"
                    )
                profile = method_config.model
            else:
                profile = method_config.easy if force == "easy" else method_config.hard
            method = FixedRoutingMethod(
                profile.model,
                force,
                role="generator" if force == "easy" else "writer",
            )
            executor = self._build_generation_executor(loaded, (profile,))
            return SmartAsk(method, executor, max_attempts=1)

        if isinstance(method_config, FixedMethodConfig):
            role = method_config.role or (
                "generator" if method_config.decision == "easy" else "writer"
            )
            method = FixedRoutingMethod(
                method_config.model.model,
                method_config.decision,
                role=role,
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
        return SmartAsk(method, executor, max_attempts=config.resolved_max_attempts)
