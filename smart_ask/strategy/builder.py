"""Compile validated declarative strategies into the sole async engine."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any, Literal

from ..conversation.engine import (
    ModelCallExecutor,
    RunObserver,
    StrategyEngine,
    StrategyMethod,
)
from ..executors.target_registry import TargetExecutorRegistry
from ..methods.memory import InMemoryRouteMemory, RouteMemory
from ..methods.strategies import (
    CascadeStrategyMethod,
    DifficultyStrategyMethod,
    FixedStrategyMethod,
    MarkerCandidatePolicy,
    ModelProfile,
    RequestTransform,
    StructuredDifficultyClassifier,
)
from .errors import StrategyBuildError
from .loader import LoadedStrategy
from .schema import (
    CascadeMethodConfig,
    DifficultyMethodConfig,
    FixedMethodConfig,
    LLMClassifierConfig,
    ModelParametersConfig,
    ModelProfileConfig,
)
from .targets import (
    TargetDefinition,
    TargetRegistry,
    default_target_registry,
)


class StrategyBuilder:
    """Resolve policy against trusted targets and build one StrategyEngine."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        targets: TargetRegistry | None = None,
        executor: ModelCallExecutor | None = None,
        http_client_factory=None,
    ) -> None:
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("env must be a mapping or None")
        resolved_env = dict(os.environ if env is None else env)
        if targets is None:
            targets = default_target_registry(resolved_env)
        if not isinstance(targets, TargetRegistry):
            raise TypeError("targets must be a TargetRegistry")
        if executor is not None and not callable(getattr(executor, "stream", None)):
            raise TypeError("executor must expose an async stream operation")
        self._env = resolved_env
        self._targets = targets
        if executor is None:
            options: dict[str, Any] = {"env": self._env}
            if http_client_factory is not None:
                options["http_client_factory"] = http_client_factory
            executor = TargetExecutorRegistry(targets, **options)
        self._executor = executor

    @property
    def targets(self) -> TargetRegistry:
        return self._targets

    @property
    def executor(self) -> ModelCallExecutor:
        return self._executor

    def required_secret_envs(self, loaded: LoadedStrategy) -> frozenset[str]:
        self._validate_loaded(loaded)
        return self._targets.required_secret_envs(loaded.config.target_ids)

    def target_snapshot(self, loaded: LoadedStrategy):
        self._validate_loaded(loaded)
        return self._targets.snapshot(loaded.config.target_ids)

    def deployment_manifest(self, loaded: LoadedStrategy) -> dict[str, Any]:
        """Return the secret-free physical target identity used for a run."""

        self._validate_loaded(loaded)
        target_ids = loaded.config.target_ids
        return {
            "status": "resolved",
            "digest": self._targets.digest(target_ids),
            "targets": list(self._targets.snapshot(target_ids)),
        }

    def build_method(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None = None,
        *,
        route_memory: RouteMemory | None = None,
    ) -> StrategyMethod:
        self._validate_loaded(loaded)
        if force not in (None, "easy", "hard"):
            raise StrategyBuildError("force must be 'easy', 'hard', or None")
        self._validate_targets(loaded)
        config = loaded.config
        method = config.method
        profiles = {
            name: self._profile(loaded, name, profile)
            for name, profile in config.profiles.items()
        }

        if force is not None:
            if isinstance(method, FixedMethodConfig):
                raise StrategyBuildError(
                    f"fixed strategy {config.name!r} does not support force overrides"
                )
            selected = profiles[method.easy if force == "easy" else method.hard]
            return FixedStrategyMethod(
                profile=selected,
                role="generator" if force == "easy" else "writer",
                transform=RequestTransform(),
            )

        if isinstance(method, FixedMethodConfig):
            return FixedStrategyMethod(
                profile=profiles[method.profile],
                role=method.role,
                transform=RequestTransform(
                    latest_user_prefix=(
                        loaded.resolve_prompt(method.prompt_prefix)
                        if method.prompt_prefix is not None
                        else ""
                    ),
                    latest_user_suffix=(
                        loaded.resolve_prompt(method.prompt_suffix)
                        if method.prompt_suffix is not None
                        else ""
                    ),
                ),
            )

        memory = route_memory
        memory_config = method.route_memory
        if memory is None and memory_config.enabled:
            memory = InMemoryRouteMemory(
                ttl_seconds=memory_config.ttl_seconds,
                max_entries=memory_config.max_entries,
            )
        classifier = self._classifier(loaded, method.classifier)
        easy = profiles[method.easy]
        hard = profiles[method.hard]
        if isinstance(method, DifficultyMethodConfig):
            return DifficultyStrategyMethod(
                classifier=classifier,
                easy=easy,
                hard=hard,
                route_memory=memory,
            )
        if isinstance(method, CascadeMethodConfig):
            escalation = method.escalation
            tool_policy = {
                "accept-and-pin": "accept_and_pin",
                "escalate": "escalate",
                "fail": "raise",
            }[escalation.tool_output]
            return CascadeStrategyMethod(
                classifier=classifier,
                candidate_policy=MarkerCandidatePolicy(
                    marker=escalation.marker,
                    self_check_suffix=loaded.resolve_prompt(
                        escalation.self_check_suffix
                    ),
                    escalation_prefix=loaded.resolve_prompt(
                        escalation.escalation_prefix
                    ),
                    tool_calls=tool_policy,
                ),
                easy=easy,
                hard=hard,
                route_memory=memory,
            )
        raise StrategyBuildError(f"unsupported method configuration: {method}")

    def build_engine(
        self,
        loaded: LoadedStrategy,
        force: Literal["easy", "hard"] | None = None,
        *,
        route_memory: RouteMemory | None = None,
        heartbeat_seconds: float = 15.0,
        observer: RunObserver | None = None,
    ) -> StrategyEngine:
        method = self.build_method(
            loaded,
            force,
            route_memory=route_memory,
        )
        limits = loaded.config.limits
        return StrategyEngine(
            method,
            self._executor,
            heartbeat_seconds=heartbeat_seconds,
            max_model_calls=limits.max_model_calls,
            max_buffer_bytes=limits.max_buffered_bytes,
            max_buffer_seconds=limits.deadline_seconds,
            deadline_seconds=limits.deadline_seconds,
            observer=observer,
            owns_executor=False,
        )

    async def aclose(self) -> None:
        """Close the deployment executor shared by engines from this builder."""

        closer = getattr(self._executor, "aclose", None)
        if callable(closer):
            await closer()

    def _classifier(
        self,
        loaded: LoadedStrategy,
        config: LLMClassifierConfig,
    ) -> StructuredDifficultyClassifier:
        continuation = {
            "easy": "route_easy",
            "hard": "route_hard",
            "raise": "raise",
            "classify-tool-result": "classify_tool_result",
        }[config.missing_input]
        parameters = {
            "max_tokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
        }
        if config.parameters.reasoning_effort is not None:
            parameters["reasoning_effort"] = config.parameters.reasoning_effort
        return StructuredDifficultyClassifier(
            profile=ModelProfile("classifier", config.target),
            prompt=loaded.resolve_prompt(config.prompt),
            projection=config.projection.replace("-", "_"),
            continuation=continuation,
            fallback=config.fallback,
            max_prompt_chars=(
                None
                if config.projection == "full-conversation"
                else config.max_prompt_chars
            ),
            parameters=parameters,
        )

    def _profile(
        self,
        loaded: LoadedStrategy,
        profile_id: str,
        config: ModelProfileConfig,
    ) -> ModelProfile:
        system = (
            ()
            if config.system_prompt is None
            else (loaded.resolve_prompt(config.system_prompt),)
        )
        return ModelProfile(
            profile_id,
            config.target,
            RequestTransform(
                system_suffix=system,
                parameters=self._parameters(config.parameters),
                keep_last_messages=config.context_messages,
            ),
        )

    @staticmethod
    def _parameters(config: ModelParametersConfig) -> dict[str, object]:
        values: dict[str, object] = {}
        if config.max_tokens is not None:
            values["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            values["temperature"] = config.temperature
        if config.reasoning_effort is not None:
            values["reasoning_effort"] = config.reasoning_effort
        return values

    @staticmethod
    def _validate_loaded(loaded: LoadedStrategy) -> None:
        if not isinstance(loaded, LoadedStrategy):
            raise TypeError("loaded must be a LoadedStrategy")

    def _validate_targets(self, loaded: LoadedStrategy) -> None:
        config = loaded.config
        try:
            targets = {
                target_id: self._targets.resolve(target_id)
                for target_id in config.target_ids
            }
        except KeyError as exc:
            raise StrategyBuildError(str(exc)) from exc
        for name, profile in config.profiles.items():
            target = targets[profile.target]
            self._validate_profile(name, profile, target)
        classifier = getattr(config.method, "classifier", None)
        if classifier is not None:
            target = targets[classifier.target]
            self._validate_reasoning(
                classifier.parameters.reasoning_effort,
                target,
                "classifier",
            )

    def _validate_profile(
        self,
        name: str,
        profile: ModelProfileConfig,
        target: TargetDefinition,
    ) -> None:
        maximum = profile.parameters.max_tokens
        if maximum is not None and maximum > target.limits.max_output_tokens:
            raise StrategyBuildError(
                f"profile {name!r} requests {maximum} output tokens but target "
                f"{target.target_id!r} allows {target.limits.max_output_tokens}"
            )
        self._validate_reasoning(
            profile.parameters.reasoning_effort,
            target,
            f"profile {name!r}",
        )

    @staticmethod
    def _validate_reasoning(
        effort: str | None,
        target: TargetDefinition,
        owner: str,
    ) -> None:
        if effort is None:
            return
        if "reasoning" not in target.capabilities:
            raise StrategyBuildError(
                f"{owner} requests reasoning on target {target.target_id!r}"
            )
        if target.transport == "groq" and effort not in ("low", "medium", "high"):
            raise StrategyBuildError(
                "Groq reasoning_effort must be low, medium, or high"
            )
