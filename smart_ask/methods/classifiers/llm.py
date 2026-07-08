"""Model-backed difficulty classifier independent of provider transport."""

from __future__ import annotations

import json
from numbers import Integral, Real
from typing import cast, Literal

from ..._numeric import is_finite_real
from ...domain import ExecutionRequest, Task
from ...executors.base import ModelExecutor
from ...metrics import StatsCollector
from .base import Difficulty, DifficultyClassification


ClassifierFallback = Literal["easy", "hard", "raise"]


class LLMDifficultyClassifier:
    """Classify task difficulty by prompting a model through an executor."""

    def __init__(
        self,
        executor: ModelExecutor,
        *,
        stats_collector: StatsCollector,
        model: str,
        prompt_prefix: str,
        fallback: ClassifierFallback,
        max_prompt_chars: int,
        max_tokens: int,
        temperature: float,
    ):
        if not isinstance(stats_collector, StatsCollector):
            raise TypeError("stats_collector must be a StatsCollector")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must expose a callable execute")
        captures_output = getattr(executor, "captures_output", None)
        if not isinstance(captures_output, bool):
            raise TypeError("executor.captures_output must be a boolean")
        if not captures_output:
            raise ValueError(
                "LLMDifficultyClassifier requires an executor that captures response text"
            )
        if (
            not isinstance(model, str)
            or not model
            or model != model.strip()
        ):
            raise ValueError("classifier model must be a non-empty trimmed string")
        if not isinstance(prompt_prefix, str) or not prompt_prefix.strip():
            raise ValueError("classifier prompt_prefix must be a non-empty string")
        if fallback not in ("easy", "hard", "raise"):
            raise ValueError("classifier fallback must be easy, hard, or raise")
        for name, value in (
            ("max_prompt_chars", max_prompt_chars),
            ("max_tokens", max_tokens),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not is_finite_real(temperature)
            or not 0 <= float(temperature) <= 2
        ):
            raise ValueError("temperature must be finite and between 0 and 2")
        self._stats_collector = stats_collector
        self._executor = stats_collector.wrap(executor, "classifier")
        self._model = model
        self._prompt_prefix = prompt_prefix
        self._fallback = fallback
        self._max_prompt_chars = int(max_prompt_chars)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

    @property
    def stats_collector(self) -> StatsCollector:
        return self._stats_collector

    @property
    def executor(self) -> ModelExecutor:
        return self._executor

    def classify(self, task: Task) -> DifficultyClassification:
        """Return an assessment or apply the configured explicit failure policy."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        try:
            result = self._executor.execute(ExecutionRequest(
                model=self._model,
                prompt=(
                    self._prompt_prefix.rstrip("\r\n")
                    + "\n"
                    + task.prompt[: self._max_prompt_chars]
                ),
                role="classifier",
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ))
        except Exception as exc:
            if self._fallback == "raise":
                raise
            return self._fallback_classification(
                reason=f"Classifier execution failed: {_exception_detail(exc)}",
                model=self._model,
            )
        try:
            def reject_duplicate_keys(pairs):
                payload = {}
                for key, value in pairs:
                    if key in payload:
                        raise ValueError(f"duplicate classifier key {key!r}")
                    payload[key] = value
                return payload

            payload = json.loads(
                result.text,
                object_pairs_hook=reject_duplicate_keys,
            )
            if not isinstance(payload, dict) or set(payload) != {"d"}:
                raise ValueError("classifier object must contain exactly the key 'd'")
            label = payload["d"]
            if label not in ("easy", "hard"):
                raise ValueError(f"unexpected classifier label {label!r}")
            difficulty: Difficulty = label
        except Exception as exc:
            if self._fallback == "raise":
                raise ValueError("classifier returned an invalid response") from exc
            return self._fallback_classification(
                reason=(
                    "Classifier response was invalid: "
                    f"{_exception_detail(exc)}"
                ),
                model=result.model or self._model,
            )

        return DifficultyClassification(
            difficulty=difficulty,
            reason=f"Classifier selected {difficulty}",
            model=result.model or self._model,
        )

    def _fallback_classification(
        self,
        *,
        reason: str,
        model: str,
    ) -> DifficultyClassification:
        difficulty = cast(Difficulty, self._fallback)
        return DifficultyClassification(
            difficulty=difficulty,
            reason=f"{reason}; defaulted to {difficulty}",
            model=model,
        )


def _exception_detail(error: Exception) -> str:
    return str(error).strip() or type(error).__name__
