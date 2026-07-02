"""Model-backed difficulty classifier independent of any provider transport."""

from __future__ import annotations

import json

from ...config import CLASSIFIER_MODEL
from ...domain import ExecutionRequest, Task
from ...executors.base import ModelExecutor
from .base import DifficultyClassification


DEFAULT_CLASSIFICATION_PROMPT = """\
You are routing a coding/AI task to either a cheap model (easy) or an expert model (hard).
Label "hard" if ANY of these are true:
- Requires dynamic programming, graph traversal, or non-obvious algorithm
- Has subtle edge cases a junior programmer would likely miss
- Needs number theory, combinatorics, or careful mathematical reasoning
- Complex multi-system design or advanced architecture decisions
Label "easy" if:
- Solvable with basic loops, string ops, or simple math
- Straightforward Q&A, explanation, debug, or format task
- Edge cases are obvious and minimal
Reply ONLY with JSON: {"d":"easy"} or {"d":"hard"}
Task:\n"""


class LLMDifficultyClassifier:
    """Classify task difficulty by prompting a model through an executor."""

    def __init__(
        self,
        executor: ModelExecutor,
        model: str = CLASSIFIER_MODEL,
        prompt_prefix: str = DEFAULT_CLASSIFICATION_PROMPT,
        max_prompt_chars: int = 1200,
        max_tokens: int = 20,
        temperature: float = 0.0,
    ):
        if not getattr(executor, "captures_output", False):
            raise ValueError(
                "LLMDifficultyClassifier requires an executor that captures response text"
            )
        self.executor = executor
        self.model = model
        self.prompt_prefix = prompt_prefix
        self.max_prompt_chars = max_prompt_chars
        self.max_tokens = max_tokens
        self.temperature = temperature

    def classify(self, task: Task) -> DifficultyClassification:
        """Return an easy/hard assessment; failures conservatively default to easy."""

        try:
            result = self.executor.execute(ExecutionRequest(
                model=self.model,
                prompt=self.prompt_prefix + task.prompt[: self.max_prompt_chars],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ))
        except Exception as exc:
            return DifficultyClassification(
                difficulty="easy",
                reason=f"Classifier execution failed; defaulted to easy: {exc}",
                model=self.model,
            )

        try:
            raw = result.text.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw).get("d", "easy")
            difficulty = parsed if parsed in ("easy", "hard") else "easy"
            reason = (
                f"Classifier selected {difficulty}"
                if parsed in ("easy", "hard")
                else f"Unexpected classifier label {parsed!r}; defaulted to easy"
            )
            return DifficultyClassification(
                difficulty=difficulty,
                reason=reason,
                model=result.model,
                usage=result.usage,
            )
        except Exception as exc:
            return DifficultyClassification(
                difficulty="easy",
                reason=f"Classifier response was invalid; defaulted to easy: {exc}",
                model=result.model,
                usage=result.usage,
            )
