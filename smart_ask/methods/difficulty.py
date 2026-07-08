"""One-shot routing method driven by a task-difficulty classifier."""

from ..domain import Context, RouteResult, Task
from .classifiers.base import DifficultyClassification, DifficultyClassifier


class DifficultyRoutingMethod:
    """Choose the easy or hard model from one difficulty classification."""

    requires_response_text = False

    def __init__(
        self,
        classifier: DifficultyClassifier,
        easy_model: str,
        hard_model: str,
    ):
        if not callable(getattr(classifier, "classify", None)):
            raise TypeError("classifier must expose a callable classify")
        for name, value in (("easy_model", easy_model), ("hard_model", hard_model)):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
        self._classifier = classifier
        self._easy_model = easy_model
        self._hard_model = hard_model

    @property
    def classifier(self) -> DifficultyClassifier:
        return self._classifier

    def route(self, task: Task, context: Context | None = None) -> RouteResult:
        """Classify and execute once, then accept the resulting response."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if context is None:
            context = Context()
        elif not isinstance(context, Context):
            raise TypeError("context must be a Context or None")
        if context.attempts:
            return RouteResult(action="accept")

        classification = self._classifier.classify(task)
        if not isinstance(classification, DifficultyClassification):
            raise TypeError(
                "classifier.classify must return a DifficultyClassification"
            )
        hard = classification.difficulty == "hard"
        return RouteResult(
            action="execute",
            model=self._hard_model if hard else self._easy_model,
            prompt=task.prompt,
            role="writer" if hard else "generator",
            phase="initial-hard" if hard else "initial-easy",
            label="classified hard" if hard else "classified easy",
            routing_events=(classification.to_routing_event(),),
        )
