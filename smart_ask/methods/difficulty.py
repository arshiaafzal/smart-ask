"""One-shot routing method driven by a task-difficulty classifier."""

from ..config import EASY_MODEL, HARD_MODEL
from ..domain import Context, RouteResult, Task
from .classifiers.base import DifficultyClassifier


class DifficultyRoutingMethod:
    """Choose the easy or hard model from one difficulty classification."""

    requires_response_text = False

    def __init__(
        self,
        classifier: DifficultyClassifier,
        easy_model: str = EASY_MODEL,
        hard_model: str = HARD_MODEL,
    ):
        self.classifier = classifier
        self.easy_model = easy_model
        self.hard_model = hard_model

    def route(self, task: Task, context: Context = Context()) -> RouteResult:
        """Classify and execute once, then accept the resulting response."""

        if context.attempts:
            return RouteResult(action="accept")

        classification = self.classifier.classify(task)
        hard = classification.difficulty == "hard"
        return RouteResult(
            action="execute",
            model=self.hard_model if hard else self.easy_model,
            prompt=task.prompt,
            role="writer" if hard else "generator",
            phase="initial-hard" if hard else "initial-easy",
            label="classified hard" if hard else "classified easy",
            routing_events=(classification.to_routing_event(),),
        )
