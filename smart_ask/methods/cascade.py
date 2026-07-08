"""Response-aware cascade routing method."""

from ..domain import Context, RouteResult, Task
from .classifiers.base import DifficultyClassification, DifficultyClassifier
from .escalation.base import EscalationDecision, EscalationPolicy


class CascadeRoutingMethod:
    """Classify once, then optionally escalate an insufficient easy response."""

    requires_response_text = True

    def __init__(
        self,
        classifier: DifficultyClassifier,
        escalation_policy: EscalationPolicy,
        easy_model: str,
        hard_model: str,
    ):
        if not callable(getattr(classifier, "classify", None)):
            raise TypeError("classifier must expose a callable classify")
        for operation in (
            "prepare_candidate_prompt",
            "assess",
            "prepare_escalation_prompt",
        ):
            if not callable(getattr(escalation_policy, operation, None)):
                raise TypeError(
                    f"escalation_policy must expose a callable {operation}"
                )
        for name, value in (("easy_model", easy_model), ("hard_model", hard_model)):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
        self._classifier = classifier
        self._escalation_policy = escalation_policy
        self._easy_model = easy_model
        self._hard_model = hard_model

    @property
    def classifier(self) -> DifficultyClassifier:
        return self._classifier

    def route(self, task: Task, context: Context | None = None) -> RouteResult:
        """Choose the next cascade action from the immutable per-task context."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if context is None:
            context = Context()
        elif not isinstance(context, Context):
            raise TypeError("context must be a Context or None")
        if not context.attempts:
            classification = self._classifier.classify(task)
            if not isinstance(classification, DifficultyClassification):
                raise TypeError(
                    "classifier.classify must return a DifficultyClassification"
                )
            event = classification.to_routing_event()
            if classification.difficulty == "hard":
                return RouteResult(
                    action="execute",
                    model=self._hard_model,
                    prompt=task.prompt,
                    role="writer",
                    phase="initial-hard",
                    label="hard direct",
                    routing_events=(event,),
                )
            return RouteResult(
                action="execute",
                model=self._easy_model,
                prompt=self._escalation_policy.prepare_candidate_prompt(task),
                role="generator",
                phase="initial-easy",
                label="easy primary",
                routing_events=(event,),
            )

        previous_route = context.previous_route
        if previous_route and previous_route.phase == "initial-easy":
            previous_attempt = context.previous_attempt
            if previous_attempt is None:
                raise RuntimeError("An initial easy route must have a model response")
            decision = self._escalation_policy.assess(previous_attempt)
            if not isinstance(decision, EscalationDecision):
                raise TypeError(
                    "escalation_policy.assess must return an EscalationDecision"
                )
            event = decision.to_routing_event()
            if decision.should_escalate:
                return RouteResult(
                    action="execute",
                    model=self._hard_model,
                    prompt=self._escalation_policy.prepare_escalation_prompt(task),
                    role="fixer",
                    phase="escalation",
                    label="hard escalation",
                    routing_events=(event,),
                )
            return RouteResult(action="accept", routing_events=(event,))

        return RouteResult(action="accept")
