"""Response-aware cascade routing method."""

from ..config import EASY_MODEL, HARD_MODEL
from ..domain import Context, RouteResult, Task
from .classifiers.base import DifficultyClassifier
from .escalation.base import EscalationPolicy


class CascadeRoutingMethod:
    """Classify once, then optionally escalate an insufficient easy response."""

    requires_response_text = True

    def __init__(
        self,
        classifier: DifficultyClassifier,
        escalation_policy: EscalationPolicy,
        easy_model: str = EASY_MODEL,
        hard_model: str = HARD_MODEL,
    ):
        self.classifier = classifier
        self.escalation_policy = escalation_policy
        self.easy_model = easy_model
        self.hard_model = hard_model

    def route(self, task: Task, context: Context = Context()) -> RouteResult:
        """Choose the next cascade action from the immutable per-task context."""

        if not context.attempts:
            classification = self.classifier.classify(task)
            event = classification.to_routing_event()
            if classification.difficulty == "hard":
                return RouteResult(
                    action="execute",
                    model=self.hard_model,
                    prompt=task.prompt,
                    role="writer",
                    phase="initial-hard",
                    label="hard direct",
                    routing_events=(event,),
                )
            return RouteResult(
                action="execute",
                model=self.easy_model,
                prompt=self.escalation_policy.prepare_candidate_prompt(task),
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
            decision = self.escalation_policy.assess(previous_attempt)
            event = decision.to_routing_event()
            if decision.should_escalate:
                return RouteResult(
                    action="execute",
                    model=self.hard_model,
                    prompt=self.escalation_policy.prepare_escalation_prompt(task),
                    role="fixer",
                    phase="escalation",
                    label="hard escalation",
                    routing_events=(event,),
                )
            return RouteResult(action="accept", routing_events=(event,))

        return RouteResult(action="accept")
