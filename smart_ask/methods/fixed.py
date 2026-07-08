"""Fixed-model routing method used by overrides and baselines."""

from ..domain import Context, RouteResult, RoutingEvent, Task


class FixedRoutingMethod:
    """Always select one configured model without classification."""

    requires_response_text = False

    def __init__(
        self,
        model: str,
        role: str,
        *,
        label: str = "fixed",
        prompt_prefix: str = "",
        prompt_suffix: str = "",
    ):
        for name, value in (("model", model), ("role", role), ("label", label)):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
        for name, value in (
            ("prompt_prefix", prompt_prefix),
            ("prompt_suffix", prompt_suffix),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
        self._model = model
        self._role = role
        self._label = label
        self._prompt_prefix = prompt_prefix
        self._prompt_suffix = prompt_suffix

    def route(self, task: Task, context: Context | None = None) -> RouteResult:
        """Execute the fixed model once, then accept its response."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if context is None:
            context = Context()
        elif not isinstance(context, Context):
            raise TypeError("context must be a Context or None")
        if context.attempts:
            return RouteResult(action="accept")
        event = RoutingEvent(
            source="fixed-method",
            outcome="fixed",
            reason="Configured fixed route",
        )
        return RouteResult(
            action="execute",
            model=self._model,
            prompt=self._prompt_prefix + task.prompt + self._prompt_suffix,
            role=self._role,
            phase="fixed",
            label=self._label,
            routing_events=(event,),
        )
