"""Shared synchronous Chat Completions mechanics with response capture."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType

from .._numeric import is_finite_real
from ..domain import ExecutionRequest, ModelResult


_MISSING = object()
_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "refusal": "refusal",
    "content_filter": "content_filter",
    "tool_call": "tool_call",
    "tool_calls": "tool_call",
    "error": "error",
}


def _field(value, name: str):
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _optional_non_negative_integer(
    value,
    field_name: str,
    provider_name: str,
) -> int | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(
            f"{provider_name} response {field_name} must be a non-negative integer"
        )
    return int(value)


def _optional_non_negative_number(
    value,
    field_name: str,
    provider_name: str,
) -> float | None:
    if value is _MISSING or value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not is_finite_real(value)
        or value < 0
    ):
        raise ValueError(
            f"{provider_name} response {field_name} must be finite and non-negative"
        )
    return float(value)


def _usage_detail(
    usage,
    group_name: str,
    field_name: str,
    provider_name: str,
) -> int | None:
    if usage is None:
        return None
    details = _field(usage, group_name)
    if details is _MISSING or details is None:
        return None
    return _optional_non_negative_integer(
        _field(details, field_name),
        f"usage.{group_name}.{field_name}",
        provider_name,
    )


def _finish_evidence(
    choice,
    refusal: str | None,
    provider_name: str,
) -> tuple[str, str | None]:
    raw_finish = _field(choice, "finish_reason")
    native_finish = _field(choice, "native_finish_reason")
    for field_name, value in (
        ("finish_reason", raw_finish),
        ("native_finish_reason", native_finish),
    ):
        if value is not _MISSING and value is not None and (
            not isinstance(value, str) or not value
        ):
            raise TypeError(
                f"{provider_name} response {field_name} must be text or None"
            )

    if native_finish is _MISSING or native_finish is None:
        native = raw_finish if isinstance(raw_finish, str) else None
    else:
        native = native_finish
    if refusal is not None:
        return "refusal", native
    if not isinstance(raw_finish, str):
        return "unknown", native
    return _FINISH_REASONS.get(raw_finish, "unknown"), native


class _ChatCompletionsExecutor:
    """Shared, private implementation for explicit provider dialects."""

    __slots__ = (
        "_client",
        "_create",
        "_system_prompts",
        "_max_tokens",
        "_temperatures",
        "_default_max_tokens",
        "_temperature",
        "_reasoning_efforts",
        "_default_reasoning_effort",
        "_provider_name",
        "_max_tokens_field",
        "_send_temperature",
        "_read_provider_cost",
    )

    captures_output = True

    def __init__(
        self,
        client,
        system_prompts: Mapping[str, str] | None = None,
        max_tokens: Mapping[str, int] | None = None,
        temperatures: Mapping[str, float] | None = None,
        reasoning_efforts: Mapping[str, str] | None = None,
        *,
        default_max_tokens: int,
        temperature: float | None,
        default_reasoning_effort: str | None,
        provider_name: str,
        max_tokens_field: str,
        send_temperature: bool,
        read_provider_cost: bool,
    ):
        if (
            isinstance(default_max_tokens, bool)
            or not isinstance(default_max_tokens, Integral)
            or default_max_tokens < 1
        ):
            raise ValueError("default_max_tokens must be a positive integer")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not is_finite_real(temperature)
            or not 0 <= float(temperature) <= 2
        ):
            raise ValueError("temperature must be finite, between 0 and 2, or None")
        efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if default_reasoning_effort is not None and (
            default_reasoning_effort not in efforts
        ):
            raise ValueError("default_reasoning_effort is invalid")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("provider_name must be non-empty text")
        if max_tokens_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("max_tokens_field is invalid")
        create = getattr(
            getattr(getattr(client, "chat", None), "completions", None),
            "create",
            None,
        )
        if not callable(create):
            raise TypeError(
                "client must expose a callable chat.completions.create"
            )
        for name, value in (
            ("system_prompts", system_prompts),
            ("max_tokens", max_tokens),
            ("temperatures", temperatures),
            ("reasoning_efforts", reasoning_efforts),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")
        resolved_system_prompts = dict(
            {} if system_prompts is None else system_prompts
        )
        resolved_max_tokens = dict({} if max_tokens is None else max_tokens)
        resolved_temperatures = dict(
            {} if temperatures is None else temperatures
        )
        resolved_reasoning_efforts = dict(
            {} if reasoning_efforts is None else reasoning_efforts
        )
        for model, prompt in resolved_system_prompts.items():
            if (
                not isinstance(model, str)
                or not model
                or model != model.strip()
            ):
                raise ValueError(
                    "system prompt model IDs must be non-empty trimmed strings"
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("system prompts must be non-empty strings")
        for model, value in resolved_max_tokens.items():
            if (
                not isinstance(model, str)
                or not model
                or model != model.strip()
            ):
                raise ValueError("max-token model IDs must be non-empty trimmed strings")
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("per-model max_tokens must be positive integers")
        for model, value in resolved_temperatures.items():
            if (
                not isinstance(model, str)
                or not model
                or model != model.strip()
            ):
                raise ValueError(
                    "temperature model IDs must be non-empty trimmed strings"
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not is_finite_real(value)
                or not 0 <= float(value) <= 2
            ):
                raise ValueError("per-model temperatures must be between 0 and 2")
        for model, value in resolved_reasoning_efforts.items():
            if (
                not isinstance(model, str)
                or not model
                or model != model.strip()
            ):
                raise ValueError(
                    "reasoning-effort model IDs must be non-empty trimmed strings"
                )
            if value not in efforts:
                raise ValueError("per-model reasoning_efforts contain an invalid value")
        self._client = client
        self._create = create
        self._system_prompts = MappingProxyType(resolved_system_prompts)
        self._max_tokens = MappingProxyType(resolved_max_tokens)
        self._temperatures = MappingProxyType(resolved_temperatures)
        self._reasoning_efforts = MappingProxyType(resolved_reasoning_efforts)
        self._default_max_tokens = int(default_max_tokens)
        self._temperature = None if temperature is None else float(temperature)
        self._default_reasoning_effort = default_reasoning_effort
        self._provider_name = provider_name.strip()
        self._max_tokens_field = max_tokens_field
        self._send_temperature = bool(send_temperature)
        self._read_provider_cost = bool(read_provider_cost)

    def execute(self, request: ExecutionRequest) -> ModelResult:
        """Call one model and retain its text and raw token-usage object."""

        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        messages = []
        system_prompt = self._system_prompts.get(request.model)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self._max_tokens.get(request.model, self._default_max_tokens)
        )
        parameters = {
            "model": request.model,
            "messages": messages,
            self._max_tokens_field: max_tokens,
        }
        if self._send_temperature:
            temperature = (
                request.temperature
                if request.temperature is not None
                else self._temperatures.get(request.model, self._temperature)
            )
            if temperature is not None:
                parameters["temperature"] = temperature
        reasoning_effort = self._reasoning_efforts.get(
            request.model,
            self._default_reasoning_effort,
        )
        if reasoning_effort is not None:
            parameters["reasoning_effort"] = reasoning_effort
        response = self._create(**parameters)
        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError(
                f"{self._provider_name} response must contain at least one choice"
            )
        choice = choices[0]
        message = _field(choice, "message")
        if message is _MISSING or message is None:
            raise ValueError(
                f"{self._provider_name} response choice must contain a message"
            )
        content = _field(message, "content")
        if content is _MISSING:
            raise ValueError(
                f"{self._provider_name} response choice must contain a message"
            )
        if content is not None and not isinstance(content, str):
            raise TypeError(
                f"{self._provider_name} response message content must be text or None"
            )
        text = content or ""
        refusal = _field(message, "refusal")
        if refusal is _MISSING or refusal is None:
            refusal = None
        elif not isinstance(refusal, str):
            raise TypeError(
                f"{self._provider_name} response message refusal must be text or None"
            )
        elif not refusal.strip():
            raise ValueError(
                f"{self._provider_name} response message refusal must be "
                "non-empty or None"
            )
        finish_reason, native_finish_reason = _finish_evidence(
            choice,
            refusal,
            self._provider_name,
        )

        call_usage = _field(response, "usage")
        if call_usage is _MISSING:
            call_usage = None
        reasoning_tokens = _usage_detail(
            call_usage,
            "completion_tokens_details",
            "reasoning_tokens",
            self._provider_name,
        )
        cached_input_tokens = _usage_detail(
            call_usage,
            "prompt_tokens_details",
            "cached_tokens",
            self._provider_name,
        )
        cache_write_input_tokens = _usage_detail(
            call_usage,
            "prompt_tokens_details",
            "cache_write_tokens",
            self._provider_name,
        )
        completion_tokens = _optional_non_negative_integer(
            (
                _field(call_usage, "completion_tokens")
                if call_usage is not None
                else _MISSING
            ),
            "usage.completion_tokens",
            self._provider_name,
        )
        prompt_tokens = _optional_non_negative_integer(
            (
                _field(call_usage, "prompt_tokens")
                if call_usage is not None
                else _MISSING
            ),
            "usage.prompt_tokens",
            self._provider_name,
        )
        provider_cost_usd = _optional_non_negative_number(
            (
                _field(call_usage, "cost")
                if self._read_provider_cost and call_usage is not None
                else _MISSING
            ),
            "usage.cost",
            self._provider_name,
        )
        if completion_tokens is not None and reasoning_tokens is not None:
            if reasoning_tokens > completion_tokens:
                raise ValueError(
                    f"{self._provider_name} response reasoning_tokens cannot exceed "
                    "completion_tokens"
                )
        visible_output_tokens = (
            None
            if finish_reason == "tool_call"
            else 0
            if not text.strip()
            else None
        )
        if (
            finish_reason != "tool_call"
            and visible_output_tokens is None
            and completion_tokens is not None
            and reasoning_tokens is not None
        ):
            visible_output_tokens = completion_tokens - reasoning_tokens
        for field_name, detail in (
            ("cached_tokens", cached_input_tokens),
            ("cache_write_tokens", cache_write_input_tokens),
        ):
            if (
                prompt_tokens is not None
                and detail is not None
                and detail > prompt_tokens
            ):
                raise ValueError(
                    f"{self._provider_name} response {field_name} cannot exceed "
                    "prompt_tokens"
                )

        reported_model = _field(response, "model")
        if (
            not isinstance(reported_model, str)
            or not reported_model
            or reported_model != reported_model.strip()
        ):
            reported_model = None
        return ModelResult(
            model=reported_model,
            text=text,
            usage=call_usage,
            raw_text=text,
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            refusal=refusal,
            applied_max_tokens=max_tokens,
            visible_output_tokens=visible_output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            provider_cost_usd=provider_cost_usd,
        )

