"""Provider-neutral Responses API executor mechanics."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType

from ..domain import ExecutionRequest, ModelResult


_MISSING = object()
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


def _field(value, name: str):
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _count(value, name: str) -> int | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(
            f"Responses API field {name} must be a non-negative integer"
        )
    return int(value)


def _detail(value, group: str, name: str) -> int | None:
    if value is None:
        return None
    details = _field(value, group)
    if details is _MISSING or details is None:
        return None
    return _count(_field(details, name), f"usage.{group}.{name}")


def _output(response) -> tuple[str, str | None]:
    direct = _field(response, "output_text")
    text_parts: list[str] = []
    refusal_parts: list[str] = []
    items = _field(response, "output")
    if isinstance(items, (list, tuple)):
        for item in items:
            if _field(item, "type") != "message":
                continue
            content = _field(item, "content")
            if not isinstance(content, (list, tuple)):
                continue
            for block in content:
                block_type = _field(block, "type")
                value = _field(
                    block,
                    "refusal" if block_type == "refusal" else "text",
                )
                if not isinstance(value, str):
                    continue
                if block_type == "refusal":
                    refusal_parts.append(value)
                elif block_type == "output_text":
                    text_parts.append(value)
    text = direct if isinstance(direct, str) else "".join(text_parts)
    refusal = "".join(refusal_parts) or None
    return text, refusal


def _finish(response, refusal: str | None) -> tuple[str, str | None]:
    status = _field(response, "status")
    if not isinstance(status, str) or not status:
        status = None
    if refusal is not None:
        return "refusal", status
    if status == "completed":
        return "stop", status
    if status == "failed":
        return "error", status
    if status == "incomplete":
        details = _field(response, "incomplete_details")
        reason = _field(details, "reason") if details is not _MISSING else _MISSING
        if reason == "max_output_tokens":
            return "length", reason
        if reason == "content_filter":
            return "content_filter", reason
        return "unknown", reason if isinstance(reason, str) else status
    return "unknown", status


class ResponsesExecutor:
    """Execute one-shot model calls through a Responses-compatible API."""

    captures_output = True
    _include_store = False

    def __init__(
        self,
        client,
        system_prompts: Mapping[str, str] | None = None,
        max_tokens: Mapping[str, int] | None = None,
        reasoning_efforts: Mapping[str, str] | None = None,
        *,
        default_max_tokens: int,
        reasoning_effort: str,
    ):
        if (
            isinstance(default_max_tokens, bool)
            or not isinstance(default_max_tokens, Integral)
            or default_max_tokens < 1
        ):
            raise ValueError("default_max_tokens must be a positive integer")
        if reasoning_effort not in _EFFORTS:
            raise ValueError("reasoning_effort is invalid")
        create = getattr(getattr(client, "responses", None), "create", None)
        if not callable(create):
            raise TypeError("client must expose a callable responses.create")
        for name, value in (
            ("system_prompts", system_prompts),
            ("max_tokens", max_tokens),
            ("reasoning_efforts", reasoning_efforts),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")
        prompts = dict(system_prompts or {})
        limits = dict(max_tokens or {})
        efforts = dict(reasoning_efforts or {})
        for model, prompt in prompts.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("system prompt model IDs must be non-empty text")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("system prompts must be non-empty text")
        for model, limit in limits.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("max-token model IDs must be non-empty text")
            if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
                raise ValueError("per-model max_tokens must be positive integers")
        for model, effort in efforts.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("reasoning-effort model IDs must be non-empty text")
            if effort not in _EFFORTS:
                raise ValueError("per-model reasoning_efforts contain an invalid value")
        self._create = create
        self._system_prompts = MappingProxyType(prompts)
        self._max_tokens = MappingProxyType(limits)
        self._reasoning_efforts = MappingProxyType(efforts)
        self._default_max_tokens = int(default_max_tokens)
        self._reasoning_effort = reasoning_effort

    def execute(self, request: ExecutionRequest) -> ModelResult:
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self._max_tokens.get(request.model, self._default_max_tokens)
        )
        effort = self._reasoning_efforts.get(
            request.model,
            self._reasoning_effort,
        )
        parameters = {
            "model": request.model,
            "input": request.prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": effort},
        }
        if self._include_store:
            parameters["store"] = False
        instructions = self._system_prompts.get(request.model)
        if instructions is not None:
            parameters["instructions"] = instructions
        response = self._create(**parameters)
        text, refusal = _output(response)
        finish_reason, native_finish_reason = _finish(response, refusal)
        usage = _field(response, "usage")
        if usage is _MISSING:
            usage = None
        input_tokens = _count(
            _field(usage, "input_tokens") if usage is not None else _MISSING,
            "usage.input_tokens",
        )
        output_tokens = _count(
            _field(usage, "output_tokens") if usage is not None else _MISSING,
            "usage.output_tokens",
        )
        reasoning_tokens = _detail(
            usage,
            "output_tokens_details",
            "reasoning_tokens",
        )
        cached_input_tokens = _detail(
            usage,
            "input_tokens_details",
            "cached_tokens",
        )
        cache_write_input_tokens = _detail(
            usage,
            "input_tokens_details",
            "cache_write_tokens",
        )
        if (
            output_tokens is not None
            and reasoning_tokens is not None
            and reasoning_tokens > output_tokens
        ):
            raise ValueError(
                "Responses API reasoning_tokens cannot exceed output_tokens"
            )
        if (
            input_tokens is not None
            and cached_input_tokens is not None
            and cached_input_tokens > input_tokens
        ):
            raise ValueError(
                "Responses API cached_tokens cannot exceed input_tokens"
            )
        visible_output_tokens = 0 if not text.strip() else None
        if (
            visible_output_tokens is None
            and output_tokens is not None
            and reasoning_tokens is not None
        ):
            visible_output_tokens = output_tokens - reasoning_tokens
        model = _field(response, "model")
        if not isinstance(model, str) or not model.strip():
            model = None
        return ModelResult(
            model=model,
            text=text,
            usage=usage,
            raw_text=text,
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            refusal=refusal,
            applied_max_tokens=max_tokens,
            visible_output_tokens=visible_output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            provider_cost_usd=None,
        )
