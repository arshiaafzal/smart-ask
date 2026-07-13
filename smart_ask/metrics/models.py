"""Minimal provider-neutral token evidence used by pricing."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    visible_output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "visible_output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
            if value is not None:
                object.__setattr__(self, name, int(value))
        prompt = self.prompt_tokens
        completion = self.completion_tokens
        total = self.total_tokens
        if prompt is not None and completion is not None:
            expected = prompt + completion
            if total is None:
                object.__setattr__(self, "total_tokens", expected)
            elif total != expected:
                raise ValueError("total_tokens must equal prompt + completion tokens")
        elif total is not None:
            known = prompt if prompt is not None else completion
            if known is not None and known > total:
                raise ValueError("a token component cannot exceed total_tokens")
        for name, value, capacity in (
            ("reasoning_tokens", self.reasoning_tokens, completion),
            ("visible_output_tokens", self.visible_output_tokens, completion),
            ("cached_input_tokens", self.cached_input_tokens, prompt),
            ("cache_write_input_tokens", self.cache_write_input_tokens, prompt),
        ):
            if value is not None and capacity is not None and value > capacity:
                raise ValueError(f"{name} exceeds its token component")
