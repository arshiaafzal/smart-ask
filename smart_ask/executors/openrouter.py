"""Direct OpenRouter execution adapter with response capture."""

from __future__ import annotations

from collections.abc import Mapping

from ..domain import ExecutionRequest, ModelResult


class OpenRouterExecutor:
    """Execute models directly through an OpenAI-compatible client."""

    captures_output = True

    def __init__(
        self,
        client,
        system_prompts: Mapping[str, str] | None = None,
        max_tokens: Mapping[str, int] | None = None,
        temperatures: Mapping[str, float] | None = None,
        default_max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.client = client
        self.system_prompts = dict(system_prompts or {})
        self.max_tokens = dict(max_tokens or {})
        self.temperatures = dict(temperatures or {})
        self.default_max_tokens = default_max_tokens
        self.temperature = temperature

    def execute(self, request: ExecutionRequest) -> ModelResult:
        """Call one model and retain its text and raw token-usage object."""

        messages = []
        system_prompt = self.system_prompts.get(request.model)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self.max_tokens.get(request.model, self.default_max_tokens)
        )
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.temperatures.get(request.model, self.temperature)
        )

        response = self.client.chat.completions.create(
            model=request.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        return ModelResult(
            model=request.model,
            text=_strip_fences(text),
            usage=response.usage,
            raw_text=text,
        )


def _strip_fences(text: str) -> str:
    """Remove an outer Markdown code fence while preserving code indentation."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return text.rstrip()

    lines = stripped.splitlines()
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "```"),
        len(lines),
    )
    inner = lines[1:closing_index]
    return "\n".join(inner).rstrip()
