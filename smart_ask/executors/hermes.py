"""Hermes CLI execution adapter."""

import subprocess
from numbers import Integral

from ..domain import ExecutionRequest, ModelResult


class HermesExecutor:
    """Execute a selected model through the external Hermes CLI."""

    __slots__ = ("_provider", "_command", "_runner")

    captures_output = False

    def __init__(
        self,
        provider: str,
        command: str,
        runner=subprocess.run,
    ):
        if (
            not isinstance(provider, str)
            or not provider
            or provider != provider.strip()
        ):
            raise ValueError("provider must be a non-empty trimmed string")
        if (
            not isinstance(command, str)
            or not command
            or command != command.strip()
        ):
            raise ValueError("command must be a non-empty trimmed string")
        if not callable(runner):
            raise TypeError("runner must be callable")
        self._provider = provider
        self._command = command
        self._runner = runner

    def execute(self, request: ExecutionRequest) -> ModelResult:
        """Run one Hermes query, ignoring unsupported optional tuning hints."""

        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        command = [
            self._command,
            "chat",
            "-q",
            request.prompt,
            "-m",
            request.model,
            "--provider",
            self._provider,
        ]
        completed = self._runner(command)
        returncode = getattr(completed, "returncode", None)
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, Integral)
        ):
            raise TypeError("Hermes runner must return an integer returncode")
        if returncode != 0:
            raise subprocess.CalledProcessError(int(returncode), command)
        captured_output = getattr(completed, "stdout", None)
        if isinstance(captured_output, bytes):
            captured_output = captured_output.decode(errors="replace")
        if captured_output is not None and not isinstance(captured_output, str):
            raise TypeError("Hermes runner stdout must be text, bytes, or None")
        output = captured_output or ""
        return ModelResult(
            # Hermes does not expose provider response metadata. The requested
            # model remains on ExecutionRequest; it is not actual-model evidence.
            model=None,
            text=output,
            raw_text=captured_output,
            finish_reason="unknown",
            output_status="unavailable" if captured_output is None else None,
            applied_max_tokens=None,
            visible_output_tokens=None,
            reasoning_tokens=None,
            cached_input_tokens=None,
            cache_write_input_tokens=None,
            provider_cost_usd=None,
        )
