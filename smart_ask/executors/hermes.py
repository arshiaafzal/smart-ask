"""Hermes CLI execution adapter."""

import subprocess

from ..domain import ExecutionRequest, ModelResult


class HermesExecutor:
    """Execute a selected model through the external Hermes CLI."""

    captures_output = False

    def __init__(
        self,
        provider: str = "openrouter",
        command: str = "hermes",
        runner=subprocess.run,
    ):
        self.provider = provider
        self.command = command
        self.runner = runner

    def execute(self, request: ExecutionRequest) -> ModelResult:
        """Run one Hermes query, ignoring unsupported optional tuning hints."""

        command = [
            self.command,
            "chat",
            "-q",
            request.prompt,
            "-m",
            request.model,
            "--provider",
            self.provider,
        ]
        completed = self.runner(command)
        returncode = getattr(completed, "returncode", None)
        if returncode not in (None, 0):
            raise subprocess.CalledProcessError(returncode, command)
        output = getattr(completed, "stdout", "") or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return ModelResult(
            model=request.model,
            text=output,
            returncode=returncode,
            raw_text=output,
        )
