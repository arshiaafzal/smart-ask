"""Opt-in content-bearing event traces for conversation debugging."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .domain import ConversationRequest, SessionContext, thaw_value


_TEXT_CHUNK_SIZE = 4000


class ConversationTraceWriter:
    """Emit compact semantic JSONL-ready events for one runtime request."""

    def __init__(
        self,
        *,
        sink: Callable[[dict[str, Any]], None] | None,
        run_id: str,
        errors: list[str],
    ):
        self._sink = sink
        self._run_id = run_id
        self._errors = errors
        self._sequence = 0
        self._failed = False

    @property
    def enabled(self) -> bool:
        return self._sink is not None and not self._failed

    def _write(self, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        self._sequence += 1
        value = {
            "schema": "smart-ask.conversation-trace-event/v1",
            "run_id": self._run_id,
            "sequence": self._sequence,
            "event": event,
            **data,
        }
        try:
            self._sink(value)
        except Exception as exc:
            self._errors.append(f"{type(exc).__name__}: {exc}")
            self._failed = True

    @staticmethod
    def _chunks(text: str) -> list[str]:
        return [
            text[offset:offset + _TEXT_CHUNK_SIZE]
            for offset in range(0, len(text), _TEXT_CHUNK_SIZE)
        ]

    def _context_block(
        self,
        scope: str,
        index: int,
        block: Any,
        *,
        message_index: int | None = None,
    ) -> None:
        value = thaw_value(block)
        common = {"scope": scope, "index": index}
        if message_index is not None:
            common["message_index"] = message_index
        text_key = next(
            (
                key for key in ("text", "content")
                if isinstance(value, dict) and isinstance(value.get(key), str)
            ),
            None,
        )
        if text_key is None or len(value[text_key]) <= _TEXT_CHUNK_SIZE:
            self._write("context_block", **common, block=value)
            return
        text = value.pop(text_key)
        chunks = self._chunks(text)
        for chunk_index, chunk in enumerate(chunks):
            self._write(
                "context_block",
                **common,
                block=value,
                text_field=text_key,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                text=chunk,
            )

    def start(
        self,
        *,
        strategy_name: str,
        strategy_digest: str,
        session: SessionContext,
        request: ConversationRequest,
    ) -> None:
        self._write(
            "run_start",
            strategy={"name": strategy_name, "digest": strategy_digest},
            session_id=session.session_id,
            agent_id=session.agent_id,
            parent_agent_id=session.parent_agent_id,
        )
        self._write(
            "request_metadata",
            parameters=thaw_value(request.parameters),
            extensions=thaw_value(request.extensions),
        )
        for index, block in enumerate(request.system):
            self._context_block("system", index, block)
        for message_index, message in enumerate(request.messages):
            self._write(
                "message_start",
                message_index=message_index,
                role=message.role,
                extensions=thaw_value(message.extensions),
            )
            for block_index, block in enumerate(message.content):
                self._context_block(
                    "message",
                    block_index,
                    block,
                    message_index=message_index,
                )
        for index, tool in enumerate(request.tools):
            self._context_block("tool", index, tool)
        self._write("request_end")

    def route(self, value: dict[str, Any]) -> None:
        self._write("route", route=value)

    def attempt_start(
        self,
        *,
        phase: str | None,
        role: str | None,
        selected_model: str | None,
        context_changes: list[dict[str, Any]],
    ) -> None:
        self._write(
            "attempt_start",
            phase=phase,
            role=role,
            selected_model=selected_model,
        )
        for index, original in enumerate(context_changes):
            change = dict(original)
            text = change.pop("text", None)
            if not isinstance(text, str) or len(text) <= _TEXT_CHUNK_SIZE:
                if text is not None:
                    change["text"] = text
                self._write(
                    "context_change",
                    phase=phase,
                    index=index,
                    change=change,
                )
                continue
            chunks = self._chunks(text)
            for chunk_index, chunk in enumerate(chunks):
                self._write(
                    "context_change",
                    phase=phase,
                    index=index,
                    change=change,
                    chunk_index=chunk_index,
                    chunk_count=len(chunks),
                    text=chunk,
                )

    def attempt_end(self, **data: Any) -> None:
        output = data.pop("output_text", None)
        self._write("attempt_end", **data)
        if not isinstance(output, str) or not output:
            return
        chunks = self._chunks(output)
        for index, chunk in enumerate(chunks):
            self._write(
                "attempt_output",
                phase=data.get("phase"),
                chunk_index=index,
                chunk_count=len(chunks),
                text=chunk,
            )

    def run_end(self, *, error: str | None, cancelled: bool) -> None:
        self._write("run_end", error=error, cancelled=cancelled)
