"""Incremental content traces for conversation-native engine invocations."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from smart_ask.conversation.domain import ConversationEvent, thaw_value
from smart_ask.conversation.model import (
    Conversation,
    DecisionId,
    DecisionRecord,
    ModelCallSpec,
    RunMetadata,
    RunRecord,
)

from .metrics import JsonlSink


_INVOCATION_SCHEMA = "smart-ask.method-invocation-trace/v2"
_SESSION_SCHEMA = "smart-ask.trace-session-index/v2"
_CHUNK_CHARS = 512
_CONVERSATION_FIELDS = (
    "system",
    "messages",
    "tools",
    "parameters",
    "extensions",
)


def _trace_mapping(value: Any) -> dict[str, Any]:
    result = thaw_value(value)
    if not isinstance(result, dict):
        raise TypeError("trace values must be mappings")
    if result.get("extensions") == {}:
        result.pop("extensions")
    return result


def _conversation_value(conversation: Conversation) -> dict[str, Any]:
    return {
        "system": [_trace_mapping(block) for block in conversation.system],
        "messages": [
            {
                "role": message.role,
                "content": [
                    _trace_mapping(block) for block in message.content
                ],
                **(
                    {"extensions": thaw_value(message.extensions)}
                    if message.extensions
                    else {}
                ),
            }
            for message in conversation.messages
        ],
        "tools": [_trace_mapping(tool) for tool in conversation.tools],
        "parameters": thaw_value(conversation.parameters),
        "extensions": thaw_value(conversation.extensions),
    }


def _full_conversation_value(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key == "messages" or item not in ({}, [])
    }


def _conversation_payload(
    value: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    replacement = {
        key: value[key]
        for key in _CONVERSATION_FIELDS
        if value[key] != base[key]
    }
    referenced: dict[str, Any] = {"conversation_ref": "run_input"}
    if replacement:
        referenced["replace"] = replacement
    full = {"conversation": _full_conversation_value(value)}
    if len(json.dumps(referenced, separators=(",", ":"))) < len(
        json.dumps(full, separators=(",", ":"))
    ):
        return referenced
    return full


def _conversation_digest(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_value(metadata: RunMetadata) -> dict[str, Any]:
    value: dict[str, Any] = {
        "strategy_name": metadata.strategy_name,
        "strategy_digest": metadata.strategy_digest,
    }
    for name in ("session_id", "agent_id", "parent_agent_id", "request_id"):
        item = getattr(metadata, name)
        if item is not None:
            value[name] = item
    if metadata.extensions:
        value["extensions"] = thaw_value(metadata.extensions)
    return value


def _defined_values(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


class TraceSessionSink:
    """Write one indexed, private trace file per method invocation."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        if self.directory.exists():
            if not self.directory.is_dir():
                raise ValueError("trace directory path is not a directory")
            if any(self.directory.iterdir()):
                raise ValueError("trace directory must be empty")
        else:
            self.directory.mkdir(parents=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.index_path = self.directory / "session.json"
        self._lock = Lock()
        self._run_numbers: dict[str, int] = {}
        self._sinks: dict[str, JsonlSink] = {}
        self._invocations: dict[str, dict[str, Any]] = {}
        self._contexts: dict[str, dict[str, Any]] = {}
        self._context_ids: dict[
            tuple[str | None, str | None, str | None, str, str],
            str,
        ] = {}
        self._inputs: dict[str, dict[str, Any]] = {}
        self._input_ids: dict[str, str] = {}
        self._first_inputs: dict[tuple[str, str], int] = {}
        self._run_conversations: dict[str, dict[str, Any]] = {}
        self._model_errors: dict[tuple[str, str], tuple[str, str]] = {}
        self._blocks: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._closed = False
        self._write_index_locked()

    def _emit_locked(self, current_run_id: str, event: str, **values: Any) -> None:
        if self._closed:
            raise RuntimeError("trace session is closed")
        try:
            sink = self._sinks[current_run_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown trace run: {current_run_id}") from exc
        sink.write({"event": event, **values})

    def _write_index_locked(self) -> None:
        invocations = sorted(
            self._invocations.values(),
            key=lambda value: value["ordinal"],
        )
        payload = {
            "schema": _SESSION_SCHEMA,
            "contexts": list(self._contexts.values()),
            "inputs": list(self._inputs.values()),
            "invocation_count": len(invocations),
            "invocations": invocations,
        }
        temporary = self.directory / ".session.json.tmp"
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        os.replace(temporary, self.index_path)
        os.chmod(self.index_path, 0o600)

    def _context_id(self, metadata: RunMetadata) -> str:
        key = (
            metadata.session_id,
            metadata.agent_id,
            metadata.parent_agent_id,
            metadata.strategy_name,
            metadata.strategy_digest,
        )
        existing = self._context_ids.get(key)
        if existing is not None:
            return existing
        context_id = f"context-{len(self._contexts) + 1}"
        value: dict[str, Any] = {
            "id": context_id,
            "strategy_name": metadata.strategy_name,
            "strategy_digest": metadata.strategy_digest,
        }
        for name in ("session_id", "agent_id", "parent_agent_id"):
            item = getattr(metadata, name)
            if item is not None:
                value[name] = item
        self._context_ids[key] = context_id
        self._contexts[context_id] = value
        return context_id

    def _input_id(self, digest: str, ordinal: int) -> str:
        existing = self._input_ids.get(digest)
        if existing is not None:
            return existing
        input_id = f"input-{len(self._inputs) + 1}"
        self._input_ids[digest] = input_id
        self._inputs[input_id] = {
            "id": input_id,
            "sha256": digest,
            "first_ordinal": ordinal,
        }
        return input_id

    def run_started(
        self,
        run_id: str,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("trace session is closed")
            if run_id in self._run_numbers:
                raise RuntimeError(f"duplicate trace run: {run_id}")
            ordinal = len(self._run_numbers) + 1
            filename = f"{ordinal:03d}-{run_id[:8]}.jsonl"
            sink = JsonlSink(str(self.directory / filename))
            conversation_value = _conversation_value(conversation)
            digest = _conversation_digest(conversation_value)
            context_id = self._context_id(metadata)
            input_id = self._input_id(digest, ordinal)
            first_input = self._first_inputs.get((context_id, digest))
            if first_input is None:
                self._first_inputs[(context_id, digest)] = ordinal
            self._run_numbers[run_id] = ordinal
            self._sinks[run_id] = sink
            self._run_conversations[run_id] = conversation_value
            invocation = {
                "ordinal": ordinal,
                "run_id": run_id,
                "file": filename,
                "context": context_id,
                "input": input_id,
                "status": "running",
            }
            if metadata.request_id is not None:
                invocation["request_id"] = metadata.request_id
            if first_input is not None:
                invocation["same_input_as"] = first_input
            self._invocations[run_id] = invocation
            sink.write({"event": "trace_start", "schema": _INVOCATION_SCHEMA})
            self._emit_locked(
                run_id,
                "run_start",
                run_id=run_id,
                ordinal=ordinal,
                metadata=_metadata_value(metadata),
            )
            self._emit_locked(
                run_id,
                "conversation",
                conversation=_full_conversation_value(conversation_value),
            )
            self._write_index_locked()

    def model_call_planned(
        self,
        run_id: str,
        call_id: str,
        sequence: int,
        spec: ModelCallSpec,
        caused_by: DecisionId | None,
    ) -> None:
        with self._lock:
            conversation = _conversation_value(spec.conversation)
            self._emit_locked(
                run_id,
                "model_call",
                **{
                    "call_id": call_id,
                    "sequence": sequence,
                    "profile_id": spec.profile_id,
                    "target_id": spec.target_id,
                    "role": spec.role,
                    **_defined_values(
                        phase=spec.phase,
                        caused_by_decision_id=caused_by,
                    ),
                    **_conversation_payload(
                        conversation,
                        self._run_conversations[run_id],
                    ),
                },
            )

    def decision_recorded(
        self,
        run_id: str,
        decision: DecisionRecord,
    ) -> None:
        with self._lock:
            self._emit_locked(
                run_id,
                "decision",
                decision_id=decision.decision_id,
                sequence=decision.sequence,
                gate=decision.gate,
                outcome=decision.outcome,
                reason_code=decision.reason_code,
                evidence_call_ids=list(decision.evidence_call_ids),
                **_defined_values(
                    selected_profile_id=decision.selected_profile_id,
                ),
            )

    def model_failed(
        self,
        run_id: str,
        call_id: str,
        error_type: str,
        message: str,
    ) -> None:
        with self._lock:
            self._model_errors[(run_id, call_id)] = (error_type, message)
            self._emit_locked(
                run_id,
                "model_error",
                call_id=call_id,
                error_type=error_type,
                message=message[:4096],
            )

    def run_failed(
        self,
        run_id: str,
        error_type: str,
        message: str,
    ) -> None:
        with self._lock:
            matching_call = next(
                (
                    call_id
                    for (candidate_run, call_id), (_kind, candidate_message)
                    in reversed(tuple(self._model_errors.items()))
                    if candidate_run == run_id and candidate_message == message
                ),
                None,
            )
            if matching_call is None:
                self._emit_locked(
                    run_id,
                    "run_error",
                    error_type=error_type,
                    message=message[:4096],
                )
            else:
                self._emit_locked(
                    run_id,
                    "run_error",
                    error_type=error_type,
                    caused_by={
                        "event": "model_error",
                        "call_id": matching_call,
                    },
                )

    def model_event(
        self,
        run_id: str,
        call_id: str,
        event: ConversationEvent,
    ) -> None:
        with self._lock:
            if event.kind == "message_start":
                data = {
                    key: value
                    for key, value in thaw_value(event.data).items()
                    if value is not None
                }
                if data.get("selected_model") == data.get("model"):
                    data.pop("selected_model", None)
                self._emit_locked(
                    run_id,
                    "provider_start",
                    call_id=call_id,
                    **data,
                )
                return
            if event.kind == "content_start":
                index = event.data.get("index")
                if not isinstance(index, int):
                    return
                block = thaw_value(event.data.get("block", {}))
                self._blocks[(run_id, call_id, index)] = {
                    "block": block,
                    "text": "",
                    "json": "",
                    "streaming": False,
                }
                return
            if event.kind == "content_delta":
                self._content_delta(run_id, call_id, event)
                return
            if event.kind == "content_stop":
                index = event.data.get("index")
                if not isinstance(index, int):
                    return
                key = (run_id, call_id, index)
                self._finish_block(
                    run_id,
                    call_id,
                    index,
                    self._blocks.pop(key, None),
                    complete=True,
                )
                return
            if event.kind == "usage":
                self._emit_locked(
                    run_id,
                    "usage",
                    call_id=call_id,
                    **{
                        key: value
                        for key, value in thaw_value(event.data).items()
                        if value is not None
                    },
                )
                return
            if event.kind == "message_delta":
                self._emit_locked(
                    run_id,
                    "provider_stop",
                    call_id=call_id,
                    **{
                        key: value
                        for key, value in thaw_value(event.data).items()
                        if value is not None
                    },
                )
                return
            if event.kind == "error":
                self._emit_locked(
                    run_id,
                    "provider_error",
                    call_id=call_id,
                    **{
                        key: value
                        for key, value in thaw_value(event.data).items()
                        if value is not None
                    },
                )

    def _content_delta(
        self,
        run_id: str,
        call_id: str,
        event: ConversationEvent,
    ) -> None:
        index = event.data.get("index")
        delta = event.data.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            # Frozen mappings are Mapping, but thawing keeps this branch small
            # and makes all persisted values ordinary JSON containers.
            delta = thaw_value(delta)
        if not isinstance(index, int) or not isinstance(delta, dict):
            return
        key = (run_id, call_id, index)
        block = self._blocks.setdefault(key, {
            "block": {"type": delta.get("type", "unknown")},
            "text": "",
            "json": "",
            "streaming": False,
        })
        text = delta.get("text")
        if isinstance(text, str):
            block["text"] += text
        json_delta = delta.get("json")
        if isinstance(json_delta, str):
            block["json"] += json_delta
        if "arguments" in delta:
            block["arguments"] = thaw_value(delta["arguments"])
        if len(block["text"]) >= _CHUNK_CHARS or len(block["json"]) >= _CHUNK_CHARS:
            self._flush_block(run_id, call_id, index, block)

    def _flush_block(
        self,
        run_id: str,
        call_id: str,
        index: int,
        block: dict[str, Any] | None,
    ) -> None:
        if not block:
            return
        if not block["streaming"]:
            self._emit_locked(
                run_id,
                "model_output_start",
                call_id=call_id,
                index=index,
                block=block["block"],
            )
            block["streaming"] = True
        values: dict[str, Any] = {"call_id": call_id, "index": index}
        if block.get("text"):
            values["text"] = block["text"]
            block["text"] = ""
        if block.get("json"):
            values["json"] = block["json"]
            block["json"] = ""
        if "arguments" in block:
            values["arguments"] = block.pop("arguments")
        if len(values) > 2:
            self._emit_locked(run_id, "model_output_chunk", **values)

    def _finish_block(
        self,
        run_id: str,
        call_id: str,
        index: int,
        block: dict[str, Any] | None,
        *,
        complete: bool,
    ) -> None:
        if not block:
            return
        if block["streaming"]:
            self._flush_block(run_id, call_id, index, block)
            if complete:
                self._emit_locked(
                    run_id,
                    "model_output_end",
                    call_id=call_id,
                    index=index,
                )
            return
        values: dict[str, Any] = {
            "call_id": call_id,
            "index": index,
            "block": block["block"],
        }
        for name in ("text", "json", "arguments"):
            if block.get(name) not in (None, ""):
                values[name] = block[name]
        if not complete:
            values["complete"] = False
        self._emit_locked(run_id, "model_output", **values)

    def run_finished(self, record: RunRecord) -> None:
        with self._lock:
            for key in [key for key in self._blocks if key[0] == record.run_id]:
                _run_id, call_id, index = key
                self._finish_block(
                    record.run_id,
                    call_id,
                    index,
                    self._blocks.pop(key),
                    complete=False,
                )
            self._emit_locked(
                record.run_id,
                "run_end",
                status=record.status,
                duration_ms=record.duration_ms,
                **_defined_values(
                    final_call_id=record.final_call_id,
                    final_decision_id=record.final_decision_id,
                    error=record.error,
                ),
            )
            invocation = self._invocations[record.run_id]
            invocation.update({
                "status": record.status,
                "duration_ms": record.duration_ms,
                **_defined_values(
                    final_call_id=record.final_call_id,
                    final_decision_id=record.final_decision_id,
                    error=record.error,
                ),
            })
            self._sinks.pop(record.run_id).close()
            self._run_conversations.pop(record.run_id, None)
            for key in [key for key in self._model_errors if key[0] == record.run_id]:
                self._model_errors.pop(key, None)
            self._write_index_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for run_id, sink in tuple(self._sinks.items()):
                for key in [key for key in self._blocks if key[0] == run_id]:
                    _run_id, call_id, index = key
                    self._finish_block(
                        run_id,
                        call_id,
                        index,
                        self._blocks.pop(key),
                        complete=False,
                    )
                self._emit_locked(run_id, "trace_closed", status="incomplete")
                self._invocations[run_id]["status"] = "incomplete"
                sink.close()
                self._run_conversations.pop(run_id, None)
            self._sinks.clear()
            self._model_errors.clear()
            self._write_index_locked()
            self._closed = True
