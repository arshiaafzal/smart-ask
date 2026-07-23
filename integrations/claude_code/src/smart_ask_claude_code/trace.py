"""Incremental content traces for conversation-native engine invocations."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
import time
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

_SESSION_SCHEMA = "smart-ask.trace-session-index/v3"
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


def _defined_values(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _inline_value(value: Any) -> str:
    if isinstance(value, str):
        if value and all(
            character.isalnum() or character in "._-/:@"
            for character in value
        ):
            return value
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _fields(**values: Any) -> str:
    return " ".join(
        f"{key}={_inline_value(value)}"
        for key, value in values.items()
        if value is not None
    )


def _clock_timestamp() -> str:
    return datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]


def _log_line(level: str, component: str, message: str) -> str:
    return f"{_clock_timestamp()} {level:<5} {component:<10} {message}".rstrip()


def _block_kind(block: dict[str, Any]) -> tuple[str, str]:
    kind = str(block.get("type", "output"))
    if kind == "thinking":
        return "DEBUG", "thinking"
    if kind == "text":
        return "INFO", "output"
    return "INFO", kind


class _TextLogSink:
    def __init__(self, path: Path):
        self.path = path
        descriptor = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        os.chmod(path, 0o600)
        self._file = os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)
        self._closed = False

    def write_line(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("trace log is closed")
        self._file.write(text.rstrip("\n") + "\n")
        self._file.flush()

    def write_continuation(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("trace log is closed")
        lines = text.splitlines() or [""]
        for line in lines:
            self._file.write(f"  {line}\n")
        self._file.flush()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._file.close()


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
        self._sinks: dict[str, _TextLogSink] = {}
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
        self._calls: dict[tuple[str, str], dict[str, Any]] = {}
        self._run_usage: dict[str, dict[str, int | float]] = {}
        self._model_errors: dict[tuple[str, str], tuple[str, str]] = {}
        self._blocks: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._closed = False
        self._write_index_locked()

    def _write_locked(
        self,
        current_run_id: str,
        level: str,
        component: str,
        message: str,
        **values: Any,
    ) -> None:
        if self._closed:
            raise RuntimeError("trace session is closed")
        try:
            sink = self._sinks[current_run_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown trace run: {current_run_id}") from exc
        fields = _fields(**values)
        rendered = message if not fields else f"{message} {fields}"
        sink.write_line(_log_line(level, component, rendered))

    def _write_content_locked(
        self,
        run_id: str,
        level: str,
        component: str,
        label: str,
        value: str,
        **metadata: Any,
    ) -> None:
        if "\n" not in value and len(value) <= 200:
            self._write_locked(
                run_id,
                level,
                component,
                f"{label}={json.dumps(value, ensure_ascii=False)}",
                **metadata,
            )
            return
        self._write_locked(
            run_id,
            level,
            component,
            f"{label} begin",
            chars=len(value),
            **metadata,
        )
        self._sinks[run_id].write_continuation(value)
        self._write_locked(run_id, level, component, f"{label} end")

    def _write_value_locked(
        self,
        run_id: str,
        component: str,
        label: str,
        value: Any,
    ) -> None:
        if isinstance(value, str):
            self._write_content_locked(
                run_id,
                "DEBUG",
                component,
                label,
                value,
            )
            return
        compact = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rendered = compact if len(compact) <= 200 else json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        self._write_content_locked(
            run_id,
            "DEBUG",
            component,
            label,
            rendered,
        )

    def _write_blocks_locked(
        self,
        run_id: str,
        component: str,
        prefix: str,
        blocks: list[dict[str, Any]],
    ) -> None:
        for index, block in enumerate(blocks, 1):
            kind = str(block.get("type", "content"))
            if len(blocks) == 1:
                label = prefix if kind == "text" else f"{prefix}.{kind}"
            else:
                label = f"{prefix}.{kind}[{index}]"
            value: Any = block.get("text")
            source = "text"
            if value is None:
                value = block.get("thinking")
                source = "thinking"
            if value is None:
                if "content" in block:
                    value = block["content"]
                    source = "content"
                else:
                    value = block.get("data")
                    source = "data"
            if source == "data" and isinstance(value, str):
                digest = sha256(value.encode("utf-8")).hexdigest()[:16]
                value = f"<data chars={len(value)} sha256={digest}…>"
            if value is not None:
                self._write_value_locked(run_id, component, label, value)
            metadata = {
                key: item
                for key, item in block.items()
                if key not in ("type", "text", "thinking", "content", "data")
            }
            if metadata:
                self._write_value_locked(
                    run_id,
                    component,
                    f"{label}.metadata",
                    metadata,
                )

    def _write_conversation_locked(
        self,
        run_id: str,
        conversation: dict[str, Any],
        *,
        component: str = "input",
    ) -> None:
        messages = conversation.get("messages", [])
        system = conversation.get("system", [])
        tools = conversation.get("tools", [])
        self._write_locked(
            run_id,
            "DEBUG",
            component,
            "context",
            messages=len(messages),
            roles=[message.get("role", "unknown") for message in messages],
            content_blocks=[len(message.get("content", [])) for message in messages],
            system_blocks=len(system),
            tools=len(tools),
        )
        for index, message in enumerate(messages, 1):
            role = str(message.get("role", "unknown"))
            prefix = role if len(messages) == 1 else f"{role}[{index}]"
            self._write_blocks_locked(
                run_id,
                component,
                prefix,
                message.get("content", []),
            )
            if message.get("extensions"):
                self._write_value_locked(
                    run_id,
                    component,
                    f"{prefix}.extensions",
                    message["extensions"],
                )
        self._write_blocks_locked(run_id, component, "system", system)
        if tools:
            self._write_locked(
                run_id,
                "DEBUG",
                component,
                "tools",
                names=[tool.get("name", "unknown") for tool in tools],
            )
            for index, tool in enumerate(tools, 1):
                name = str(tool.get("name", "unknown"))
                self._write_locked(
                    run_id,
                    "DEBUG",
                    component,
                    "tool",
                    index=index,
                    name=name,
                )
                for key, value in tool.items():
                    if key == "name":
                        continue
                    self._write_value_locked(
                        run_id,
                        component,
                        f"tool[{index}].{key}",
                        value,
                    )
        parameters = conversation.get("parameters")
        if parameters:
            self._write_locked(
                run_id,
                "DEBUG",
                component,
                "parameters",
                **parameters,
            )
        extensions = conversation.get("extensions")
        if extensions:
            self._write_value_locked(
                run_id,
                component,
                "extensions",
                extensions,
            )

    def _call_component(self, run_id: str, call_id: str) -> str:
        state = self._calls.get((run_id, call_id))
        if state is None:
            return "model"
        return str(state["component"])

    def _record_usage_locked(
        self,
        run_id: str,
        call_id: str,
        usage: dict[str, Any],
    ) -> None:
        state = self._calls.setdefault((run_id, call_id), {
            "component": "model",
            "planned_at": time.perf_counter(),
            "usage": {},
        })
        previous = state["usage"]
        totals = self._run_usage.setdefault(run_id, {})
        for key, value in usage.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                previous[key] = value
                continue
            old_value = previous.get(key, 0)
            if not isinstance(old_value, (int, float)) or isinstance(old_value, bool):
                old_value = 0
            totals[key] = totals.get(key, 0) + value - old_value
            previous[key] = value

    @staticmethod
    def _usage_fields(usage: dict[str, Any]) -> dict[str, Any]:
        result = _defined_values(
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
        )
        for source, target in (
            ("reasoning_tokens", "reasoning_tokens"),
            ("cache_read_tokens", "cache_read_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, (int, float)) and value:
                result[target] = value
        cost = usage.get("provider_cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            result["cost_usd"] = cost
        if usage.get("complete") is False:
            result["usage_complete"] = False
        return result

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
            filename = f"{ordinal:03d}-{run_id[:8]}.log"
            sink = _TextLogSink(self.directory / filename)
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
            self._run_usage[run_id] = {}
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
            self._write_locked(
                run_id,
                "INFO",
                "run",
                "started",
                strategy=metadata.strategy_name,
            )
            trace_metadata = _defined_values(
                session=metadata.session_id,
                agent=metadata.agent_id,
                parent_agent=metadata.parent_agent_id,
                request=metadata.request_id,
            )
            if trace_metadata:
                self._write_locked(
                    run_id,
                    "DEBUG",
                    "run",
                    "context",
                    **trace_metadata,
                )
            self._write_conversation_locked(
                run_id,
                _full_conversation_value(conversation_value),
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
            conversation_payload = _conversation_payload(
                conversation,
                self._run_conversations[run_id],
            )
            component = spec.role or "model"
            self._calls[(run_id, call_id)] = {
                "component": component,
                "planned_at": time.perf_counter(),
                "usage": {},
            }
            replacement = conversation_payload.get("replace", {})
            context = (
                "run_input"
                if conversation_payload.get("conversation_ref") == "run_input"
                else "custom"
            )
            self._write_locked(
                run_id,
                "DEBUG",
                component,
                "planned",
                call=call_id,
                phase=spec.phase,
                profile=spec.profile_id,
                target=spec.target_id,
                context=context,
                changed=list(replacement) or None,
                caused_by=caused_by,
            )
            for name, value in replacement.items():
                if name == "parameters" and isinstance(value, dict):
                    self._write_locked(
                        run_id,
                        "DEBUG",
                        component,
                        "context.parameters",
                        **value,
                    )
                else:
                    self._write_value_locked(
                        run_id,
                        component,
                        f"context.{name}",
                        value,
                    )
            custom = conversation_payload.get("conversation")
            if isinstance(custom, dict):
                self._write_conversation_locked(
                    run_id,
                    custom,
                    component=f"{component}.input",
                )

    def decision_recorded(
        self,
        run_id: str,
        decision: DecisionRecord,
    ) -> None:
        with self._lock:
            self._write_locked(
                run_id,
                "INFO",
                "router",
                "decided",
                gate=decision.gate,
                outcome=decision.outcome,
                selected_profile=decision.selected_profile_id,
                reason=decision.reason_code,
                evidence=list(decision.evidence_call_ids) or None,
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
            component = self._call_component(run_id, call_id)
            self._write_locked(
                run_id,
                "ERROR",
                component,
                "failed",
                call=call_id,
                type=error_type,
                error=message[:4096],
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
                self._write_locked(
                    run_id,
                    "ERROR",
                    "run",
                    "failed",
                    type=error_type,
                    error=message[:4096],
                )
            else:
                self._write_locked(
                    run_id,
                    "ERROR",
                    "run",
                    "failed",
                    type=error_type,
                    caused_by=matching_call,
                )

    def model_event(
        self,
        run_id: str,
        call_id: str,
        event: ConversationEvent,
    ) -> None:
        with self._lock:
            component = self._call_component(run_id, call_id)
            state = self._calls.setdefault((run_id, call_id), {
                "component": component,
                "planned_at": time.perf_counter(),
                "usage": {},
            })
            if event.kind == "message_start":
                data = {
                    key: value
                    for key, value in thaw_value(event.data).items()
                    if value is not None
                }
                if data.get("selected_model") == data.get("model"):
                    data.pop("selected_model", None)
                state["started_at"] = time.perf_counter()
                self._write_locked(
                    run_id,
                    "INFO",
                    component,
                    "started",
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
                usage = {
                    key: value
                    for key, value in thaw_value(event.data).items()
                    if value is not None
                }
                self._record_usage_locked(run_id, call_id, usage)
                return
            if event.kind == "message_delta":
                data = {
                    key: value
                    for key, value in thaw_value(event.data).items()
                    if value is not None
                }
                started_at = state.get("started_at", state["planned_at"])
                duration_ms = (time.perf_counter() - started_at) * 1000
                self._write_locked(
                    run_id,
                    "INFO",
                    component,
                    "finished",
                    stop=data.pop("stop_reason", None),
                    **self._usage_fields(state["usage"]),
                    duration_ms=round(duration_ms, 1),
                    **data,
                )
                return
            if event.kind == "error":
                data = {
                    key: value
                    for key, value in thaw_value(event.data).items()
                    if value is not None
                }
                self._write_locked(
                    run_id,
                    "ERROR",
                    component,
                    "provider_error",
                    **data,
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
        component = self._call_component(run_id, call_id)
        level, label = _block_kind(block["block"])
        metadata = {
            key: value
            for key, value in block["block"].items()
            if key != "type" and value is not None
        }
        if not block["streaming"]:
            self._write_locked(
                run_id,
                level,
                component,
                f"{label} begin",
                **metadata,
            )
            block["streaming"] = True
        if block.get("text"):
            self._sinks[run_id].write_continuation(block["text"])
            block["text"] = ""
        if block.get("json"):
            self._sinks[run_id].write_continuation(block["json"])
            block["json"] = ""
        if "arguments" in block:
            rendered = json.dumps(
                block.pop("arguments"),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            self._sinks[run_id].write_continuation(rendered)

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
        component = self._call_component(run_id, call_id)
        level, label = _block_kind(block["block"])
        if block["streaming"]:
            self._flush_block(run_id, call_id, index, block)
            self._write_locked(
                run_id,
                level,
                component,
                f"{label} end",
                complete=False if not complete else None,
            )
            return
        metadata = {
            key: value
            for key, value in block["block"].items()
            if key != "type" and value is not None
        }
        value = block.get("text") or block.get("json")
        if value in (None, "") and "arguments" in block:
            value = json.dumps(
                block["arguments"],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if isinstance(value, str):
            self._write_content_locked(
                run_id,
                level,
                component,
                label,
                value,
                **metadata,
                complete=False if not complete else None,
            )
            return
        self._write_locked(
            run_id,
            level,
            component,
            label,
            **metadata,
            complete=False if not complete else None,
        )

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
            usage = self._run_usage.get(record.run_id, {})
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = (
                    usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0)
                )
            call_count = sum(
                1 for run_id, _call_id in self._calls if run_id == record.run_id
            )
            level = {
                "completed": "INFO",
                "cancelled": "WARN",
            }.get(record.status, "ERROR")
            self._write_locked(
                record.run_id,
                level,
                "run",
                record.status,
                calls=call_count,
                tokens=total_tokens,
                cost_usd=usage.get("provider_cost_usd"),
                duration_ms=round(record.duration_ms, 1),
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
            self._run_usage.pop(record.run_id, None)
            for key in [key for key in self._calls if key[0] == record.run_id]:
                self._calls.pop(key, None)
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
                self._write_locked(
                    run_id,
                    "WARN",
                    "run",
                    "trace_closed",
                    status="incomplete",
                )
                self._invocations[run_id]["status"] = "incomplete"
                sink.close()
                self._run_conversations.pop(run_id, None)
            self._sinks.clear()
            self._calls.clear()
            self._run_usage.clear()
            self._model_errors.clear()
            self._write_index_locked()
            self._closed = True
