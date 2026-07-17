"""
Offline routing quality tests — zero API calls.

Covers:
  - _project_tool_result projection logic (sync unit tests)
  - Full DifficultyStrategyMethod pipeline with injected classifier labels
  - Fallback behaviour on classifier error
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from smart_ask.conversation.domain import ConversationMessage
from smart_ask.conversation.model import (
    Conversation,
    ConversationEvent,
    ModelCallResult,
    RunMetadata,
)
from smart_ask.methods.strategies import (
    DifficultyStrategyMethod,
    ModelProfile,
    RequestTransform,
    RoutingInputError,
    StructuredDifficultyClassifier,
)

# ── Shared helpers (mirrors test_conversation_strategy_methods.py) ─────────────

_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "smart_ask/resources/prompts/agentic-difficulty-v1.txt"
)
_PROMPT = _PROMPT_PATH.read_text()

_CLF_PROFILE = ModelProfile(
    "classifier",
    "vendor-classifier",
    RequestTransform(parameters={"temperature": 0}),
)
_EASY = ModelProfile(
    "easy",
    "vendor-easy",
    RequestTransform(parameters={"max_tokens": 8192}),
)
_HARD = ModelProfile(
    "hard",
    "vendor-hard",
    RequestTransform(parameters={"max_tokens": 8192}),
)


def _result(call_id: str, text: str, *, stop_reason: str = "stop") -> ModelCallResult:
    return ModelCallResult(
        call_id=call_id,
        events=(ConversationEvent("message_stop", {"stop_reason": stop_reason}),),
        selected_model="selected/model",
        actual_model="actual/model",
        text=text,
        stop_reason=stop_reason,
        input_tokens=10,
        output_tokens=3,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        provider_cost_usd=None,
        tool_call_count=0,
        stream_complete=True,
        output_status="usable",
        duration_ms=1.0,
    )


class _FakeRunScope:
    def __init__(self, buffered=()):
        self.buffered = list(buffered)
        self.metadata = RunMetadata("strategy", "digest", session_id="session-1")
        self.calls = []
        self.decisions = []
        self.live = []
        self.success_operations = []

    async def call_buffered(self, spec, *, caused_by=None):
        self.calls.append(spec)
        value = self.buffered.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def defer_success(self, op):
        self.success_operations.append(op)

    def record_decision(self, draft):
        self.decisions.append(draft)
        return f"decision-{len(self.decisions)}"

    def plan_live(self, spec, *, caused_by):
        self.live.append(spec)
        return object()

    def plan_replay(self, value, *, accepted_by):
        return object()


def _make_classifier(*, continuation="classify_tool_result", fallback="easy"):
    return StructuredDifficultyClassifier(
        profile=_CLF_PROFILE,
        prompt=_PROMPT,
        projection="latest_user_text",
        continuation=continuation,
        fallback=fallback,
        max_prompt_chars=2000,
        parameters={"max_tokens": 20},
    )


def _make_method(*, continuation="classify_tool_result", fallback="easy"):
    return DifficultyStrategyMethod(
        classifier=_make_classifier(continuation=continuation, fallback=fallback),
        easy=_EASY,
        hard=_HARD,
        route_memory=None,
    )


def _tool_result_conv(content: str, *, first_content: str | None = None) -> Conversation:
    """Conversation ending with tool_result block(s)."""
    msgs: list[ConversationMessage] = [
        ConversationMessage(
            role="user",
            content=({"type": "text", "text": "fix bugs in impl.py"},),
        ),
        ConversationMessage(
            role="assistant",
            content=({"type": "tool_call", "id": "call-1", "name": "bash", "arguments": {}},),
        ),
    ]
    if first_content is not None:
        # Two user messages with tool_results (look-back window test).
        msgs.append(ConversationMessage(
            role="user",
            content=({"type": "tool_result", "tool_call_id": "call-1", "content": first_content},),
        ))
        msgs.append(ConversationMessage(
            role="assistant",
            content=({"type": "tool_call", "id": "call-2", "name": "read", "arguments": {}},),
        ))
        msgs.append(ConversationMessage(
            role="user",
            content=({"type": "tool_result", "tool_call_id": "call-2", "content": content},),
        ))
    else:
        msgs.append(ConversationMessage(
            role="user",
            content=({"type": "tool_result", "tool_call_id": "call-1", "content": content},),
        ))
    return Conversation(
        system=({"type": "text", "text": "system"},),
        messages=tuple(msgs),
        tools=({"name": "bash", "input_schema": {"type": "object"}},),
        parameters={},
        extensions={},
    )


def _text_only_conv(text: str = "fix the bug in impl.py") -> Conversation:
    """Conversation ending with a plain user text message."""
    return Conversation(
        system=({"type": "text", "text": "system"},),
        messages=(
            ConversationMessage(
                role="user",
                content=({"type": "text", "text": text},),
            ),
        ),
        tools=(),
        parameters={},
        extensions={},
    )


# ── Projection unit tests (sync) ──────────────────────────────────────────────

class ProjectionTests(unittest.TestCase):
    """Unit-test _project_tool_result without any model call."""

    def _clf(self) -> StructuredDifficultyClassifier:
        return _make_classifier()

    def test_single_tool_result_text_is_extracted(self):
        conv = _tool_result_conv("output line 1\noutput line 2")
        projected = self._clf()._project_tool_result(conv)
        text = projected.messages[-1].content[0]["text"]
        self.assertIn("output line 1", text)
        self.assertIn("output line 2", text)

    def test_long_output_tail_is_taken_within_budget(self):
        long_output = "A" * 3000
        conv = _tool_result_conv(long_output)
        projected = self._clf()._project_tool_result(conv)
        text = projected.messages[-1].content[0]["text"]
        self.assertLessEqual(len(text), 2000)
        # Tail was taken, so the last chars are the end of the long output.
        self.assertTrue(text.endswith("A" * 200))

    def test_two_tool_results_combined_with_separator(self):
        conv = _tool_result_conv("second result", first_content="first result")
        projected = self._clf()._project_tool_result(conv)
        text = projected.messages[-1].content[0]["text"]
        self.assertIn("first result", text)
        self.assertIn("second result", text)
        self.assertIn("---", text)

    def test_no_tool_result_raises_routing_input_error(self):
        conv = _text_only_conv("fix the bug")
        with self.assertRaises(RoutingInputError):
            self._clf()._project_tool_result(conv)


# ── Classifier routing tests (async, injected labels) ─────────────────────────

class ClassifierRoutingTests(unittest.IsolatedAsyncioTestCase):
    """Full routing pipeline with FakeRunScope-injected labels."""

    async def _route(self, conv: Conversation, label: str) -> str:
        run = _FakeRunScope([_result("c", f'{{"d":"{label}"}}')])
        await _make_method().respond(conv, run)
        return run.live[0].profile_id

    # ── Text turns (projection = latest_user_text path) ──

    async def test_user_fix_text_routes_hard(self):
        self.assertEqual(
            await self._route(_text_only_conv("fix all bugs in impl.py"), "hard"),
            "hard",
        )

    async def test_glob_result_routes_easy(self):
        self.assertEqual(
            await self._route(_tool_result_conv("impl.py\ntest_impl.py"), "easy"),
            "easy",
        )

    async def test_source_code_read_routes_hard(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv("class MyClass:\n    def __init__(self):\n        pass\n"),
                "hard",
            ),
            "hard",
        )

    async def test_edit_success_routes_easy(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv("The file has been updated successfully."),
                "easy",
            ),
            "easy",
        )

    async def test_file_created_routes_easy(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv("File created successfully."),
                "easy",
            ),
            "easy",
        )

    async def test_pytest_all_pass_routes_easy(self):
        self.assertEqual(
            await self._route(_tool_result_conv("5 passed in 0.04s"), "easy"),
            "easy",
        )

    async def test_unittest_ok_routes_easy(self):
        self.assertEqual(
            await self._route(_tool_result_conv("Ran 5 tests in 0.02s\n\nOK"), "easy"),
            "easy",
        )

    async def test_pytest_failure_routes_hard(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv(
                    "FAILED test_impl.py::test_foo - AssertionError\n1 failed in 0.01s"
                ),
                "hard",
            ),
            "hard",
        )

    async def test_assertion_error_routes_hard(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv("E   AssertionError: expected 5 got 3"),
                "hard",
            ),
            "hard",
        )

    async def test_traceback_routes_hard(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv(
                    "Traceback (most recent call last):\n  File test.py line 10\nValueError"
                ),
                "hard",
            ),
            "hard",
        )

    async def test_type_error_keyword_routes_hard(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv("TypeError: unexpected keyword argument 'store_cv_values'"),
                "hard",
            ),
            "hard",
        )

    async def test_edit_fail_string_not_found_routes_hard(self):
        self.assertEqual(
            await self._route(
                _tool_result_conv("String to replace not found in file."),
                "hard",
            ),
            "hard",
        )

    async def test_empty_tool_result_falls_back_to_easy(self):
        # Empty content → classifier call still made but may error → fallback = easy.
        conv = _tool_result_conv("")
        run = _FakeRunScope([RuntimeError("classifier unavailable")])
        await _make_method(fallback="easy").respond(conv, run)
        self.assertEqual(run.live[0].profile_id, "easy")


# ── End-to-end DifficultyStrategyMethod tests ─────────────────────────────────

class EndToEndMethodTests(unittest.IsolatedAsyncioTestCase):
    """DifficultyStrategyMethod wires projection + label → profile_id."""

    async def test_test_pass_routes_to_easy(self):
        conv = _tool_result_conv("5 passed in 0.04s")
        run = _FakeRunScope([_result("c", '{"d":"easy"}')])
        await _make_method().respond(conv, run)
        self.assertEqual(run.live[0].profile_id, "easy")

    async def test_test_fail_routes_to_hard(self):
        conv = _tool_result_conv(
            "FAILED test_impl.py::test_upload - AssertionError: 0o600 != 0o644\n"
            "1 failed in 0.02s"
        )
        run = _FakeRunScope([_result("c", '{"d":"hard"}')])
        await _make_method().respond(conv, run)
        self.assertEqual(run.live[0].profile_id, "hard")

    async def test_classifier_api_error_falls_back_to_easy(self):
        conv = _tool_result_conv("5 passed in 0.04s")
        run = _FakeRunScope([RuntimeError("connection timeout")])
        await _make_method(fallback="easy").respond(conv, run)
        self.assertEqual(run.live[0].profile_id, "easy")


if __name__ == "__main__":
    unittest.main()
