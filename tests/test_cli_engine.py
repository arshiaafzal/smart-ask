import asyncio
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from smart_ask import _terminal, cli
from smart_ask.conversation import (
    ConversationEvent,
    DecisionRecord,
    ModelCallRecord,
    ProviderRequestRecord,
    RunRecord,
)
from smart_ask.conversation.model import DecisionId
from smart_ask.strategy import load_strategy


def _events(text: str) -> tuple[ConversationEvent, ...]:
    return (
        ConversationEvent("message_start", {"model": "provider/model"}),
        ConversationEvent(
            "content_start",
            {"index": 0, "block": {"type": "text"}},
        ),
        ConversationEvent(
            "content_delta",
            {"index": 0, "delta": {"type": "text", "text": text}},
        ),
        ConversationEvent("content_stop", {"index": 0}),
        ConversationEvent(
            "usage",
            {"input_tokens": 2, "output_tokens": 1},
        ),
        ConversationEvent("message_delta", {"stop_reason": "stop"}),
        ConversationEvent("message_stop"),
    )


def _record(metadata) -> RunRecord:
    decision_id = DecisionId("decision-1")
    return RunRecord(
        run_id=f"run-{metadata.request_id}",
        metadata=metadata,
        status="completed",
        started_at=1.0,
        duration_ms=2.0,
        decisions=(DecisionRecord(
            decision_id=decision_id,
            gate="fixed",
            outcome="selected",
            reason_code="configured",
            selected_profile_id="primary",
            evidence_call_ids=(),
            sequence=1,
        ),),
        model_calls=(ModelCallRecord(
            call_id="call-1",
            sequence=1,
            profile_id="primary",
            target_id="target",
            selected_model="provider/model",
            role="generator",
            phase="fixed",
            caused_by_decision_id=decision_id,
            provider_request_ids=("provider-request-1",),
            status="completed",
        ),),
        provider_requests=(ProviderRequestRecord(
            provider_request_id="provider-request-1",
            call_id="call-1",
            sequence=1,
            status="completed",
            target_id="target",
            requested_max_output_tokens=100,
            selected_model="provider/model",
            actual_model="provider/model",
            input_tokens=2,
            output_tokens=1,
            visible_output_tokens=1,
            reasoning_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            provider_cost_usd=0.001,
            stop_reason="stop",
            stream_complete=True,
            tool_call_count=0,
            visible_text_chars=8,
            output_status="usable",
            time_to_first_output_ms=1.0,
            duration_ms=2.0,
        ),),
        final_call_id="call-1",
        final_decision_id=decision_id,
    )


class _Handle:
    def __init__(self, events, record):
        self._events = events
        self._record = record

    async def _stream(self):
        for event in self._events:
            yield event

    def events(self):
        return self._stream()

    async def result(self):
        return self._record


class _Engine:
    def __init__(self):
        self.conversations = []
        self.closed = False

    def start(self, conversation, metadata):
        self.conversations.append(conversation)
        number = len(self.conversations)
        return _Handle(_events(f"answer-{number}"), _record(metadata))

    async def aclose(self):
        self.closed = True


class ProductCliEngineTests(unittest.TestCase):
    def test_main_delegates_anthropic_gateway_subcommand(self):
        with patch("smart_ask.gateways.anthropic.cli.main") as gateway_main:
            cli.main([
                "gateway",
                "anthropic",
                "serve",
                "--config",
                "gateway.yaml",
            ])

        gateway_main.assert_called_once_with([
            "serve",
            "--config",
            "gateway.yaml",
        ])

    def test_repl_passes_complete_conversation_snapshots_to_engine(self):
        engine = _Engine()
        loaded = SimpleNamespace(
            config=SimpleNamespace(name="test-strategy"),
            digest="a" * 64,
        )
        output = StringIO()

        with patch("builtins.input", side_effect=["second", "/exit"]):
            with redirect_stdout(output):
                asyncio.run(cli._run_session(
                    engine=engine,
                    loaded_strategy=loaded,
                    initial_input="first",
                ))

        self.assertTrue(engine.closed)
        self.assertEqual(len(engine.conversations), 2)
        self.assertEqual(
            [message.role for message in engine.conversations[1].messages],
            ["user", "assistant", "user"],
        )
        prior_answer = engine.conversations[1].messages[1].content[0]
        self.assertEqual(prior_answer["text"], "answer-1")
        rendered = output.getvalue()
        self.assertIn("answer-1", rendered)
        self.assertIn("answer-2", rendered)
        self.assertIn("decision 1: fixed = selected", rendered)
        self.assertIn("call-1", rendered)
        self.assertIn("Session: 2 turns", rendered)

    def test_parser_no_longer_accepts_dry_run(self):
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                cli._parser().parse_args(["--dry-run", "hello"])

    def test_welcome_uses_v3_profiles_and_targets(self):
        config = load_strategy("builtin:product").config
        output = StringIO()
        with patch.object(_terminal, "_teaser"):
            with redirect_stdout(output):
                _terminal.show_welcome(config)

        rendered = output.getvalue()
        self.assertIn("Targets:", rendered)
        self.assertIn("easy ->", rendered)
        self.assertNotIn("--dry-run", rendered)


if __name__ == "__main__":
    unittest.main()
