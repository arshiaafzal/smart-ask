from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

from smart_ask.domain import ExecutionRequest, ModelResult
from smart_ask.executors import HermesExecutor, OpenRouterExecutor

from tests.helpers import FakeClient


def openrouter_response(
    *,
    content="answer",
    finish_reason="stop",
    native_finish_reason="end_turn",
    refusal=None,
    usage=None,
):
    return SimpleNamespace(
        model="provider/actual",
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            native_finish_reason=native_finish_reason,
            message=SimpleNamespace(content=content, refusal=refusal),
        )],
        usage=usage,
    )


class ModelResultEvidenceTests(unittest.TestCase):
    def test_status_is_derived_from_observable_response_evidence(self):
        usable = ModelResult("model", "answer")
        empty = ModelResult("model", " \n")
        truncated = ModelResult("model", "partial", finish_reason="length")
        refused = ModelResult(
            "model",
            "",
            finish_reason="refusal",
            refusal="I cannot help with that.",
        )

        self.assertEqual(usable.output_status, "usable")
        self.assertIs(usable.output_empty, False)
        self.assertEqual(empty.output_status, "empty")
        self.assertIs(empty.output_empty, True)
        self.assertEqual(truncated.output_status, "truncated")
        self.assertIs(truncated.output_empty, False)
        self.assertIs(truncated.max_tokens_reached, True)
        self.assertEqual(refused.output_status, "refused")
        self.assertIs(refused.output_empty, True)
        self.assertIs(refused.max_tokens_reached, False)
        self.assertIsNone(usable.max_tokens_reached)

    def test_unavailable_output_can_retain_known_finish_evidence(self):
        result = ModelResult(
            "model",
            "",
            raw_text=None,
            finish_reason="stop",
            output_status="unavailable",
        )

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.output_status, "unavailable")
        self.assertIsNone(result.output_empty)
        self.assertIs(result.max_tokens_reached, False)

    def test_contradictory_or_invalid_evidence_is_rejected(self):
        invalid_values = (
            ({"finish_reason": "done"}, "finish_reason"),
            ({"finish_reason": "stop", "output_status": "empty"}, "output_status"),
            ({"output_status": "unavailable"}, "output_status"),
            (
                {"finish_reason": "length", "max_tokens_reached": False},
                "max_tokens_reached",
            ),
            ({"reasoning_tokens": -1}, "reasoning_tokens"),
            ({"cached_input_tokens": True}, "cached_input_tokens"),
            ({"applied_max_tokens": 0}, "applied_max_tokens"),
            ({"provider_cost_usd": float("nan")}, "provider_cost_usd"),
            ({"visible_output_tokens": 0}, "visible_output_tokens"),
            ({"refusal": "  "}, "refusal"),
        )
        for fields, message in invalid_values:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, message):
                    ModelResult("model", "answer", **fields)

        with self.assertRaisesRegex(ValueError, "visible_output_tokens"):
            ModelResult("model", "", visible_output_tokens=1)

    def test_evidence_is_immutable(self):
        result = ModelResult("model", "answer", reasoning_tokens=3)

        with self.assertRaises(FrozenInstanceError):
            result.reasoning_tokens = 4


class OpenRouterResponseEvidenceTests(unittest.TestCase):
    def execute(self, response, *, max_tokens=40):
        return OpenRouterExecutor(
            FakeClient([response]),
            default_max_tokens=100,
            temperature=0,
        ).execute(ExecutionRequest(
            "requested/model",
            "prompt",
            "writer",
            max_tokens=max_tokens,
        ))

    def test_captures_finish_refusal_and_detailed_usage(self):
        result = self.execute(openrouter_response(
            usage={
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 7},
                "prompt_tokens_details": {
                    "cached_tokens": 11,
                    "cache_write_tokens": 5,
                },
                "cost": 0.0123,
            },
        ))

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.native_finish_reason, "end_turn")
        self.assertEqual(result.output_status, "usable")
        self.assertIsNone(result.refusal)
        self.assertEqual(result.applied_max_tokens, 40)
        self.assertIs(result.max_tokens_reached, False)
        self.assertEqual(result.visible_output_tokens, 13)
        self.assertEqual(result.reasoning_tokens, 7)
        self.assertEqual(result.cached_input_tokens, 11)
        self.assertEqual(result.cache_write_input_tokens, 5)
        self.assertEqual(result.provider_cost_usd, 0.0123)

    def test_supports_attribute_usage_and_normalizes_tool_calls(self):
        usage = SimpleNamespace(
            completion_tokens=6,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=2,
                cache_write_tokens=1,
            ),
        )
        result = self.execute(openrouter_response(
            content=None,
            finish_reason="tool_calls",
            native_finish_reason=None,
            usage=usage,
        ))

        self.assertEqual(result.finish_reason, "tool_call")
        self.assertEqual(result.native_finish_reason, "tool_calls")
        self.assertEqual(result.output_status, "empty")
        self.assertIsNone(result.visible_output_tokens)

    def test_supports_mapping_response_shapes(self):
        result = self.execute({
            "model": "provider/actual",
            "choices": [{
                "finish_reason": "stop",
                "native_finish_reason": "end_turn",
                "message": {"content": "answer", "refusal": None},
            }],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 1},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        })

        self.assertEqual(result.model, "provider/actual")
        self.assertEqual(result.visible_output_tokens, 2)
        self.assertEqual(result.cached_input_tokens, 4)

    def test_refusal_takes_precedence_over_provider_stop_reason(self):
        result = self.execute(openrouter_response(
            content=None,
            finish_reason="stop",
            native_finish_reason="stop",
            refusal="Request refused.",
        ))

        self.assertEqual(result.finish_reason, "refusal")
        self.assertEqual(result.native_finish_reason, "stop")
        self.assertEqual(result.output_status, "refused")
        self.assertEqual(result.refusal, "Request refused.")

    def test_length_marks_visible_output_as_truncated(self):
        result = self.execute(openrouter_response(
            content="partial",
            finish_reason="length",
            native_finish_reason="max_tokens",
        ))

        self.assertEqual(result.output_status, "truncated")
        self.assertIs(result.max_tokens_reached, True)

    def test_empty_tool_call_does_not_invent_visible_text_tokens(self):
        result = self.execute(openrouter_response(
            content=None,
            finish_reason="tool_calls",
            usage=SimpleNamespace(
                completion_tokens=12,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        ))

        self.assertEqual(result.finish_reason, "tool_call")
        self.assertIsNone(result.visible_output_tokens)

    def test_missing_detail_remains_unknown_instead_of_becoming_zero(self):
        result = self.execute(openrouter_response(
            usage=SimpleNamespace(completion_tokens=12),
        ))

        self.assertIsNone(result.reasoning_tokens)
        self.assertIsNone(result.visible_output_tokens)
        self.assertIsNone(result.cached_input_tokens)
        self.assertIsNone(result.cache_write_input_tokens)
        self.assertIsNone(result.provider_cost_usd)

    def test_unknown_provider_finish_reason_is_preserved_as_native_evidence(self):
        result = self.execute(openrouter_response(
            finish_reason="provider_new_reason",
            native_finish_reason=None,
        ))

        self.assertEqual(result.finish_reason, "unknown")
        self.assertEqual(result.native_finish_reason, "provider_new_reason")
        self.assertIsNone(result.max_tokens_reached)

    def test_rejects_impossible_or_malformed_usage_detail(self):
        malformed_usage = (
            {"completion_tokens": 4, "completion_tokens_details": {
                "reasoning_tokens": 5,
            }},
            {"completion_tokens": 4, "prompt_tokens_details": {
                "cached_tokens": -1,
            }},
            {
                "prompt_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
            {"completion_tokens": True},
            {"cost": -0.1},
            {"cost": True},
        )
        for usage in malformed_usage:
            with self.subTest(usage=usage):
                with self.assertRaises(ValueError):
                    self.execute(openrouter_response(usage=usage))


class HermesResponseEvidenceTests(unittest.TestCase):
    def test_unavailable_provider_evidence_stays_unknown(self):
        result = HermesExecutor(
            provider="openrouter",
            command="hermes",
            runner=lambda _command: SimpleNamespace(
                returncode=0,
                stdout="answer",
            ),
        ).execute(ExecutionRequest(
            "requested/model",
            "prompt",
            "writer",
            max_tokens=50,
        ))

        self.assertIsNone(result.model)
        self.assertEqual(result.finish_reason, "unknown")
        self.assertEqual(result.output_status, "usable")
        self.assertIsNone(result.native_finish_reason)
        self.assertIsNone(result.applied_max_tokens)
        self.assertIsNone(result.max_tokens_reached)
        self.assertIsNone(result.visible_output_tokens)
        self.assertIsNone(result.reasoning_tokens)
        self.assertIsNone(result.cached_input_tokens)
        self.assertIsNone(result.cache_write_input_tokens)
        self.assertIsNone(result.provider_cost_usd)

    def test_absent_stdout_is_unavailable_not_empty(self):
        result = HermesExecutor(
            provider="openrouter",
            command="hermes",
            runner=lambda _command: SimpleNamespace(
                returncode=0,
                stdout=None,
            ),
        ).execute(ExecutionRequest("model", "prompt", "writer"))

        self.assertEqual(result.text, "")
        self.assertIsNone(result.raw_text)
        self.assertEqual(result.output_status, "unavailable")
        self.assertIsNone(result.output_empty)

    def test_captured_empty_stdout_is_observed_empty(self):
        result = HermesExecutor(
            provider="openrouter",
            command="hermes",
            runner=lambda _command: SimpleNamespace(
                returncode=0,
                stdout="",
            ),
        ).execute(ExecutionRequest("model", "prompt", "writer"))

        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.output_status, "empty")
        self.assertIs(result.output_empty, True)


if __name__ == "__main__":
    unittest.main()
