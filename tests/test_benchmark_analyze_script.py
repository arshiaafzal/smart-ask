import json
from pathlib import Path
import tempfile
import unittest

from benchmark.analyze import analyze_file


class BenchmarkAnalyzeScriptTests(unittest.TestCase):
    def test_terminal_handoff_counts_the_visible_easy_response_once(self):
        record = {
            "run": {
                "schema": "smart-ask.run/v2",
                "final_call_id": "call-1",
                "decisions": [
                    {
                        "gate": "terminal_handoff",
                        "outcome": "attempt",
                        "selected_profile_id": "easy",
                    },
                    {
                        "gate": "terminal_handoff",
                        "outcome": "accept_easy",
                        "selected_profile_id": "easy",
                    },
                ],
                "model_calls": [{
                    "call_id": "call-1",
                    "profile_id": "easy",
                    "role": "finalizer",
                }],
                "provider_requests": [{
                    "call_id": "call-1",
                    "actual_model": "claude-sonnet-4-6",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "provider_cost_usd": 0.001,
                }],
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "metrics.jsonl")
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            summary = analyze_file(path)

        self.assertEqual(summary["routing"], {
            "total_turns": 1,
            "easy_turns": 1,
            "hard_turns": 0,
            "easy_pct": 100.0,
        })
        self.assertIn("claude-sonnet (finalizer)", summary["model_breakdown"])


if __name__ == "__main__":
    unittest.main()
