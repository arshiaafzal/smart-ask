import unittest

from smart_ask.benchmarks.counterfactual import evaluate_counterfactual_routing
from smart_ask.benchmarks.routing_analysis import analyze_routing


def record(strategy, task, profile, passed, score, cost):
    return {
        "strategy_id": strategy,
        "task_id": task,
        "decisions": [{
            "decision_id": "decision-1",
            "gate": "difficulty",
            "outcome": "easy" if profile == "easy" else "hard",
            "selected_profile_id": profile,
        }],
        "model_calls": [{
            "call_id": "call-1",
            "profile_id": profile,
            "target_id": profile + "-target",
            "selected_model": profile + "/model",
            "role": "writer",
            "caused_by_decision_id": "decision-1",
            "provider_request_ids": ["request-1"],
        }],
        "provider_requests": [{
            "provider_request_id": "request-1",
            "call_id": "call-1",
            "target_id": profile + "-target",
            "input_tokens": 10,
            "output_tokens": 2,
            "provider_cost_usd": cost,
        }],
        "final_call": "call-1",
        "evaluation": {"passed": passed, "score": score},
    }


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_routing_funnel_is_derived_from_decisions_and_calls(self):
        report = analyze_routing([
            record("routed", "one", "easy", True, 1.0, 0.1),
            record("routed", "two", "hard", True, 1.0, 1.0),
        ])

        self.assertEqual(report["transitions"], {
            "easy → response": 1,
            "hard → response": 1,
            "start → easy": 1,
            "start → hard": 1,
        })
        self.assertEqual(report["tokens_by_profile"], {"easy": 12, "hard": 12})
        self.assertEqual(report["tokens_by_transition"], {
            "start → easy": 12,
            "start → hard": 12,
        })

    def test_counterfactual_regret_uses_matched_baselines(self):
        records = [
            record("routed", "one", "hard", True, 1.0, 1.0),
            record("cheap", "one", "easy", True, 1.0, 0.1),
            record("hard", "one", "hard", True, 1.0, 1.0),
        ]
        report = evaluate_counterfactual_routing(
            records,
            routed_strategy="routed",
            cheap_strategy="cheap",
            hard_strategy="hard",
        )

        self.assertEqual(report["unnecessary_expensive"], 1)
        self.assertAlmostEqual(report["cost_regret_usd"], 0.9)
        self.assertEqual(report["cheap_opportunity_capture"], 0.0)


if __name__ == "__main__":
    unittest.main()
