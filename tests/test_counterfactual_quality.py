"""Counterfactual routing-quality diagnostics over canonical evidence."""

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from smart_ask.benchmarks.counterfactual import evaluate_counterfactual_routing
from smart_ask.strategy.loader import compute_strategy_digest
from smart_ask.strategy.schema import StrategyConfig


_TASK_IDS = ("t1", "t2", "t3", "t4", "t5")
_CHEAP_PASSES = {"t1": True, "t2": True, "t3": False, "t4": False, "t5": False}
_EXPENSIVE_PASSES = {
    "t1": True,
    "t2": True,
    "t3": True,
    "t4": True,
    "t5": False,
}
_ROUTED_PASSES = {"t1": True, "t2": True, "t3": False, "t4": True, "t5": False}
_ROUTED_PHASES = {
    "t1": ["initial-easy"],
    "t2": ["initial-hard"],
    "t3": ["initial-easy"],
    "t4": ["initial-easy", "escalation"],
    "t5": ["initial-hard"],
}


def _config(name, method):
    return StrategyConfig.model_validate({
        "schema_version": 2,
        "name": name,
        "method": method,
        "generation": {"type": "openrouter"},
    })


def _fixed_config(name, model, role, *, prompt_prefix=None, prompt_suffix=None):
    method = {
        "type": "fixed",
        "role": role,
        "model": {"model": model},
    }
    if prompt_prefix is not None:
        method["prompt_prefix"] = {"type": "inline", "text": prompt_prefix}
    if prompt_suffix is not None:
        method["prompt_suffix"] = {"type": "inline", "text": prompt_suffix}
    return _config(name, method)


def _routed_config():
    return _config("routed", {
        "type": "cascade",
        "classifier": {
            "type": "llm",
            "model": "classifier-model",
            "executor": {"type": "openrouter"},
            "prompt": {"type": "inline", "text": "classify"},
            "fallback": "raise",
        },
        "escalation": {
            "type": "marker",
            "marker": "ESCALATE",
            "self_check_suffix": {
                "type": "inline",
                "text": "Return ESCALATE when unsure.",
            },
            "escalation_prefix": {"type": "inline", "text": "Fix this: "},
        },
        "easy": {"model": "cheap-model"},
        "hard": {"model": "expensive-model"},
    })


def _difficulty_config():
    payload = _routed_config().model_dump(mode="json")
    payload["name"] = "routed-difficulty"
    payload["method"]["type"] = "difficulty"
    payload["method"].pop("escalation")
    return StrategyConfig.model_validate(payload)


def _strategy_snapshot(config):
    return {
        "name": config.name,
        "digest": compute_strategy_digest(config, {}),
        "config": config.model_dump(mode="json"),
        "prompts": [],
    }


def _record(strategy_id, task_id, passed, phases, cost):
    return {
        "strategy_id": strategy_id,
        "task_id": task_id,
        "error": None,
        "evaluation": {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
        },
        "attempts": [
            {"route": {"phase": phase}}
            for phase in phases
        ],
        "metrics": {"cost": {"total_usd": cost}},
    }


def _artifacts(*, duplicate_cheap=False):
    cheap = _fixed_config(
        "fixed-cheap",
        "cheap-model",
        "generator",
        prompt_suffix="Return ESCALATE when unsure.",
    )
    expensive = _fixed_config("fixed-expensive", "expensive-model", "writer")
    routed = _routed_config()
    strategies = [
        _strategy_snapshot(cheap),
        _strategy_snapshot(expensive),
        _strategy_snapshot(routed),
    ]
    if duplicate_cheap:
        strategies.append(_strategy_snapshot(
            _fixed_config(
                "fixed-cheap-copy",
                "cheap-model",
                "generator",
                prompt_suffix="Return ESCALATE when unsure.",
            )
        ))
    manifest = {
        "schema_version": 5,
        "strategies": strategies,
        "case_ids": list(_TASK_IDS),
    }
    records = []
    for task_id in _TASK_IDS:
        records.extend([
            _record(
                "fixed-cheap",
                task_id,
                _CHEAP_PASSES[task_id],
                ["fixed"],
                1.0,
            ),
            _record(
                "fixed-expensive",
                task_id,
                _EXPENSIVE_PASSES[task_id],
                ["fixed"],
                None if task_id == "t4" else 10.0,
            ),
            _record(
                "routed",
                task_id,
                _ROUTED_PASSES[task_id],
                _ROUTED_PHASES[task_id],
                {"t1": 1.1, "t2": 10.1, "t3": 1.1, "t4": 11.1, "t5": 10.1}[task_id],
            ),
        ])
        if duplicate_cheap:
            records.append(_record(
                "fixed-cheap-copy",
                task_id,
                _CHEAP_PASSES[task_id],
                ["fixed"],
                1.0,
            ))
    return manifest, records


def _evaluate(manifest, records):
    # Artifact-schema validation has its own adversarial test suite. These
    # isolated tests exercise the pure derivation against snapshots that retain
    # exactly the v4 fields this module consumes.
    with (
        patch(
            "smart_ask.benchmarks.counterfactual.validate_manifest"
        ) as manifest_validator,
        patch(
            "smart_ask.benchmarks.counterfactual.validate_records",
            side_effect=lambda values, _manifest: list(values),
        ) as record_validator,
    ):
        report = evaluate_counterfactual_routing(manifest, records)
    manifest_validator.assert_called_once_with(manifest)
    record_validator.assert_called_once()
    return report


class CounterfactualQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.records = _artifacts()

    def test_paired_metrics_and_cost_quality_coverage(self):
        report = _evaluate(self.manifest, self.records)
        self.assertEqual(report["schema_version"], 1)
        self.assertIn("do not establish", report["evidence_caveat"])
        self.assertIn("configured easy and hard", report["baseline_semantics"])
        self.assertEqual(len(report["strategies"]), 1)
        routed = report["strategies"][0]
        self.assertEqual(routed["strategy_id"], "routed")
        self.assertEqual(routed["method"], "cascade")
        self.assertEqual(routed["baselines"]["status"], "available")
        self.assertEqual(
            routed["baselines"]["cheap"]["strategy_id"],
            "fixed-cheap",
        )
        self.assertEqual(
            routed["baselines"]["expensive"]["strategy_id"],
            "fixed-expensive",
        )
        self.assertEqual(routed["fully_paired_tasks"], 5)

        metrics = routed["metrics"]
        self.assertEqual(
            metrics["cheap_opportunity_capture"],
            {
                "numerator": 1,
                "denominator": 2,
                "value": 0.5,
                "evidence_tasks": 5,
                "unavailable_tasks": 0,
                "unavailable_reasons": {},
            },
        )
        self.assertEqual(
            metrics["unnecessary_expensive_rate"]["numerator"],
            1,
        )
        self.assertEqual(
            metrics["unnecessary_expensive_rate"]["denominator"],
            3,
        )
        self.assertEqual(metrics["unsafe_cheap_rate"]["value"], 0.5)
        self.assertEqual(metrics["escalation_precision"]["value"], 1.0)

        tasks = {task["task_id"]: task for task in routed["per_task"]}
        self.assertEqual(tasks["t1"]["route_path"], "start -> cheap -> accept")
        self.assertEqual(tasks["t2"]["selection"], "expensive_direct")
        self.assertTrue(
            tasks["t3"]["metrics"]["unsafe_cheap_rate"]["value"]
        )
        self.assertEqual(tasks["t4"]["selection"], "escalated")
        self.assertEqual(tasks["t3"]["oracle_baseline"], "expensive")
        self.assertEqual(
            tasks["t5"]["metrics"]["quality_regret"]["reasons"],
            ["no_passing_baseline"],
        )

        self.assertEqual(metrics["quality_regret"]["available_tasks"], 4)
        self.assertEqual(metrics["quality_regret"]["total"], 1.0)
        self.assertEqual(metrics["quality_regret"]["mean"], 0.25)
        self.assertEqual(metrics["cost_regret_usd"]["available_tasks"], 3)
        self.assertAlmostEqual(metrics["cost_regret_usd"]["total"], 0.3)
        self.assertEqual(
            metrics["cost_regret_usd"]["unavailable_reasons"],
            {
                "expensive_baseline_cost_unavailable": 1,
                "no_passing_baseline": 1,
            },
        )
        self.assertEqual(
            tasks["t4"]["metrics"]["quality_regret"]["status"],
            "available",
        )
        self.assertEqual(
            tasks["t4"]["metrics"]["cost_regret_usd"]["reasons"],
            ["expensive_baseline_cost_unavailable"],
        )

    def test_record_order_does_not_change_report(self):
        expected = _evaluate(self.manifest, self.records)
        actual = _evaluate(self.manifest, reversed(self.records))
        self.assertEqual(actual, expected)

    def test_provider_reported_cost_is_preferred_without_mixing_sources(self):
        records = deepcopy(self.records)
        for record in records:
            estimated = record["metrics"]["cost"]["total_usd"]
            record["metrics"]["cost"]["provider_reported"] = {
                "total_usd": estimated,
            }

        report = _evaluate(self.manifest, records)["strategies"][0]
        tasks = {task["task_id"]: task for task in report["per_task"]}
        self.assertEqual(
            tasks["t1"]["metrics"]["cost_regret_usd"]["source"],
            "provider_reported",
        )
        self.assertEqual(
            report["metrics"]["cost_regret_usd"]["sources"],
            {"provider_reported": 3},
        )

    def test_oracle_is_the_least_cost_passing_baseline(self):
        records = deepcopy(self.records)
        for record in records:
            if record["task_id"] != "t1":
                continue
            if record["strategy_id"] == "fixed-cheap":
                record["metrics"]["cost"]["total_usd"] = 10.0
            elif record["strategy_id"] == "fixed-expensive":
                record["metrics"]["cost"]["total_usd"] = 1.0

        task = next(
            item
            for item in _evaluate(self.manifest, records)["strategies"][0]["per_task"]
            if item["task_id"] == "t1"
        )
        self.assertEqual(task["oracle_baseline"], "expensive")
        self.assertEqual(task["oracle_cost_source"], "catalog_estimate")
        self.assertAlmostEqual(
            task["metrics"]["cost_regret_usd"]["value"],
            0.1,
        )

    def test_missing_task_evidence_is_not_counted_as_failure_or_zero(self):
        records = [
            record for record in self.records
            if not (
                record["strategy_id"] == "fixed-expensive"
                and record["task_id"] == "t3"
            )
        ]
        routed = _evaluate(self.manifest, records)["strategies"][0]
        self.assertEqual(routed["fully_paired_tasks"], 4)
        self.assertEqual(routed["incomplete_pair_tasks"], 1)
        task = next(item for item in routed["per_task"] if item["task_id"] == "t3")
        self.assertEqual(
            task["full_pair_evidence"]["reasons"],
            ["missing_expensive_record"],
        )
        self.assertIsNone(
            task["metrics"]["unsafe_cheap_rate"]["eligible"]
        )
        self.assertEqual(
            routed["metrics"]["unsafe_cheap_rate"]["unavailable_tasks"],
            1,
        )

    def test_missing_and_ambiguous_baselines_are_explicit(self):
        missing_manifest = deepcopy(self.manifest)
        missing_manifest["strategies"] = [
            strategy for strategy in missing_manifest["strategies"]
            if strategy["name"] != "fixed-expensive"
        ]
        missing_records = [
            record for record in self.records
            if record["strategy_id"] != "fixed-expensive"
        ]
        missing = _evaluate(missing_manifest, missing_records)["strategies"][0]
        self.assertEqual(missing["baselines"]["status"], "unavailable")
        self.assertEqual(
            missing["baselines"]["reasons"],
            ["missing_expensive_baseline"],
        )
        self.assertEqual(missing["fully_paired_tasks"], 0)
        self.assertEqual(
            missing["metrics"]["cheap_opportunity_capture"]["denominator"],
            2,
        )
        self.assertEqual(
            missing["metrics"]["unnecessary_expensive_rate"]["denominator"],
            3,
        )
        self.assertEqual(
            missing["metrics"]["unsafe_cheap_rate"]["denominator"],
            1,
        )
        self.assertEqual(
            missing["metrics"]["unsafe_cheap_rate"]["numerator"],
            0,
        )

        ambiguous_manifest, ambiguous_records = _artifacts(duplicate_cheap=True)
        ambiguous = _evaluate(
            ambiguous_manifest,
            ambiguous_records,
        )["strategies"][0]
        self.assertEqual(
            ambiguous["baselines"]["cheap"]["status"],
            "ambiguous",
        )
        self.assertEqual(
            ambiguous["baselines"]["cheap"]["candidate_strategy_ids"],
            ["fixed-cheap", "fixed-cheap-copy"],
        )
        self.assertEqual(
            ambiguous["baselines"]["reasons"],
            ["ambiguous_cheap_baseline"],
        )

    def test_matching_uses_complete_profiles_and_supports_difficulty(self):
        mismatched = deepcopy(self.manifest)
        cheap = next(
            strategy for strategy in mismatched["strategies"]
            if strategy["name"] == "fixed-cheap"
        )
        cheap["config"]["method"]["model"]["parameters"]["max_tokens"] = 99
        report = _evaluate(mismatched, self.records)["strategies"][0]
        self.assertEqual(report["baselines"]["cheap"]["status"], "missing")
        self.assertEqual(
            report["baselines"]["reasons"],
            ["missing_cheap_baseline"],
        )

        difficulty_manifest = deepcopy(self.manifest)
        difficulty_manifest["strategies"] = [
            strategy for strategy in difficulty_manifest["strategies"]
            if strategy["name"] != "routed"
        ] + [_strategy_snapshot(_difficulty_config())]
        next(
            strategy for strategy in difficulty_manifest["strategies"]
            if strategy["name"] == "fixed-cheap"
        )["config"]["method"].pop("prompt_suffix")
        difficulty_records = []
        for record in self.records:
            copied = deepcopy(record)
            if copied["strategy_id"] == "routed":
                copied["strategy_id"] = "routed-difficulty"
                if copied["task_id"] == "t4":
                    copied["attempts"] = [{"route": {"phase": "initial-easy"}}]
            difficulty_records.append(copied)
        difficulty = _evaluate(
            difficulty_manifest,
            difficulty_records,
        )["strategies"][0]
        self.assertEqual(difficulty["method"], "difficulty")
        self.assertEqual(difficulty["baselines"]["status"], "available")
        self.assertEqual(
            difficulty["metrics"]["escalation_precision"]["denominator"],
            0,
        )

    def test_cascade_rejects_plain_fixed_cheap_baseline(self):
        manifest = deepcopy(self.manifest)
        cheap = next(
            strategy for strategy in manifest["strategies"]
            if strategy["name"] == "fixed-cheap"
        )
        cheap["config"]["method"].pop("prompt_suffix")

        routed = _evaluate(manifest, self.records)["strategies"][0]
        self.assertEqual(routed["baselines"]["cheap"]["status"], "missing")
        self.assertEqual(
            routed["baselines"]["reasons"],
            ["missing_cheap_baseline"],
        )
        self.assertEqual(
            routed["baselines"]["cheap"]["required_user_prompt_transform"],
            {"suffix": "Return ESCALATE when unsure."},
        )

    def test_unsafe_cheap_uses_observed_routed_failure_not_fixed_replicate(self):
        records = deepcopy(self.records)
        cheap = next(
            record for record in records
            if record["strategy_id"] == "fixed-cheap"
            and record["task_id"] == "t3"
        )
        cheap["evaluation"] = {"passed": True, "score": 1.0}

        task = next(
            item
            for item in _evaluate(self.manifest, records)["strategies"][0]["per_task"]
            if item["task_id"] == "t3"
        )
        self.assertTrue(task["metrics"]["unsafe_cheap_rate"]["value"])

    def test_cost_regret_falls_back_to_catalog_as_one_source(self):
        records = deepcopy(self.records)
        for record in records:
            estimated = record["metrics"]["cost"]["total_usd"]
            if estimated is not None:
                record["metrics"]["cost"]["provider_reported"] = {
                    "total_usd": estimated,
                }
        routed_t1 = next(
            record for record in records
            if record["strategy_id"] == "routed" and record["task_id"] == "t1"
        )
        routed_t1["metrics"]["cost"]["provider_reported"] = {
            "total_usd": None,
        }

        report = _evaluate(self.manifest, records)["strategies"][0]
        task = next(item for item in report["per_task"] if item["task_id"] == "t1")
        self.assertEqual(task["oracle_cost_source"], "catalog_estimate")
        self.assertEqual(
            task["metrics"]["cost_regret_usd"]["source"],
            "catalog_estimate",
        )
        self.assertTrue(report["metrics"]["cost_regret_usd"]["mixed_sources"])
        self.assertIsNone(report["metrics"]["cost_regret_usd"]["total"])
        self.assertEqual(
            set(report["metrics"]["cost_regret_usd"]["by_source"]),
            {"catalog_estimate", "provider_reported"},
        )

    def test_quality_regret_never_emits_non_finite_json_numbers(self):
        records = deepcopy(self.records)
        routed = next(
            record for record in records
            if record["strategy_id"] == "routed" and record["task_id"] == "t1"
        )
        cheap = next(
            record for record in records
            if record["strategy_id"] == "fixed-cheap"
            and record["task_id"] == "t1"
        )
        expensive = next(
            record for record in records
            if record["strategy_id"] == "fixed-expensive"
            and record["task_id"] == "t1"
        )
        routed["evaluation"]["score"] = -1.7e308
        cheap["evaluation"]["score"] = 1.7e308
        expensive["evaluation"] = {"passed": False, "score": 0.0}

        task = next(
            item
            for item in _evaluate(self.manifest, records)["strategies"][0]["per_task"]
            if item["task_id"] == "t1"
        )
        self.assertEqual(
            task["metrics"]["quality_regret"]["reasons"],
            ["non_finite_delta"],
        )

    def test_inputs_are_validated_before_analysis(self):
        with patch(
            "smart_ask.benchmarks.counterfactual.validate_manifest",
            side_effect=ValueError("invalid schema_version"),
        ):
            with self.assertRaisesRegex(ValueError, "schema_version"):
                evaluate_counterfactual_routing(self.manifest, self.records)

        with (
            patch("smart_ask.benchmarks.counterfactual.validate_manifest"),
            patch(
                "smart_ask.benchmarks.counterfactual.validate_records",
                side_effect=ValueError("invalid finite score"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                evaluate_counterfactual_routing(self.manifest, self.records)


if __name__ == "__main__":
    unittest.main()
