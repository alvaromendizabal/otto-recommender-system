"""Metric accounting, target isolation, and actual exported-artifact checks."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "recommendation_audit", HERE / "audit_recommendations.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import independent auditor")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture() -> tuple[dict, dict]:
    predictions = {
        "schema_version": 1,
        "k": 20,
        "candidate_limit": 21,
        "rows": [
            {
                "session": session,
                "query_ts": 100,
                "objectives": {
                    action: {
                        "candidates": list(range(1, 22)),
                        "recommendations": list(range(1, 21)),
                    }
                    for action in AUDIT.ACTIONS
                },
            }
            for session in (7, 8)
        ],
    }
    evaluation = {
        "schema_version": 1,
        "synthetic": True,
        "queries": [
            {"session": 7, "query_ts": 100, "prefix_events": 3},
            {"session": 8, "query_ts": 100, "prefix_events": 6},
        ],
        "targets": [
            {"session": 7, "item": item, "ts": 101, "action": "clicks"} for item in (1, 21, 22)
        ],
    }
    return predictions, evaluation


class RecommendationAuditTests(unittest.TestCase):
    def test_counts_distinguish_ranked_reachable_and_unreachable_targets(self) -> None:
        result = AUDIT.audit_payload(*fixture())
        clicks = result["overall"]["objectives"]["clicks"]
        self.assertEqual(clicks["hits"], 1)
        self.assertEqual(clicks["ranking_shortfall"], 1)
        self.assertEqual(clicks["retrieval_shortfall"], 1)
        self.assertEqual(clicks["denominator"], 3)
        self.assertEqual(clicks["sessions_with_targets"], 1)
        self.assertEqual(result["overall"]["sessions"], 2)
        self.assertAlmostEqual(result["overall"]["weighted_evaluable_mass"], 0.1)
        self.assertEqual(result["prefix_slices"]["5_or_more_events"]["sessions"], 1)

    def test_deduplication_cap_and_pooling_use_counts_instead_of_session_means(self) -> None:
        predictions, evaluation = fixture()
        evaluation["targets"] = [
            {"session": 7, "item": item, "ts": 101, "action": "clicks"} for item in range(1, 26)
        ]
        evaluation["targets"] += [
            {"session": 8, "item": 22, "ts": 101, "action": "clicks"},
            {"session": 8, "item": 22, "ts": 102, "action": "clicks"},
        ]
        report = AUDIT.audit_payload(predictions, evaluation)
        clicks = report["overall"]["objectives"]["clicks"]
        self.assertEqual(clicks["denominator"], 21)
        self.assertEqual(clicks["hits"], 20)
        self.assertEqual(clicks["candidate_hits"], 20)
        self.assertAlmostEqual(clicks["recall_at_20"], 20 / 21)
        self.assertNotEqual(clicks["recall_at_20"], 0.5)
        for count in ("hits", "candidate_hits", "denominator"):
            self.assertEqual(
                clicks[count],
                sum(
                    slice_["objectives"]["clicks"][count]
                    for slice_ in report["prefix_slices"].values()
                ),
            )

    def test_invalid_coverage_duplicates_and_temporal_boundaries_reject(self) -> None:
        predictions, evaluation = fixture()
        mutations = (
            lambda p, e: p["rows"].append(copy.deepcopy(p["rows"][0])),
            lambda p, e: e["queries"].pop(),
            lambda p, e: e["targets"][0].update(session=999),
            lambda p, e: e["targets"][0].update(ts=100),
            lambda p, e: e["targets"][0].update(item=True),
            lambda p, e: e["targets"][0].update(action="unknown"),
            lambda p, e: p["rows"][0]["objectives"]["clicks"]["recommendations"].append(1),
            lambda p, e: p["rows"][0]["objectives"]["clicks"].update(recommendations=[999]),
        )
        for mutate in mutations:
            left, right = copy.deepcopy(predictions), copy.deepcopy(evaluation)
            mutate(left, right)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                AUDIT.audit_payload(left, right)

    def test_complete_generated_population_and_immutable_audit_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(HERE.parent / "scripts/run_public_demo.py"),
                    "--output",
                    str(root / "demo"),
                ],
                check=True,
                capture_output=True,
            )
            command = [
                sys.executable,
                "-S",
                str(HERE / "audit_recommendations.py"),
                "--demo-output",
                str(root / "demo"),
                "--output",
                str(root / "audit.json"),
                "--html",
                str(root / "diagnostics.html"),
            ]
            subprocess.run(command, check=True, capture_output=True)
            original = (root / "audit.json").read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(original, (root / "audit.json").read_bytes())
            report = json.loads(original)
            self.assertEqual(report["overall"]["sessions"], 48)
            self.assertEqual(sum(s["sessions"] for s in report["prefix_slices"].values()), 48)
            overall = report["overall"]
            self.assertAlmostEqual(
                overall["weighted_recall_at_20"]
                + overall["weighted_ranking_shortfall"]
                + overall["weighted_retrieval_shortfall"],
                overall["weighted_evaluable_mass"],
            )
            self.assertIn("synthetic", (root / "diagnostics.html").read_text())
            (root / "audit.json").write_text("preserve me")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((root / "audit.json").read_text(), "preserve me")
            (root / "demo/evaluation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Changed source"):
                AUDIT.audit(root / "demo")


if __name__ == "__main__":
    unittest.main()
