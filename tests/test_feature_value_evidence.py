"""Reconcile published feature-value evidence without fitting any models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/feature_value_pilot.json"


def test_pooled_scores_reconcile_with_integer_hits() -> None:
    report = json.loads(REPORT.read_text())
    assert len(report["arms"]) == 9
    for arm in report["arms"].values():
        assert arm["denominators"] == [495, 153, 92]
        assert [sum(x) for x in zip(*arm["fold_hits"], strict=True)] == arm["hits"]
        actual = sum(w * h / d for w, h, d in zip(
            (0.1, 0.3, 0.6), arm["hits"], arm["denominators"], strict=True
        ))
        assert math.isclose(actual, arm["weighted_recall_at_20"], abs_tol=1e-12)


def test_paired_differences_and_ambiguous_intervals() -> None:
    report = json.loads(REPORT.read_text())
    for name, result in report["comparisons"].items():
        left, right = name.split("_minus_")
        actual = (report["arms"][left]["weighted_recall_at_20"]
                  - report["arms"][right]["weighted_recall_at_20"])
        assert math.isclose(actual, result["difference"], abs_tol=1e-12)
        lower, upper = result["descriptive_95_interval"]
        assert lower < 0 < upper
        assert result["bootstrap_replicates"] == 1000


def test_temporal_scope_and_target_censoring_are_explicit() -> None:
    protocol = json.loads(REPORT.read_text())["protocol"]
    assert protocol["role"] == "fit"
    assert protocol["selection_access"] is protocol["evaluation_access"] is False
    assert protocol["training_targets_censored_before_negative_sampling"] is True
    assert protocol["history_end_ms"] == 1660687200000
    assert protocol["source_sessions"] == 1024
    assert protocol["validation_sessions"] == 512
    for fold in protocol["folds"]:
        assert fold["latest_training_query_ms"] < fold["cutoff_ms"] - 6 * 3600000
        assert fold["validation_sessions"] == 256


def test_no_promotion_or_leaderboard_claim() -> None:
    report = json.loads(REPORT.read_text())
    assert report["decisions"]["feature_engineering_complete"] is False
    assert report["decisions"]["feature_retention_decisions"] == 0
    assert report["decisions"]["kaggle_submission_made"] is False
    assert report["execution"]["new_models"] == 54
    assert report["execution"]["replay_new_models"] == 0
    assert report["execution"]["identical_replay_metrics"] is True
    assert report["new_feature_family"]["features"] == 12
    assert len(report["source"]["result_sha256"]) == 64


def test_relative_family_does_not_pass_both_fold_gate() -> None:
    report = json.loads(REPORT.read_text())
    for comparison in ("baseline_relative_minus_baseline", "shared_relative_minus_shared"):
        differences = report["comparisons"][comparison]["fold_differences"]
        assert min(differences) < 0 < max(differences)


def test_candidate_oracle_is_not_reported_as_a_trained_score() -> None:
    report = json.loads(REPORT.read_text())
    oracle = report["candidate_oracle"]
    expected = sum(w * h / d for w, h, d in zip(
        (0.1, 0.3, 0.6), oracle["hits"], oracle["denominators"], strict=True
    ))
    assert math.isclose(expected, oracle["weighted_recall_at_20"], abs_tol=1e-12)
    assert max(a["weighted_recall_at_20"] for a in report["arms"].values()) < expected
    assert "not a Kaggle score" in oracle["interpretation"]


def test_executed_research_notebook_and_receipt_match() -> None:
    receipt = json.loads((ROOT / "notebooks/research/feature_value_pilot.receipt.json").read_text())
    path = ROOT / receipt["notebook"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["notebook_sha256"]
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == receipt["evidence_sha256"]
    notebook = json.loads(path.read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == receipt["code_cells"] == 4
    assert receipt["model_fits"] == 0
    assert all(type(c["execution_count"]) is int for c in code)
    assert all(o["output_type"] != "error" for c in code for o in c.get("outputs", []))
