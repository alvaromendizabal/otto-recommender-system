"""Public contract tests for the fitting-qualified click/cart selection stage."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "research" / "frontier"


def load_contract() -> dict:
    return json.loads((FRONTIER / "click_cart_selection_contract.json").read_text())


def test_verified_incumbent_and_gap_are_frozen() -> None:
    data = load_contract()
    state = data["competition_state"]
    assert state["submission_id"] == 56542128
    assert state["private_score"] == 0.57586
    assert state["public_score"] == 0.57601
    assert state["historical_private_winner"] == 0.60503
    assert abs(state["private_gap"] - (0.60503 - 0.57586)) < 1e-12


def test_fitting_promotion_arithmetic_and_stability() -> None:
    data = load_contract()["fitting_result"]
    assert data["decision"] == "PROMOTE_TO_SELECTION"
    assert data["candidate_budget"] == 400
    assert data["feature_count"] == 589
    assert data["clicks"]["challenger_hits"] - data["clicks"]["baseline_hits"] == 186
    assert data["carts"]["challenger_hits"] - data["carts"]["baseline_hits"] == 53
    assert abs(
        data["combined_weighted_gain"]
        - (data["clicks"]["weighted_gain"] + data["carts"]["weighted_gain"])
    ) < 1e-12
    assert data["nonnegative_folds"] == 5
    assert all(value > 0 for value in data["chronological_fold_gains"])
    assert min(data["chronological_fold_gains"]) == data["worst_fold_gain"]


def test_fitting_gate_is_recomputed_from_published_evidence() -> None:
    data = load_contract()["fitting_result"]
    gate = data["frozen_gate"]
    assert data["combined_weighted_gain"] >= gate["min_combined_weighted_gain"]
    assert data["clicks"]["hit_gain"] >= gate["min_click_hit_gain"]
    assert data["carts"]["hit_gain"] >= gate["min_cart_hit_gain"]
    assert data["nonnegative_folds"] >= gate["min_nonnegative_folds"]
    assert data["worst_fold_gain"] >= gate["min_worst_fold_gain"]


def test_selection_remains_preregistered_and_unopened() -> None:
    data = load_contract()
    selection = data["selection_contract"]
    assert selection["cohort_sessions"] == 20_000
    assert selection["bootstrap_replicates"] == 2_000
    assert selection["selection_labels_opened"] is False
    assert selection["reserved_evaluation_labels_opened"] is False
    assert data["reserved_evaluation"]["cohort_sessions"] == 432_492
    assert data["reserved_evaluation"]["status"] == "BLOCKED_UNTIL_SELECTION_PASSES"
    gate = selection["frozen_gate"]
    assert gate["min_combined_weighted_gain"] == 0.0025
    assert gate["min_click_hit_gain"] == 0
    assert gate["min_cart_hit_gain"] == 10
    assert gate["paired_bootstrap_lower_bound_must_be_positive"] is True


def test_public_contract_excludes_private_artifacts() -> None:
    data = load_contract()
    forbidden_keys = {
        "session_ids",
        "cohort_ids",
        "labels",
        "predictions",
        "access_token",
        "secret_access_key",
        "checkpoint_bytes",
    }

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(data)
    excluded = set(data["publication_boundary"]["excludes"])
    assert "row-level labels" in excluded
    assert "row-level predictions" in excluded
    assert "private runners" in excluded
    assert "model checkpoints" in excluded


def test_protocol_document_matches_machine_readable_contract() -> None:
    text = (FRONTIER / "09_click_cart_selection_protocol.md").read_text()
    assert "0.57586 private / 0.57601 public" in text
    assert "+0.003535" in text
    assert "2,000 replicates" in text
    assert "432,492 sessions" in text
    assert "Selection and reserved-evaluation labels" not in text or "closed" in text.lower()
