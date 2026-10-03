"""Public contract tests for the completed source-aware click/cart selection stage."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "research" / "frontier"


def load_contract() -> dict:
    return json.loads((FRONTIER / "click_cart_selection_contract.json").read_text())


def test_verified_submission_is_frozen_without_competitor_comparison() -> None:
    data = load_contract()
    state = data["verified_submission"]
    assert state["submission_id"] == 56542128
    assert state["private_score"] == 0.57586
    assert state["public_score"] == 0.57601
    serialized = json.dumps(data).lower()
    assert "historical_private_winner" not in serialized
    assert "private_gap" not in serialized
    assert "0.60503" not in serialized


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


def test_selection_result_is_closed_and_recomputes_failure() -> None:
    data = load_contract()
    selection = data["selection_contract"]
    result = data["selection_result"]
    assert data["status"] == "SELECTION_CLOSED"
    assert selection["cohort_sessions"] == 20_000
    assert selection["bootstrap_replicates"] == 2_000
    assert result["decision"] == "STOP_CLICK_CART_SELECTION"
    assert result["selection_labels_opened"] is True
    assert result["reserved_evaluation_labels_opened"] is False
    assert result["click_hit_gain"] == 259
    assert result["cart_hit_gain"] == -61
    assert result["combined_weighted_gain"] < selection["frozen_gate"]["min_combined_weighted_gain"]
    assert result["paired_bootstrap_95_ci"][0] < 0
    assert result["first_half_gain"] < 0
    assert result["second_half_gain"] < 0
    assert result["passed"] is False


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


def test_protocol_document_matches_completed_contract() -> None:
    text = (FRONTIER / "09_click_cart_selection_protocol.md").read_text()
    assert "0.57586 private / 0.57601 public" in text
    assert "+0.003535" in text
    assert "2,000 replicates" in text
    assert "STOP_CLICK_CART_SELECTION" in text
    assert "+259" in text
    assert "-61" in text
    assert "0.60503" not in text
