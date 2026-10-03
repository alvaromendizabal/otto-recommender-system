"""Public contract tests for the completed click/cart selection stage."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "research" / "frontier"


def load_contract() -> dict:
    return json.loads((FRONTIER / "click_cart_selection_contract.json").read_text())


def test_verified_incumbent_is_published_without_external_score_comparison() -> None:
    state = load_contract()["competition_state"]
    assert state["submission_id"] == 56542128
    assert state["private_score"] == 0.57586
    assert state["public_score"] == 0.57601
    assert "historical_private_winner" not in state
    assert "private_gap" not in state


def test_fitting_result_is_preserved() -> None:
    data = load_contract()["fitting_result"]
    assert data["decision"] == "PROMOTE_TO_SELECTION"
    assert data["candidate_budget"] == 400
    assert data["feature_count"] == 589
    assert data["clicks"]["challenger_hits"] - data["clicks"]["baseline_hits"] == 186
    assert data["carts"]["challenger_hits"] - data["carts"]["baseline_hits"] == 53
    assert data["nonnegative_folds"] == 5


def test_independent_selection_result_is_closed() -> None:
    data = load_contract()["selection_result"]
    assert data["decision"] == "STOP_CLICK_CART_SELECTION"
    assert data["cohort_sessions"] == 20_000
    assert data["bootstrap_replicates"] == 2_000
    assert data["clicks"]["hit_gain"] == 259
    assert data["carts"]["hit_gain"] == -61
    assert data["combined_weighted_gain"] < 0
    assert data["paired_bootstrap_95_interval"][0] < 0
    assert data["paired_bootstrap_95_interval"][1] > 0
    assert data["selection_labels_opened"] is True
    assert data["reserved_evaluation_labels_opened"] is False


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
    assert "STOP_CLICK_CART_SELECTION" in text
    assert "−61 hits" in text
