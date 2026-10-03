"""Publication contracts for the contextual, sequence and ensemble validation frontier."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "research" / "frontier"
REPORTS = ROOT / "reports" / "research"


def test_latest_frontier_status_is_internally_consistent() -> None:
    data = json.loads((REPORTS / "frontier_status_20261003.json").read_text())
    assert data["verified_submission"]["submission_id"] == 56542128
    assert data["verified_submission"]["private_score"] == 0.57586
    assert data["verified_submission"]["public_score"] == 0.57601

    active = data["active_frontier"]
    assert active["name"] == "Final Reserve V2"
    assert active["status"] == "PREDICTION_PREPARATION_ACTIVE"
    assert active["sessions_total"] == 412_492
    assert active["sessions_predictions_frozen"] == 200_000
    assert active["sessions_remaining"] == 212_492
    assert active["shards_complete"] == 4
    assert active["shards_total"] == 9
    assert active["reserve_labels_opened"] is False


def test_validation_lineage_preserves_measured_decisions() -> None:
    data = json.loads((REPORTS / "frontier_status_20261003.json").read_text())
    records = {item["name"]: item for item in data["completed_frontier"]}

    source = records["source-aware click/cart independent selection"]
    assert source["decision"] == "STOP_CLICK_CART_SELECTION"
    assert source["click_hit_gain"] == 259
    assert source["cart_hit_gain"] == -61
    assert source["bootstrap_95_ci"][0] < 0 < source["bootstrap_95_ci"][1]

    stack = records["heterogeneous OOF stack"]
    assert stack["decision"] == "PROMOTE_TO_FRESH_SELECTION_V2"
    assert stack["combined_gain"] == 0.00420318
    assert stack["cart_hit_gain"] == 64
    assert stack["combined_nonnegative_folds"] == 5
    assert stack["cart_nonnegative_folds"] == 5

    fresh = records["Fresh Selection V2"]
    assert fresh["decision"] == "PROMOTE_TO_FINAL_RESERVE_V2"
    assert fresh["combined_gain"] == 0.00319387
    assert fresh["click_hit_gain"] == 136
    assert fresh["cart_hit_gain"] == 47
    assert fresh["bootstrap_95_ci"][0] > 0
    assert fresh["first_half_gain"] > 0
    assert fresh["second_half_gain"] > 0


def test_current_employer_facing_surfaces_are_project_evidence_first() -> None:
    readme = (ROOT / "README.md").read_text()
    frontier = (FRONTIER / "README.md").read_text()
    current = (FRONTIER / "10_contextual_sequence_stack_validation.md").read_text()

    assert "200,000 / 412,492" in readme
    assert "Fresh Selection V2" in readme
    assert "Final Reserve V2" in frontier
    assert "+0.004203" in current
    assert "+0.003194" in current


def test_publication_boundary_excludes_private_competition_artifacts() -> None:
    data = json.loads((REPORTS / "frontier_status_20261003.json").read_text())
    excluded = set(data["publication_boundary"]["excludes"])

    assert "raw competition data" in excluded
    assert "row-level labels" in excluded
    assert "row-level predictions" in excluded
    assert "session identifiers" in excluded
    assert "private runners" in excluded
    assert "full model checkpoints" in excluded
    assert "credentials" in excluded


def test_current_frontier_document_matches_status() -> None:
    text = (FRONTIER / "10_contextual_sequence_stack_validation.md").read_text()

    assert "+0.004203" in text
    assert "+0.003194" in text
    assert "[+0.001865, +0.004554]" in text
    assert "200,000 / 412,492 sessions" in text
    assert "4 / 9 shards" in text
    assert "reserve labels remain **unopened**" in text
