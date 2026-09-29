"""Contracts for the promoted similarity stack and current v31 frontier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _status() -> dict[str, Any]:
    return json.loads(
        (ROOT / "reports/research/frontier_status_20260928.json").read_text()
    )


def test_similarity_submission_is_current_verified_incumbent() -> None:
    competition = _status()["competition"]
    incumbent = competition["incumbent"]
    assert incumbent["ref"] == 56542128
    assert incumbent["private"] == 0.57586
    assert incumbent["public"] == 0.57601
    assert incumbent["sessions"] == 1_671_803
    assert incumbent["rows"] == 5_015_409
    assert incumbent["sha256"] == (
        "3e2e8085da1ae804ae9c71d626b0818b599c2d686552c4b3e70cb76587642d44"
    )
    assert competition["delta_vs_prior"] == {"private": 0.00446, "public": 0.00435}
    assert competition["remaining_private_gap"] == 0.02917


def test_similarity_promotion_preserves_reserved_temporal_evidence() -> None:
    similarity = _status()["promoted_similarity_stack"]
    assert similarity["feature_count"] == 137
    assert similarity["feature_blocks"] == {
        "established": 102,
        "candidate_to_session_similarity": 28,
        "neural_query_diagnostics": 7,
    }
    evaluation = similarity["evaluation"]
    assert evaluation["sessions"] == 432_492
    assert evaluation["weighted_gain"] == 0.004396
    assert evaluation["net_order_hits"] == 508
    assert evaluation["paired_95_interval"] == [0.003721, 0.005046]
    assert evaluation["first_half_gain"] > 0
    assert evaluation["second_half_gain"] > 0
    assert similarity["decision"] == "PROMOTED_AND_DEPLOYED"


def test_closed_xgb_and_active_v31_are_not_overclaimed() -> None:
    status = _status()
    closed = {row["name"]: row for row in status["closed_since_previous_snapshot"]}
    xgb = closed["fixed GPU XGBoost order family"]
    assert xgb["decision"].startswith("STOP_")
    assert xgb["evidence"]["net_order_hits"] == 4
    assert xgb["evidence"]["evaluation_labels_read"] is False

    active = {row["name"]: row for row in status["active_frontier"]}
    v31 = active["v31 attention candidate union"]
    assert "cross-validation in progress" in v31["status"]
    assert v31["planned_feature_count"] == 184
    assert v31["planned_candidate_cap"] == 800


def test_submission_receipt_matches_frontier_snapshot() -> None:
    receipt = json.loads(
        (ROOT / "reports/submissions/similarity_stack_20260925.json").read_text()
    )
    incumbent = _status()["competition"]["incumbent"]
    assert receipt["submission_ref"] == incumbent["ref"]
    assert receipt["private_score"] == incumbent["private"]
    assert receipt["public_score"] == incumbent["public"]
    assert receipt["sha256"] == incumbent["sha256"]
