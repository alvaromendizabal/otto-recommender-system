"""Public regression tests for the October 3 frontier snapshot."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "research" / "frontier"
STATUS = ROOT / "reports" / "research" / "frontier_status_20261003.json"


def test_current_status_and_incumbent_are_consistent() -> None:
    data = json.loads(STATUS.read_text())
    assert data["competition"]["incumbent"]["ref"] == 56542128
    assert data["competition"]["incumbent"]["private"] == 0.57586
    assert data["competition"]["incumbent"]["public"] == 0.57601
    assert data["active_frontier"]["name"] == "Final Reserve V2"
    assert data["active_frontier"]["predicted_sessions"] == 200_000
    assert data["active_frontier"]["reserve_sessions"] == 412_492
    assert data["active_frontier"]["completed_shards"] == 4
    assert data["active_frontier"]["total_shards"] == 9
    assert data["active_frontier"]["labels_opened"] is False


def test_stack_and_fresh_selection_evidence_is_published() -> None:
    data = json.loads(STATUS.read_text())
    by_name = {row["name"]: row for row in data["recent_research"]}
    stack = by_name["heterogeneous OOF stack"]
    assert stack["chosen_arm"] == "zmean_seq__all"
    assert stack["combined_gain"] > 0.004
    assert stack["cart_hit_gain"] == 64
    assert stack["nonnegative_combined_folds"] == 5

    fresh = by_name["Fresh Selection V2"]
    assert fresh["decision"] == "PASS"
    assert fresh["sessions"] == 20_000
    assert fresh["combined_gain"] >= 0.0025
    assert fresh["click_hit_gain"] >= 0
    assert fresh["cart_hit_gain"] >= 10
    assert fresh["paired_95_interval"][0] > 0
    assert fresh["first_half_gain"] >= 0
    assert fresh["second_half_gain"] >= 0


def test_final_reserve_gate_is_frozen() -> None:
    gate = json.loads(STATUS.read_text())["active_frontier"]["frozen_gate"]
    assert gate["min_combined_weighted_gain"] == 0.0025
    assert gate["min_click_hit_gain"] == 0
    assert gate["min_cart_hit_gain"] == 200
    assert gate["bootstrap_95_lower_bound_gt_zero"] is True
    assert gate["min_nonnegative_quarters"] == 3
    assert gate["worst_quarter_min_gain"] == -0.0005


def test_current_employer_facing_docs_do_not_compare_to_external_score_targets() -> None:
    current_docs = [
        ROOT / "README.md",
        FRONTIER / "README.md",
        FRONTIER / "08_neural_objective_frontier.md",
        FRONTIER / "09_click_cart_selection_protocol.md",
        FRONTIER / "10_contextual_and_sequence_frontier.md",
        FRONTIER / "11_heterogeneous_stack_and_fresh_selection.md",
        FRONTIER / "12_reserved_evaluation_v2.md",
    ]
    forbidden = (
        "0.60503",
        "historical private winner",
        "historical private winning benchmark",
        "remaining private-score gap",
        "remaining private gap",
    )
    for path in current_docs:
        text = path.read_text().lower()
        for token in forbidden:
            assert token not in text, f"{token!r} leaked into {path}"


def test_public_boundary_excludes_private_artifacts() -> None:
    data = json.loads(STATUS.read_text())
    boundary = " ".join(data["publication_boundary"]).lower()
    for token in ("raw data", "row-level labels", "row-level", "cohort ids", "private checkpoints"):
        assert token in boundary
