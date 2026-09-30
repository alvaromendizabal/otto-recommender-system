"""Publication contracts for the September 29 transition/model frontier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _status() -> dict[str, Any]:
    return json.loads(
        (ROOT / "reports/research/frontier_status_20260929.json").read_text()
    )


def test_verified_incumbent_is_not_replaced_by_in_progress_research() -> None:
    status = _status()
    c = status["competition"]
    assert c["incumbent"]["ref"] == 56542128
    assert c["incumbent"]["private"] == 0.57586
    assert c["incumbent"]["public"] == 0.57601
    assert c["remaining_private_gap"] == 0.02917


def test_completed_frontier_branches_preserve_frozen_decisions() -> None:
    completed = {row["name"]: row for row in _status()["completed_since_previous_snapshot"]}

    dense = completed["dense neural interaction and score stack"]
    assert dense["decision"].startswith("STOP_")
    assert dense["best_arm"]["net_order_hits"] == 7
    assert dense["best_arm"]["weighted_gain"] == 0.000234716
    assert dense["selection_labels_read"] is False

    transition = completed["transition and source representation"]
    assert transition["decision"].startswith("STOP_")
    assert transition["feature_count"] == 496
    assert transition["best_arm"]["net_order_hits"] == 37
    assert transition["best_arm"]["weighted_gain"] == 0.001240639
    assert transition["required_weighted_gain"] == 0.0015
    assert transition["selection_labels_read"] is False


def test_active_xgb_evaluation_is_explicitly_unfinished() -> None:
    active = {row["name"]: row for row in _status()["active_frontier"]}
    xgb = active["transition-source XGBoost and fixed blends"]
    assert xgb["fitting_gate_passed"] is True
    assert xgb["selection_sessions_completed"] == 20_000
    assert xgb["selection_gate_passed"] is True
    assert xgb["evaluation_sessions_total"] == 432_492
    assert 0 < xgb["last_observed_evaluation_sessions"] < xgb["evaluation_sessions_total"]
    assert xgb["final_evaluation_result_published"] is False
    assert xgb["submission_attempted"] is False


def test_execution_ledger_excludes_active_run_from_completed_counts() -> None:
    ledger = json.loads(
        (ROOT / "reports/research/execution_ledger_20260929.json").read_text()
    )
    outcomes = ledger["owner_run_outcomes"]
    assert outcomes["successful_as_designed"] == 20
    assert outcomes["engineering_command_environment_ui_resource_failures"] == 16
    assert outcomes["nonblocking_post_result_or_ui_incidents"] == 3
    costs = ledger["active_compute_cost_estimate_usd"]
    assert costs["completed_successful_runs"] == 65.418989
    assert costs["completed_failed_runs_and_incidents"] == 9.835856
    assert costs["current_in_progress_run_last_observed_usd"] == 4.1734
