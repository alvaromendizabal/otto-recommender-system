"""Publication contracts for the October 1 candidate/model frontier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _status() -> dict[str, Any]:
    return json.loads((ROOT / "reports/research/frontier_status_20261001.json").read_text())


def test_incumbent_stays_verified_until_a_new_submission_exists() -> None:
    c = _status()["competition"]
    assert c["incumbent"]["ref"] == 56542128
    assert c["incumbent"]["private"] == 0.57586
    assert c["incumbent"]["public"] == 0.57601
    assert c["remaining_private_gap"] == 0.02917
    assert c["newer_submission_published"] is False


def test_xgb_reserved_evaluation_is_closed_without_deployment() -> None:
    rows = {r["name"]: r for r in _status()["completed_since_20260929"]}
    xgb = rows["transition-source XGBoost and fixed blends"]
    assert xgb["decision"] == "STOP_TRANSITION_XGB_EVALUATION_GATE"
    assert xgb["weighted_gain"] == 0.0026653974297953242
    assert xgb["net_additional_order_hits"] == 308
    assert xgb["passed"] is False
    assert xgb["submission_attempted"] is False


def test_candidate_ceiling_is_not_presented_as_model_gain() -> None:
    rows = {r["name"]: r for r in _status()["completed_since_20260929"]}
    mf = rows["MF plus v42 candidate-ceiling frontier"]
    assert mf["role"].endswith("candidate oracle only")
    assert mf["weighted_ceiling_gain"] == 0.03221529913505483
    assert mf["recovered_targets"]["orders"] == 69
    ranker = rows["expanded-pool candidate-aware LambdaRank"]
    assert ranker["weighted_gain"] < 0


def test_complete_third_place_frontier_preserves_frozen_gate() -> None:
    rows = {r["name"]: r for r in _status()["completed_since_20260929"]}
    final = rows["785-feature complete third-place CPU frontier"]
    assert final["chosen_arm"] == "zmean"
    assert final["weighted_gain"] == 0.0032340114735465966
    assert final["required_weighted_gain"] == 0.0035
    assert final["hit_gain"] == {"clicks": 54, "carts": 3, "orders": 17}
    assert final["nonnegative_folds"] == 4
    assert final["passed"] is False
    assert final["selection_targets_read"] is False


def test_ledger_keeps_scientific_and_engineering_outcomes_separate() -> None:
    ledger = json.loads((ROOT / "reports/research/execution_ledger_20261001.json").read_text())
    outcomes = ledger["owner_run_outcomes"]
    assert outcomes["successful_as_designed"] == 28
    assert outcomes["engineering_command_environment_ui_resource_failures"] == 22
    assert outcomes["nonblocking_post_result_or_ui_incidents"] == 6
    costs = ledger["active_compute_cost_estimate_usd"]
    assert costs["completed_successful_runs_lower_bound"] == 80.6755
    assert costs["completed_failed_runs_known_lower_bound"] == 10.2559
    assert costs["known_accounted_active_compute_lower_bound"] == 91.8734


def test_gpu_frontier_is_future_work_not_a_claimed_result() -> None:
    active = {r["name"]: r for r in _status()["active_frontier"]}
    gpu = active["first-place v29/v27 GPU adaptation"]
    assert gpu["scientific_result_published"] is False
    assert gpu["submission_attempted"] is False
