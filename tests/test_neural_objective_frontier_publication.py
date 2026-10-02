"""Publication contracts for the October 2 neural/objective frontier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _status() -> dict[str, Any]:
    return json.loads((ROOT / "reports/research/frontier_status_20261002.json").read_text())


def test_verified_kaggle_incumbent_is_unchanged() -> None:
    c = _status()["competition"]
    assert c["incumbent"]["ref"] == 56542128
    assert c["incumbent"]["private"] == 0.57586
    assert c["incumbent"]["public"] == 0.57601
    assert c["remaining_private_gap"] == 0.02917
    assert c["newer_submission_published"] is False


def test_order_source_and_queryformer_branches_are_closed_without_selection() -> None:
    completed = {x["name"]: x for x in _status()["completed_since_20261001"]}
    source = completed["order heterogeneous source-aware fusion"]
    assert source["decision"] == "STOP_SOURCE_FUSION_FITTING"
    assert source["stable_pool"]["net_order_hits"] == 0
    assert source["expanded_pool"]["net_order_hits"] == -40
    qf = completed["QueryFormer-style order field-sequence interaction"]
    assert qf["field400"]["net_order_hits"] == 8
    assert qf["qf400"]["net_order_hits"] == -10
    assert qf["qf1200"]["net_order_hits"] == -44
    assert qf["selection_targets_read"] is False


def test_click_cart_fitting_result_qualifies_only_for_selection() -> None:
    completed = {x["name"]: x for x in _status()["completed_since_20261001"]}
    cc = completed["objective-specific click/cart multisource ranking"]
    assert cc["decision"] == "PROMOTE_TO_SELECTION"
    assert cc["candidate_pool"] == 400
    assert cc["feature_count"] == 589
    assert cc["clicks"]["net_hits"] == 186
    assert cc["carts"]["net_hits"] == 53
    assert cc["combined_weighted_gain"] == 0.00353469
    assert cc["nonnegative_folds"] == 5
    assert cc["worst_fold_gain"] > 0
    assert cc["selection_targets_read"] is False
    assert cc["evaluation_targets_read"] is False


def test_active_selection_does_not_claim_evaluation_or_submission() -> None:
    active = {x["name"]: x for x in _status()["active_frontier"]}
    sel = active["click/cart multisource selection"]
    assert sel["selection_targets_read"] is False
    assert sel["evaluation_targets_read"] is False
    assert sel["submission_attempted"] is False
    assert sel["selection_gate"]["min_combined_weighted_gain"] == 0.0025
    assert sel["bootstrap_replicates"] == 2000


def test_execution_ledger_preserves_failure_science_distinction() -> None:
    ledger = json.loads((ROOT / "reports/research/execution_ledger_20261002.json").read_text())
    tally = ledger["since_explicit_tally_directive"]
    assert tally["successful_intended_executions"] == 14
    assert tally["blocking_execution_failures"] == 13
    assert tally["valid_negative_experiments_among_successes"] == 12
    assert tally["decisive_negative_scientific_results_recovered_from_failed_executions"] == 1
    cumulative = ledger["cumulative_owner_run_outcomes"]
    assert cumulative["successful_scientific_executions"] == 34
    assert cumulative["blocking_execution_failures"] == 29
    costs = ledger["active_compute_cost_estimate_usd"]
    assert costs["successful_scientific_compute_lower_bound"] == 84.399
    assert costs["failed_blocking_compute_known_lower_bound"] == 11.1922
    assert costs["known_accounted_active_compute_lower_bound"] == 96.5332
