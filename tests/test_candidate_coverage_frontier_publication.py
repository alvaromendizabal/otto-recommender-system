from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = json.loads(
    (ROOT / "reports/research/current_frontier_20261005.json").read_text(
        encoding="utf-8"
    )
)
DOC = (
    ROOT / "research/frontier/12_corrected_comparator_candidate_coverage.md"
).read_text(encoding="utf-8")


def test_corrected_comparator_rejects_challenger() -> None:
    corrected = STATUS["research_frontier"]["corrected_comparator"]
    assert corrected["top20_membership_mismatches"] == 0
    assert corrected["weighted_gain"] == 0.0039569
    assert corrected["click_hit_gain"] == -3529
    assert corrected["cart_hit_gain"] == 1922
    assert corrected["decision"] == "rejected_click_gate"
    assert "failed qualification and was rejected" in DOC.lower()


def test_candidate_coverage_diagnosis_is_arithmetically_consistent() -> None:
    diag = STATUS["research_frontier"]["time_controlled_diagnostics"]
    assert diag["cart_denominator"] == 4778
    assert diag["strongest_ranked_cart_hits"] == 2181
    assert diag["original_candidate_pool_ceiling_hits"] == 2824
    assert diag["within_pool_misses"] == 2824 - 2181
    assert diag["outside_pool_misses"] == 4778 - 2824
    assert diag["sampled_cart_output_catalog_ceiling_hits"] == 3005


def test_current_frontier_keeps_public_private_boundary() -> None:
    public = STATUS["public_reproduction_scope"]
    excluded = set(public["excludes"])
    assert "row-level predictions" in excluded
    assert "private runners" in excluded
    assert "full checkpoints" in excluded
    assert "credentials" in excluded

    serialized = (json.dumps(STATUS, sort_keys=True) + DOC).lower()
    for token in (
        "/home/" + "sagemaker-user",
        "s3://",
        ".pyz",
        "aws_" + "access_key",
        "secret_" + "access_key",
    ):
        assert token not in serialized
