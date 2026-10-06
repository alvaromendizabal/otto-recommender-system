from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
FRONTIER = (ROOT / "research/frontier/README.md").read_text(encoding="utf-8")
INTEGRITY = (
    ROOT / "research/frontier/11_validation_integrity_reconciliation.md"
).read_text(encoding="utf-8")
COVERAGE = (
    ROOT / "research/frontier/12_corrected_comparator_candidate_coverage.md"
).read_text(encoding="utf-8")
STATUS = json.loads(
    (ROOT / "reports/research/current_frontier_20261005.json").read_text(
        encoding="utf-8"
    )
)


def test_verified_champion_is_consistent() -> None:
    champion = STATUS["verified_competition_champion"]
    assert champion["private_score"] == 0.57586
    assert champion["public_score"] == 0.57601
    assert str(champion["submission_reference"]) in INTEGRITY
    assert "0.57586 private / 0.57601 public" in README
    assert "0.57586 private / 0.57601 public" in FRONTIER


def test_corrected_challenger_is_rejected_not_promoted() -> None:
    frontier = STATUS["research_frontier"]
    corrected = frontier["corrected_comparator"]
    assert frontier["click_cart_stack_status"] == "rejected_after_corrected_comparator"
    assert corrected["top20_membership_mismatches"] == 0
    assert corrected["click_hit_gain"] == -3529
    assert corrected["cart_hit_gain"] == 1922
    assert corrected["decision"] == "rejected_click_gate"
    assert "corrected click/cart challenger is **rejected**" in README.lower()
    assert "failed qualification and was rejected" in INTEGRITY.lower()


def test_candidate_availability_is_current_frontier() -> None:
    diag = STATUS["research_frontier"]["time_controlled_diagnostics"]
    assert diag["cart_denominator"] == 4778
    assert diag["strongest_ranked_cart_hits"] == 2181
    assert diag["original_candidate_pool_ceiling_hits"] == 2824
    assert diag["outside_pool_misses"] == 1954
    assert "candidate availability" in README.lower()
    assert "candidate availability" in FRONTIER.lower()
    assert "candidate availability" in COVERAGE.lower()


def test_public_status_has_no_private_execution_material() -> None:
    serialized = json.dumps(STATUS, sort_keys=True).lower()
    tokens = (
        "/home/" + "sagemaker-user",
        "s3://",
        ".pyz",
        "aws_" + "access_key",
        "secret_" + "access_key",
        "session_" + "ids",
    )
    for token in tokens:
        assert token not in serialized


def test_publication_boundary_is_explicit() -> None:
    excluded = set(STATUS["public_reproduction_scope"]["excludes"])
    assert "row-level predictions" in excluded
    assert "private runners" in excluded
    assert "full checkpoints" in excluded
    assert "credentials" in excluded
