from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
FRONTIER = (ROOT / "research/frontier/README.md").read_text(encoding="utf-8")
INTEGRITY = (ROOT / "research/frontier/11_validation_integrity_reconciliation.md").read_text(encoding="utf-8")
STATUS = json.loads((ROOT / "reports/research/current_frontier_20261005.json").read_text(encoding="utf-8"))


def test_verified_champion_is_consistent() -> None:
    champion = STATUS["verified_competition_champion"]
    assert champion["private_score"] == 0.57586
    assert champion["public_score"] == 0.57601
    assert str(champion["submission_reference"]) in INTEGRITY
    assert "0.57586 private / 0.57601 public" in README
    assert "0.57586 private / 0.57601 public" in FRONTIER


def test_newer_stack_is_not_publicly_promoted() -> None:
    assert STATUS["research_frontier"]["click_cart_stack_status"] == "research_comparator_reconciliation"
    assert STATUS["research_frontier"]["later_promotion_interpretation"] == "withdrawn_pending_corrected_comparator"
    assert "not currently promoted" in README
    assert "withdrawn pending comparator reconciliation" in INTEGRITY.lower()


def test_public_status_has_no_private_execution_material() -> None:
    serialized = json.dumps(STATUS, sort_keys=True).lower()
    for token in ("/home/sagemaker-user", "s3://", ".pyz", "aws_access_key", "secret_access_key", "session_ids"):
        assert token not in serialized


def test_publication_boundary_is_explicit() -> None:
    excluded = set(STATUS["public_reproduction_scope"]["excludes"])
    assert "row-level predictions" in excluded
    assert "private runners" in excluded
    assert "full checkpoints" in excluded
    assert "credentials" in excluded
