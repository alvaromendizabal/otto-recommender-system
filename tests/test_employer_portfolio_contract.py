from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
OVERVIEW = (ROOT / "docs/EMPLOYER_OVERVIEW.md").read_text(encoding="utf-8")
ARCH = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
PORTFOLIO = (ROOT / "docs/PORTFOLIO.md").read_text(encoding="utf-8")
STATUS = json.loads((ROOT / "reports/research/current_frontier_20261005.json").read_text(encoding="utf-8"))

EMPLOYER_SURFACES = "\n".join((README, OVERVIEW, ARCH, PORTFOLIO)).lower()


def test_first_minute_story_is_present() -> None:
    assert "60-second overview" in README
    assert "what i owned" in README.lower()
    assert "216.7m" in README
    assert "1.67m" in README
    assert "5.02m" in README
    assert "0.57586 private / 0.57601 public" in README


def test_three_review_depths_are_explicit() -> None:
    entrypoints = STATUS["employer_review"]["recommended_entrypoints"]
    assert entrypoints["60_seconds"] == "docs/EMPLOYER_OVERVIEW.md"
    assert "docs/ARCHITECTURE.md" in entrypoints["5_minutes"]
    assert "docs/REPRODUCIBILITY.md" in entrypoints["deep_dive"]


def test_no_score_chasing_language_on_employer_surfaces() -> None:
    forbidden = (
        "beat " + "the top",
        "top " + "score",
        "trying " + "to beat",
        "close the gap " + "to the top",
    )
    for phrase in forbidden:
        assert phrase not in EMPLOYER_SURFACES


def test_publication_boundary_remains_private() -> None:
    for phrase in (
        "/home/" + "sagemaker-user",
        "aws_" + "access_key",
        "secret_" + "access_key",
        "private runner " + "command",
    ):
        assert phrase not in EMPLOYER_SURFACES


def test_current_challenger_status_is_consistent() -> None:
    assert STATUS["research_frontier"]["click_cart_stack_status"] == "research_comparator_reconciliation"
    assert "not deployed" in OVERVIEW.lower() or "blocked" in OVERVIEW.lower()
    assert "comparator reconciliation" in PORTFOLIO.lower()
