from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
OVERVIEW = (ROOT / "docs/EMPLOYER_OVERVIEW.md").read_text(encoding="utf-8")
ARCH = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
PORTFOLIO = (ROOT / "docs/PORTFOLIO.md").read_text(encoding="utf-8")
STATUS = json.loads(
    (ROOT / "reports/research/current_frontier_20261005.json").read_text(
        encoding="utf-8"
    )
)

EMPLOYER_SURFACES = "\n".join((README, OVERVIEW, ARCH, PORTFOLIO)).lower()


def test_first_minute_story_is_present() -> None:
    readme = README.lower()
    assert "60-second overview" in readme
    assert "what i owned" in readme
    assert "216.7m" in readme
    assert "1.67m" in readme
    assert "5.02m" in readme
    assert "0.57586 private / 0.57601 public" in readme


def test_three_review_depths_are_explicit() -> None:
    entrypoints = STATUS["employer_review"]["recommended_entrypoints"]
    assert entrypoints["60_seconds"] == "docs/EMPLOYER_OVERVIEW.md"
    assert "docs/ARCHITECTURE.md" in entrypoints["5_minutes"]
    assert "docs/REPRODUCIBILITY.md" in entrypoints["deep_dive"]
    assert "research/frontier/12_corrected_comparator_candidate_coverage.md" in (
        entrypoints["deep_dive"]
    )


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


def test_current_research_state_is_consistent() -> None:
    frontier = STATUS["research_frontier"]
    assert frontier["click_cart_stack_status"] == "rejected_after_corrected_comparator"
    assert "rejected" in OVERVIEW.lower()
    assert "candidate availability" in PORTFOLIO.lower()
    assert "candidate availability" in README.lower()
