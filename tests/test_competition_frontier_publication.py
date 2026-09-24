"""Contracts for the current competition-frontier publication."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontier_snapshot_preserves_verified_incumbent_and_gap() -> None:
    status = json.loads((ROOT / "reports/research/frontier_status_20260923.json").read_text())
    c = status["competition"]
    assert c["incumbent"]["ref"] == 56504354
    assert c["incumbent"]["private"] == 0.5714
    assert c["incumbent"]["public"] == 0.57166
    assert c["incumbent"]["sha256"] == "4b6770f3b750e3c265ad1f2094f6caa8791d31b93fd90656ca14761e55ed8c0d"
    assert c["remaining_private_gap"] == 0.03363
    assert c["threshold14"]["private"] < c["incumbent"]["private"]


def test_closed_hypotheses_are_not_presented_as_promotions() -> None:
    status = json.loads((ROOT / "reports/research/frontier_status_20260923.json").read_text())
    decisions = {row["name"]: row["decision"] for row in status["closed_hypotheses"]}
    assert decisions["neural-aware order reranker"].startswith("STOP_")
    assert decisions["neural residual insertion"].startswith("STOP_")
    assert decisions["three-seed LightGBM order ensemble"].startswith("STOP_")
    assert decisions["cart threshold refinement"].startswith("CLOSE_")


def test_scorecard_is_executed_and_has_post_figure_sentinel() -> None:
    nb = json.loads((ROOT / "research/frontier/04_competition_frontier.ipynb").read_text())
    code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert [cell["execution_count"] for cell in code] == [1, 2, 3, 4]
    assert "COMPETITION_FRONTIER_SCORECARD_COMPLETE" in str(code[-1]["outputs"])
