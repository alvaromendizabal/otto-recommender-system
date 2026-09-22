"""A partial evaluation may demonstrate recovery, never a final model score."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "research/frontier"


def test_snapshot_counts_and_scope() -> None:
    s = json.loads((ROOT / "training_scale_status.json").read_text())
    assert s["status"] == "PAUSED_CHECKPOINTED"
    assert s["new_models_completed"] == s["new_model_fit_attempts"] == 3
    assert s["evaluation_chunks_completed"] * 64 == s["evaluation_sessions_completed"]
    assert s["evaluation_sessions_completed"] + s["remaining_sessions"] == 16384
    assert s["remaining_sessions"] == 7 * 64
    assert s["final_score"] is None and s["promotion_decision"] is None
    assert not s["submission_produced"] and not s["kaggle_submitted"]
    assert s["scaled_training_rows"] == 1993202
    assert s["order_comparison_queries"]["scaled"] >= 2 * s["order_comparison_queries"]["pilot"]


def test_progress_notebook_outputs_and_sentinel() -> None:
    nb = json.loads((ROOT / "02_training_scale_status.ipynb").read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in cells] == [1, 2, 3, 4]
    outputs = [o for c in cells for o in c["outputs"]]
    assert not any(o["output_type"] == "error" for o in outputs)
    plots = [o["data"] for o in outputs if "application/vnd.plotly.v1+json" in o.get("data", {})]
    assert len(plots) == 2
    for plot in plots:
        assert "image/svg+xml" in plot
        assert plot["application/vnd.plotly.v1+json"]["layout"]["width"] == 900
    assert "SCALE_STATUS_REVIEW_COMPLETE" in str(cells[-1]["outputs"])
    assert plots[0]["application/vnd.plotly.v1+json"]["data"][1]["y"] == [20668, 4548, 2644]
    assert plots[1]["application/vnd.plotly.v1+json"]["data"][0]["x"] == [15936]


def test_snapshot_excludes_private_rows_and_account_locations() -> None:
    text = (ROOT / "training_scale_status.json").read_text()
    forbidden = (
        "session_ids", "train_ids", "evaluation_ids", "access_token",
        "s3://", "arn:aws", "/home/",
    )
    assert all(key not in text for key in forbidden)
