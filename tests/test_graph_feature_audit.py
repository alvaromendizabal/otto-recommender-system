"""The audit catches changed scores, session ledgers, coverage and model bytes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.graph_feature_audit import audit


@pytest.fixture(scope="module")
def completed_study(tmp_path_factory):
    from test_graph_signals import test_complete_feature_only_study_preserves_every_baseline_row

    root = tmp_path_factory.mktemp("graph_audit")
    with pytest.MonkeyPatch.context() as patch:
        test_complete_feature_only_study_preserves_every_baseline_row(root, patch)
    output = root / "graph_features"
    shutil.copyfile(root / "inputs/corpus/queries.parquet", output / "queries.parquet")
    contract = json.loads((output / "study_contract.json").read_text())
    launch = {
        "task": "graph_features", "source_commit": "test_fixture",
        "study": contract["config"],
        "graphs": {
            family: [{"path": "manifest.json", "sha256": digest}]
            for family, digest in contract["graphs"].items()
        },
        "corpus": [{"key": "corpus/queries.parquet",
                    "sha256": sha256_file(output / "queries.parquet")}],
    }
    (output / "launch.json").write_text(json.dumps(launch))
    return output


def test_audit_recomputes_all_sessions_and_model_lineage(completed_study: Path) -> None:
    result = audit(completed_study)
    assert result["status"] == "passed"
    assert result["arms_verified"] == 4 and result["models_verified"] == 12
    assert result["candidate_coverage_equal_per_session_and_objective"] is True
    assert set(result["arms"]["both"]["paired_gain"]["objectives"]) == {
        "clicks", "carts", "orders",
    }


@pytest.mark.parametrize("change", ["score", "session", "coverage", "model", "ledger", "launch"])
def test_audit_rejects_tampering(completed_study: Path, tmp_path: Path, change: str) -> None:
    shutil.copytree(completed_study, tmp_path / "study")
    root = tmp_path / "study"
    if change == "score":
        file = root / "results.json"
        value = json.loads(file.read_text())
        value["arms"]["both"]["weighted_recall_at_20"] += 0.01
        file.write_text(json.dumps(value))
    elif change in ("session", "coverage"):
        file = root / "models/both/selection_statistics.parquet"
        frame = pl.read_parquet(file)
        frame = frame.slice(1) if change == "session" else frame.with_columns(
            (pl.col("coverage_clicks") + 1).alias("coverage_clicks")
        )
        frame.write_parquet(file)
    elif change == "model":
        (root / "models/both/clicks/model.txt").write_text("changed model")
    elif change == "ledger":
        (root / "queries.parquet").write_bytes(b"changed ledger")
    else:
        file = root / "launch.json"
        value = json.loads(file.read_text())
        value["study"]["training"]["seed"] += 1
        file.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit(root)
