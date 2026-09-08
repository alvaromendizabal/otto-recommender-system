"""Exercise the complete ablation-to-seal transition without reading evaluation labels."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research import study


def test_study_seals_native_models_after_complete_selection_queries(tmp_path, monkeypatch):
    names = (
        "source_time_score",
        "hist_all_all",
        "repeat_all_n50_count",
        "query_events",
        "intent_repeat_carts",
        "graph_time_all_n3_uniform_sum",
    )
    rng = np.random.default_rng(4)
    group_size = 40
    sessions_per_role = 120
    query_ids = {
        "fit": np.arange(sessions_per_role),
        "selection": np.arange(sessions_per_role, 2 * sessions_per_role),
    }

    class QueryFixture:
        def __init__(self, corpus, role):
            assert role in query_ids, "evaluation queries must remain inaccessible before sealing"
            self.session = query_ids[role]
            self.denominators = np.ones((sessions_per_role, 3), dtype=int)
            self.manifest = {"input_id": "synthetic-chronological-fixture"}

    monkeypatch.setattr(study, "Queries", QueryFixture)
    for role, ids in query_ids.items():
        directory = tmp_path / f"{role}_cache"
        directory.mkdir()
        y = np.tile(np.r_[1, np.zeros(group_size - 1)], sessions_per_role).astype(np.int8)
        x = np.column_stack(
            [
                y + rng.normal(0, 0.05, y.size),
                y + rng.normal(0, 0.2, y.size),
                rng.random(y.size),
                np.ones(y.size),
                rng.random(y.size),
                rng.random(y.size),
            ]
        )
        frame = pl.DataFrame(x.astype(np.float32), schema=list(names)).with_columns(
            pl.Series("session", np.repeat(ids, group_size).astype(np.int32)),
            pl.Series("aid", np.tile(np.arange(group_size), sessions_per_role).astype(np.int32)),
            pl.Series(
                "candidate_position",
                np.tile(np.arange(group_size), sessions_per_role).astype(np.int16),
            ),
            *[pl.Series(f"target_{o}", y) for o in ("clicks", "carts", "orders")],
            *[
                pl.Series(f"baseline_{o}", rng.random(y.size).astype(np.float32))
                for o in ("clicks", "carts", "orders")
            ],
        )
        path = directory / "part-0000.parquet"
        frame.write_parquet(path)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "role": role,
                    "features": list(names),
                    "rows": frame.height,
                    "sessions": sessions_per_role,
                    "input_id": role,
                    "parts": [path.name],
                    "files": {path.name: sha256_file(path)},
                }
            )
        )
    (tmp_path / "screening").mkdir()
    (tmp_path / "screening/selection.json").write_text(json.dumps({"retained": names}))
    (tmp_path / "retrieval").mkdir()
    (tmp_path / "retrieval/manifest.json").write_text(json.dumps({"input_id": "synthetic-graph"}))
    report = study.run_ablations(
        tmp_path,
        {"rounds": 5, "patience": 3, "threads": 1, "seed": 4},
        logger=logging.getLogger("study-test"),
    )
    assert report["status"] == "passed" and report["selection_sessions"] == sessions_per_role
    assert len(report["variants"]) == 8
    seal = json.loads((tmp_path / "evaluation_seal.json").read_text())
    assert seal["evaluation_labels_consulted"] is False
    assert seal["seal_id"] == report["seal_id"]
    for model in seal["models"].values():
        assert sha256_file(tmp_path / model["path"]) == model["sha256"]
    assert report["variants"]["core"]["weighted_recall_at_20"] > 0.9
    with pytest.raises(ValueError, match="different experiment"):
        study.run_ablations(
            tmp_path,
            {"rounds": 5, "patience": 3, "threads": 1, "seed": 9},
            logger=logging.getLogger("study-test"),
        )
