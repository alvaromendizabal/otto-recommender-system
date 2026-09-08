"""Historical retrieval independently rejects future fitting and corrupt recovery."""

from __future__ import annotations

import json
import logging

import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.retrievers import build_retrievers


def fit(tmp_path, *, late=False):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    history = corpus / "history.parquet"
    pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "aid": [10, 20, 30, 10, 20],
            "ts": [10, 20, 30, 40, 200 if late else 50],
            "event_type": [0, 1, 2, 0, 1],
            "event_index": [0, 1, 2, 0, 1],
        }
    ).write_parquet(history)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {"protocol": {"history_end": 100}, "files": {"history.parquet": sha256_file(history)}}
        )
    )
    config = {
        "history_tail": 30,
        "position_window": 1,
        "source_partitions": 2,
        "neighbors_per_score": 2,
        "time_window_hours": 24,
    }
    return build_retrievers(
        corpus,
        tmp_path / "retrieval",
        config,
        logger=logging.getLogger("retrieval-test"),
        threads=1,
        memory_gib=1,
    )


def test_graph_scores_equal_hand_calculation_and_resume_is_verified(tmp_path):
    report = fit(tmp_path)
    graph = pl.read_parquet(tmp_path / "retrieval/parts/*.parquet")
    pairs = {(r["source_aid"], r["target_aid"]) for r in graph.iter_rows(named=True)}
    assert pairs == {(10, 20), (20, 10), (20, 30), (30, 20)}
    row = graph.filter((pl.col("source_aid") == 10) & (pl.col("target_aid") == 20))
    assert row["time_score"][0] == pytest.approx(2, rel=1e-5)
    assert row["cart_score"][0] == pytest.approx(6, rel=1e-5)
    assert row["order_score"][0] == pytest.approx(4, rel=1e-5)
    stats = pl.read_parquet(tmp_path / "retrieval/history_statistics.parquet")
    assert stats.filter(pl.col("aid") == 20)["hist_carts_all"][0] == 2
    assert report["query_labels_used"] is False
    part = tmp_path / "retrieval/parts/part-000.parquet"
    modified = part.stat().st_mtime_ns
    assert fit(tmp_path)["reused_parts"] == 2
    assert part.stat().st_mtime_ns == modified
    part.write_bytes(b"truncated")
    assert fit(tmp_path)["reused_parts"] == 1
    assert sha256_file(part) == report["files"]["parts/part-000.parquet"]


def test_future_history_is_rejected_before_any_retriever_fits(tmp_path):
    with pytest.raises(ValueError, match="crosses"):
        fit(tmp_path, late=True)
    assert not (tmp_path / "retrieval/history_statistics.parquet").exists()
