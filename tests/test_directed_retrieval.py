"""Historical retrieval independently rejects future fitting and corrupt recovery."""

from __future__ import annotations

import json
import logging

import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.directed_retrievers import build_retrievers


def fit(tmp_path, *, late=False, direction="symmetric", window=1):
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
        "position_window": window,
        "direction": direction,
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


def test_forward_graph_preserves_sequence_direction_and_longer_horizon(tmp_path):
    fit(tmp_path, direction="forward", window=2)
    graph = pl.read_parquet(tmp_path / "retrieval/parts/*.parquet")
    pairs = set(zip(graph["source_aid"], graph["target_aid"], strict=True))
    assert pairs == {(10, 20), (10, 30), (20, 30)}
    assert graph.filter((pl.col("source_aid") == 10) & (pl.col("target_aid") == 30))["order_score"][
        0
    ] == pytest.approx(4 / 2**0.5, rel=1e-5)
    with pytest.raises(ValueError, match="different fitted-history contract"):
        fit(tmp_path, direction="symmetric", window=2)


def test_invalid_direction_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="direction"):
        fit(tmp_path, direction="backwards")


def test_complete_retrieval_study_preserves_roles_and_recovery(tmp_path, monkeypatch):
    from test_research_protocol import build

    from otto_recsys.research import retrieval_study
    from otto_recsys.research.dataset import Queries
    from otto_recsys.research.materialize import model_cache

    source = tmp_path / "source"
    source.mkdir()
    records = [
        (session, 1 + index, base + index, index % 3, index)
        for session in range(330)
        for base in [0 if session < 80 else 100 if session < 200 else 200 if session < 320 else 300]
        for index in range(40)
    ]
    pl.DataFrame(
        records, schema=["session", "aid", "ts", "event_type", "event_index"], orient="row"
    ).write_parquet(source / "part-000000.parquet")
    inputs, output = tmp_path / "inputs", tmp_path / "study"
    inputs.mkdir()
    build(source, inputs / "corpus")
    graph = {
        "history_tail": 30,
        "position_window": 2,
        "source_partitions": 2,
        "neighbors_per_score": 3,
        "time_window_hours": 24,
    }
    logger = logging.getLogger("retrieval-study-test")
    build_retrievers(
        inputs / "corpus", inputs / "retrieval", graph, logger=logger, threads=1, memory_gib=1
    )
    names = ("source_time_score", "repeat_all_n3_count", "hist_clicks_all")
    for role in ("fit", "selection"):
        model_cache(
            inputs / "corpus",
            inputs / "retrieval",
            inputs / f"{role}_cache",
            role=role,
            names=names,
            budget=400,
            negatives=60,
            seed=2,
            workers=1,
            logger=logger,
        )

    def development_queries(corpus, role):
        assert role in {"fit", "selection"}, "evaluation access forbidden"
        return Queries(corpus, role)

    monkeypatch.setattr(retrieval_study, "Queries", development_queries)
    config = {
        "baseline_features": list(names),
        "retrieval": graph,
        "graph_threads": 1,
        "graph_memory_gib": 1,
        "feature_workers": 1,
        "training": {"rounds": 2, "patience": 1, "threads": 1, "seed": 2},
    }
    published = []
    result = retrieval_study.run_study(
        inputs, output, config, logger=logger, publish=published.append
    )
    assert result["status"] == "passed"
    assert set(result["arms"]) == {"baseline", "wide_symmetric", "wide_forward"}
    assert result["arms"]["baseline"]["objectives"]["orders"]["denominator"] > 0
    assert (output / "wide_forward/retrieval/parts/part-000.parquet") in published
    assert (
        retrieval_study.run_study(inputs, output, config, logger=logger, publish=published.append)
        == result
    )
    with pytest.raises(ValueError, match="different contract"):
        retrieval_study.run_study(
            inputs,
            output,
            {**config, "feature_workers": 2},
            logger=logger,
            publish=published.append,
        )
