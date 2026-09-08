"""Raw events to frozen evaluation, including completely unretrievable labels."""

from __future__ import annotations

import json
import logging

import polars as pl
import pytest

from otto_recsys.research.evaluation import run_evaluation, verify_seal
from otto_recsys.research.materialize import model_cache
from otto_recsys.research.protocol import TemporalProtocol, build_temporal_corpus
from otto_recsys.research.retrievers import build_retrievers
from otto_recsys.research.study import run_ablations


def build_evaluated_study(tmp_path):
    logger = logging.getLogger("evaluation-test")
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for period, base in enumerate((10, 110, 210, 310)):
        for i in range(80):
            for index, kind in enumerate((0, 0, 0, 1, 2)):
                aid = (i + index) % 64 + 1
                if period == 3 and kind == 2:
                    aid = 999  # every held-out order is outside the historical candidate catalog
                records.append((period * 1000 + i, aid, base + index, kind, index))
    pl.DataFrame(
        records, schema=["session", "aid", "ts", "event_type", "event_index"], orient="row"
    ).write_parquet(source / "part-0000.parquet")
    build_temporal_corpus(
        source,
        tmp_path / "corpus",
        TemporalProtocol(100, 200, 300, 400),
        logger=logger,
        threads=1,
        memory_gib=1,
    )
    build_retrievers(
        tmp_path / "corpus",
        tmp_path / "retrieval",
        {
            "history_tail": 30,
            "position_window": 3,
            "source_partitions": 2,
            "neighbors_per_score": 40,
            "time_window_hours": 24,
        },
        logger=logger,
        threads=1,
        memory_gib=1,
    )
    names = (
        "source_time_score",
        "hist_all_all",
        "repeat_all_n50_count",
        "query_events",
        "intent_repeat_carts",
        "graph_time_all_n3_uniform_sum",
    )
    (tmp_path / "screening").mkdir()
    (tmp_path / "screening/selection.json").write_text(json.dumps({"retained": names}))
    for role in ("fit", "selection"):
        model_cache(
            tmp_path / "corpus",
            tmp_path / "retrieval",
            tmp_path / f"{role}_cache",
            role=role,
            names=names,
            budget=400,
            negatives=20,
            seed=4,
            workers=1,
            logger=logger,
        )
    run_ablations(tmp_path, {"rounds": 5, "patience": 3, "threads": 1, "seed": 4}, logger=logger)
    report = run_evaluation(
        tmp_path, seed=4, workers=1, threads=1, logger=logger, bootstrap_replicates=100
    )
    return report


def test_raw_event_pipeline_keeps_unseen_truth_and_recovers_evaluation_parts(tmp_path):
    logger = logging.getLogger("evaluation-test")
    report = build_evaluated_study(tmp_path)
    assert report["sessions"] == 80
    for model in ("selected", "core", "fusion"):
        orders = report["scores"][model]["objectives"]["orders"]
        assert orders["denominator"] == 80
        assert orders["hits"] == orders["recall_at_20"] == 0
    assert report["candidate_frontier"]["400"]["objectives"]["orders"]["recall_at_20"] == 0
    assert (
        run_evaluation(
            tmp_path, seed=4, workers=1, threads=1, logger=logger, bootstrap_replicates=100
        )
        == report
    )
    part = tmp_path / "evaluation/part-0000.parquet"
    part.unlink()
    recovered = run_evaluation(
        tmp_path, seed=4, workers=1, threads=1, logger=logger, bootstrap_replicates=100
    )
    assert recovered["scores"] == report["scores"]
    assert recovered["files"] == report["files"]
    seal = verify_seal(tmp_path)
    (tmp_path / seal["models"]["clicks"]["path"]).write_text("corrupt model")
    with pytest.raises(ValueError, match="checksum"):
        verify_seal(tmp_path)
