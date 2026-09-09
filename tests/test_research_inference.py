"""Frozen native models to complete prediction files on observed events only."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest
from test_research_evaluation import build_evaluated_study

from otto_recsys.research.deployment import prepare_history
from otto_recsys.research.inference import (
    ObservedQueries,
    export_replay,
    run_prediction,
    validate_submission,
)
from otto_recsys.research.retrievers import build_retrievers


def test_frozen_models_generate_complete_submission_and_recover_corrupt_parts(
    tmp_path, monkeypatch
):
    # Synthetic model/recovery fixture; real source rejection has separate tests.
    monkeypatch.setattr(
        "otto_recsys.research.inference.require_competition_input",
        lambda _: {"source": "synthetic-recovery-test"},
    )
    logger = logging.getLogger("inference-test")
    build_evaluated_study(tmp_path)
    test = tmp_path / "test"
    test.mkdir()
    records = [(5000 + i, (i + j) % 64 + 1, 430 + j, j % 3, j) for i in range(20) for j in range(5)]
    frame = pl.DataFrame(
        records, schema=["session", "aid", "ts", "event_type", "event_index"], orient="row"
    )
    frame.write_parquet(test / "part-0000.parquet")
    deployment = tmp_path / "deployment"
    history = prepare_history(
        tmp_path / "source", test, deployment / "corpus", logger=logger, threads=1, memory_gib=1
    )
    assert history["test_sessions"] == 20
    assert history["overlapping_sessions"] == 0
    assert history["history_max_ts"] < history["protocol"]["history_end"]
    build_retrievers(
        deployment / "corpus",
        deployment / "retrieval",
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
    arguments = dict(
        model_root=tmp_path,
        test=test,
        retrieval=deployment / "retrieval",
        output=deployment / "prediction",
        workers=1,
        threads=1,
        logger=logger,
    )
    result = run_prediction(**arguments)
    assert result["sessions"] == 20 and result["rows"] == 60
    assert run_prediction(**arguments)["sha256"] == result["sha256"]
    (deployment / "prediction/part-0000.csv").write_text("partial prediction")
    recovered = run_prediction(**arguments)
    assert recovered["sha256"] == result["sha256"]
    assert recovered["files"] == result["files"]
    replay = export_replay(
        tmp_path,
        test,
        deployment / "retrieval",
        deployment / "prediction",
        tmp_path / "inference_replay",
        sessions=8,
    )
    assert replay["sessions"] == 8
    assert replay["full_prediction"]["sha256"] == result["sha256"]
    assert pl.read_csv(tmp_path / "inference_replay/expected.csv").height == 24
    broken = pl.read_csv(deployment / "prediction/submission.csv.gz").with_columns(
        pl.lit(" ".join(["1"] * 20)).alias("labels")
    )
    broken.write_csv(deployment / "invalid.csv")
    with pytest.raises(ValueError, match="unique"):
        validate_submission(deployment / "invalid.csv", np.arange(5000, 5020))
    frame.with_columns(pl.lit(1).alias("ts")).write_parquet(test / "part-0000.parquet")
    with pytest.raises(ValueError, match="different model or test"):
        run_prediction(**arguments)
    with pytest.raises(ValueError, match="precede disjoint"):
        prepare_history(
            tmp_path / "source",
            test,
            deployment / "invalid_history",
            logger=logger,
            threads=1,
            memory_gib=1,
        )


def test_observed_queries_reject_incomplete_session_event_indices(tmp_path):
    pl.DataFrame(
        {
            "session": [1, 1],
            "aid": [2, 3],
            "ts": [10, 20],
            "event_type": [0, 1],
            "event_index": [0, 2],
        }
    ).write_parquet(tmp_path / "part-0000.parquet")
    with pytest.raises(ValueError, match="contiguous"):
        ObservedQueries(tmp_path)
