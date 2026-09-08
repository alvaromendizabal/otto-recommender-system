"""Parallel generation matches observed-only formulas and verifies recovery."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest
from test_research_protocol import build, make_source

from otto_recsys.research.dataset import Queries
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.materialize import model_cache
from otto_recsys.research.retrievers import build_retrievers


def test_parallel_cache_matches_direct_features_and_reuses_complete_parts(tmp_path):
    logger = logging.getLogger("materialize-test")
    source = make_source(tmp_path / "source")
    corpus, retrieval = tmp_path / "corpus", tmp_path / "retrieval"
    build(source, corpus)
    build_retrievers(
        corpus,
        retrieval,
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
    names = ("source_time_score", "repeat_all_n3_count", "hist_clicks_all")
    output = tmp_path / "features"
    arguments = dict(
        corpus=corpus,
        retrieval=retrieval,
        output=output,
        role="selection",
        names=names,
        budget=4,
        negatives=2,
        seed=2,
        workers=2,
        logger=logger,
    )
    report = model_cache(**arguments)
    part = output / "part-0000.parquet"
    modified = part.stat().st_mtime_ns
    assert model_cache(**arguments)["files"] == report["files"]
    assert part.stat().st_mtime_ns == modified
    actual = pl.read_parquet(part)
    engine = FeatureEngine(retrieval)
    queries = Queries(corpus, "selection")
    prefix = queries.prefix(0)
    candidates = engine.candidates(prefix, 4)
    assert np.array_equal(
        actual.select(names).to_numpy(), engine.transform(prefix, candidates, names)
    )
    assert np.array_equal(actual["aid"].to_numpy(), candidates.aid)
    assert report["sessions"] == queries.session.size
    part.write_bytes(b"interrupted transfer")
    assert model_cache(**arguments)["files"] == report["files"]
    with pytest.raises(ValueError, match="roles"):
        model_cache(**{**arguments, "role": "evaluation"})
