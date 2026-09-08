"""Complete official metrics, paired uncertainty, and native-model recovery."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pytest

from otto_recsys.research.metrics import official_score, one_query, paired_bootstrap, ranked_hits
from otto_recsys.research.training import fit_model


def test_ranking_metrics_count_missing_candidates_and_resolve_ties_by_item():
    statistics = one_query(np.array([8, 2, 5]), np.array([2, 9]), np.ones(3), k=1)
    assert statistics["hits"] == statistics["denominator"] == statistics["candidate_hits"] == 1
    assert statistics["mrr"] == statistics["ndcg"] == 1
    missing = one_query(np.array([], dtype=np.int64), np.array([9]), np.array([]))
    assert missing["hits"] == 0 and missing["denominator"] == 1
    assert official_score(np.array([[1, 0, 1], [0, 1, 0]]), np.ones((2, 3), dtype=int))[
        "weighted_recall_at_20"
    ] == pytest.approx(0.5)
    scores = np.ones(6)
    aids = np.array([3, 2, 1, 6, 4, 5])
    target = np.array([0, 0, 1, 0, 1, 0])
    assert np.array_equal(ranked_hits(scores, aids, target, np.array([3, 3]), 1), [1, 1])
    assert np.array_equal(ranked_hits(scores, aids, target, np.array([2, 4]), 1), [0, 1])
    with pytest.raises(ValueError, match="capped"):
        official_score(np.ones((2, 3)), np.zeros((2, 3)))


def test_paired_intervals_preserve_pairing_and_are_reproducible():
    denominator = np.ones((100, 3), dtype=int)
    hits = np.tile([[1, 0, 1], [0, 1, 0]], (50, 1))
    same = paired_bootstrap(hits, hits, denominator, seed=7, replicates=100)
    assert same["gain_interval"] == [0, 0]
    better = paired_bootstrap(hits, np.zeros_like(hits), denominator, seed=7, replicates=100)
    assert better["gain_interval"][0] > 0
    assert better == paired_bootstrap(
        hits, np.zeros_like(hits), denominator, seed=7, replicates=100
    )


def test_trained_ranker_improves_control_and_restores_corrupt_published_model(tmp_path):
    rng = np.random.default_rng(4)
    groups = np.full(160, 40)
    y = np.tile(np.r_[1, np.zeros(39)], 160)
    x = np.column_stack([y + rng.normal(0, 0.05, y.size), rng.normal(size=y.size)]).astype(
        np.float32
    )
    aids = np.tile(np.arange(40), 160)
    arguments = dict(
        fit_x=x,
        fit_y=y,
        fit_groups=groups,
        valid_x=x,
        valid_y=y,
        valid_aids=aids,
        valid_groups=groups,
        denominator=160,
        names=("signal", "noise"),
        directory=tmp_path,
        lineage={"fit_role": "fit", "selection_role": "selection", "fixture": "synthetic"},
        config={"threads": 1, "seed": 4, "rounds": 10, "patience": 5},
        logger=logging.getLogger("research-training-test"),
    )
    state = fit_model(**arguments)
    model = lgb.Booster(model_file=str(tmp_path / "model.txt"))
    score = ranked_hits(model.predict(x, num_threads=1), aids, y, groups).sum() / 160
    assert score > 0.9
    (tmp_path / "model.txt").write_text("incomplete publication")
    reused = fit_model(**arguments)
    assert reused["model_sha256"] == state["model_sha256"]
    assert reused["features"] == state["features"]
    with pytest.raises(ValueError, match="different experiment"):
        fit_model(**{**arguments, "config": {**arguments["config"], "seed": 8}})
