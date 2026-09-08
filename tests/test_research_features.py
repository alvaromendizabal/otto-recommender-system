"""Independent feature formulas, prefix availability and selective computation."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.features import FeatureEngine, Prefix, feature_catalog
from otto_recsys.research.retrievers import HISTORY_HOURS


def engine(tmp_path):
    root = tmp_path / "retrieval"
    (root / "parts").mkdir(parents=True)
    stats = {"aid": [1, 2, 3], "history_first_ts": [0, 0, 0], "history_last_ts": [90, 90, 90]}
    for action in ("all", "clicks", "carts", "orders"):
        for horizon in (0, *HISTORY_HOURS):
            name = f"hist_{action}_h{horizon}" if horizon else f"hist_{action}_all"
            stats[name] = [20, 10, 5]
    pl.DataFrame(stats).write_parquet(root / "history_statistics.parquet")
    pl.DataFrame(
        {
            "source_aid": [1, 1, 2],
            "target_aid": [2, 3, 3],
            "time_score": [8.0, 4.0, 2.0],
            "cart_score": [4.0, 2.0, 1.0],
            "order_score": [2.0, 1.0, 0.5],
        }
    ).write_parquet(root / "parts/part-000.parquet")
    files = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*.parquet")}
    (root / "manifest.json").write_text(json.dumps({"files": files, "history_end": 100}))
    return FeatureEngine(root)


def test_graph_aggregates_and_repeat_features_have_independent_expected_values(tmp_path):
    fitted = engine(tmp_path)
    prefix = Prefix(42, np.array([1, 2, 1]), np.array([100, 200, 300]), np.array([0, 1, 0]))
    candidates = fitted.candidates(prefix)
    names = (
        "graph_time_all_n3_uniform_sum",
        "graph_time_all_n3_uniform_mean",
        "graph_time_all_n3_uniform_std",
        "graph_time_carts_n3_uniform_sum",
        "repeat_all_n3_count",
        "repeat_all_n3_last_position",
        "repeat_all_n3_span",
    )
    matrix = fitted.transform(prefix, candidates, names)
    idx = np.flatnonzero(candidates.aid == 3)[0]
    assert matrix[idx, :4] == pytest.approx([10, 10 / 3, np.std([4, 2, 4]), 2])
    idx = np.flatnonzero(candidates.aid == 1)[0]
    assert matrix[idx, 4:] == pytest.approx([2, 1, 2])
    full = fitted.transform(prefix, candidates)
    assert full.dtype == np.float32 and np.isfinite(full).all()
    assert full.shape[1] == 1482
    assert np.array_equal(matrix, full[:, [fitted.names.index(n) for n in names]])
    assert len({f.name for f in feature_catalog()}) == 1482


def test_nested_candidate_budgets_cold_items_and_invalid_prefixes(tmp_path):
    fitted = engine(tmp_path)
    prefix = Prefix(1, np.array([1, 999]), np.array([100, 110]), np.array([0, 1]))
    all_candidates = fitted.candidates(prefix, 4)
    assert np.array_equal(fitted.candidates(prefix, 2).aid, all_candidates.aid[:2])
    assert all_candidates.aid[0] == 999
    assert len(set(all_candidates.aid)) == all_candidates.aid.size
    x = fitted.transform(prefix, all_candidates, ("history_known", "hist_all_all"))
    assert np.array_equal(x[0], [0, 0])
    assert np.array_equal(fitted.candidates(prefix).aid, fitted.candidates(prefix).aid)
    with pytest.raises(ValueError, match="precedes"):
        fitted.candidates(Prefix(1, np.array([1]), np.array([50]), np.array([0])))
    with pytest.raises(ValueError, match="ordering"):
        fitted.candidates(Prefix(1, np.array([1, 2]), np.array([120, 110]), np.array([0, 1])))
    with pytest.raises(ValueError, match="schema"):
        fitted.transform(prefix, all_candidates, ("target",))
