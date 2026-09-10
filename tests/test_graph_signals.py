"""Affinities preserve candidate identity and reject future or corrupt graph inputs."""

import json
import logging
import shutil

import numpy as np
import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.features import Prefix
from otto_recsys.research.graph_signals import GraphSignals, feature_names


def graph_fixture(root):
    (root / "parts").mkdir(parents=True)
    path = root / "parts/part-000.parquet"
    pl.DataFrame(
        {
            "source_aid": [1, 2],
            "target_aid": [3, 3],
            "time_score": [2.0, 5.0],
            "cart_score": [6.0, 15.0],
            "order_score": [8.0, 20.0],
        }
    ).write_parquet(path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "history_end": 100,
                "observed_history_max_ts": 99,
                "query_labels_used": False,
                "files": {"parts/part-000.parquet": sha256_file(path)},
            }
        )
    )
    return GraphSignals(root, "forward")


def test_affinities_equal_hand_calculation_and_ignore_other_candidates(tmp_path):
    graph = graph_fixture(tmp_path)
    prefix = Prefix(1, np.array([1, 2, 1]), np.array([101, 102, 103]), np.array([0, 1, 2]))
    aids = np.array([3, 999])
    values = graph.transform(prefix, aids)
    names = feature_names("forward")
    assert values.shape == (2, 72)
    assert values[0, names.index("wide_forward_time_all_n5_sum")] == 9
    assert values[0, names.index("wide_forward_time_all_n5_max")] == 5
    assert values[0, names.index("wide_forward_cart_carts_n5_sum")] == 15
    assert values[0, names.index("wide_forward_order_orders_n1_sum")] == 8
    assert not values[1].any()
    np.testing.assert_array_equal(graph.transform(prefix, np.array([3])), values[:1])
    with pytest.raises(ValueError, match="cutoff"):
        graph.transform(Prefix(1, np.array([1]), np.array([99]), np.array([0])), aids)
    (tmp_path / "parts/part-000.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        GraphSignals(tmp_path, "forward")


def test_complete_feature_only_study_preserves_every_baseline_row(tmp_path, monkeypatch):
    from test_directed_retrieval import test_complete_retrieval_study_preserves_roles_and_recovery

    from otto_recsys.research.graph_feature_study import run_study

    test_complete_retrieval_study_preserves_roles_and_recovery(tmp_path, monkeypatch)
    inputs = tmp_path / "inputs"
    for family in ("symmetric", "forward"):
        shutil.copytree(tmp_path / f"study/wide_{family}/retrieval", inputs / f"graphs/{family}")
    previous = json.loads((tmp_path / "study/results.json").read_text())
    names = ["source_time_score", "repeat_all_n3_count", "hist_clicks_all"]
    config = {
        "baseline_features": names,
        "screening_rows": 1000,
        "training": {"rounds": 2, "patience": 1, "threads": 1, "seed": 2},
        "expected_baseline_score": previous["arms"]["baseline"]["weighted_recall_at_20"],
    }
    output = tmp_path / "graph_features"
    args = dict(logger=logging.getLogger("graph-features-test"), publish=lambda path: None)
    result = run_study(inputs, output, config, **args)
    assert result["status"] == "passed" and result["evaluation_access"] is False
    assert set(result["arms"]) == {"baseline", "symmetric", "forward", "both"}
    for role in ("fit", "selection"):
        old = pl.read_parquet(inputs / f"{role}_cache/part-0000.parquet")
        new = pl.read_parquet(output / f"{role}_cache/part-0000.parquet")
        assert new.select(old.columns).equals(old)
        assert new.width == old.width + 144
    repeated = run_study(inputs, output, config, **args)
    assert repeated == result
