"""Hand calculations, millisecond boundaries, invariant candidates and recovery."""

import json
import logging
import shutil

import numpy as np
import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.domain_features import (
    DOMAIN_FAMILIES,
    NormalizedGraphSignals,
    episode_features,
    feature_names,
    funnel_features,
)
from otto_recsys.research.features import Prefix
from otto_recsys.research.graph_signals import GraphSignals


def test_funnel_tracks_observed_action_order_and_millisecond_gaps():
    prefix = Prefix(
        1,
        np.array([1, 2, 1, 1, 1, 2]),
        np.array([0, 1, 3_600_000, 7_200_000, 10_800_000, 14_400_000]),
        np.array([0, 0, 1, 2, 0, 1]),
    )
    aids = np.array([1, 2, 999])
    values = funnel_features(prefix, aids)
    names = feature_names("funnel")

    def value(row, name):
        return values[row, names.index(f"domain_funnel_{name}")]

    assert value(0, "full_count_share") == pytest.approx(4 / 6)
    assert value(0, "cart_without_later_order") == 0
    assert value(1, "cart_without_later_order") == 1
    assert value(0, "click_after_cart") == value(0, "click_after_order") == 1
    assert value(0, "previous_action_2") == value(0, "last_action_0") == 1
    assert value(0, "last_age_log_hours") == pytest.approx(np.log(2))
    assert value(0, "last_unique_reciprocal_rank") == 0.5
    assert value(1, "last_unique_reciprocal_rank") == 1
    for left, right in ((0, 1), (1, 2), (2, 0)):
        assert value(0, f"transition_{left}_{right}_count_log") == pytest.approx(np.log(2))
        assert value(0, f"transition_{left}_{right}_mean_gap_log_hours") == pytest.approx(np.log(2))
    assert value(2, "seen") == 0
    np.testing.assert_array_equal(funnel_features(prefix, aids[[2, 0]]), values[[2, 0]])


def test_episode_gap_is_strict_and_uses_milliseconds():
    prefix = Prefix(
        1, np.array([1, 2, 3]), np.array([0, 1_800_000, 3_600_001]), np.array([1, 0, 2])
    )
    values = episode_features(prefix, np.array([1, 2, 3, 999]))
    names = feature_names("episode")

    def column(gap, name):
        return values[:, names.index(f"domain_episode_s{gap}_{name}")]

    np.testing.assert_array_equal(column(1800, "seen"), [0, 0, 1, 0])
    np.testing.assert_array_equal(column(1800, "previous_episode_seen"), [1, 1, 0, 0])
    np.testing.assert_array_equal(column(7200, "seen"), [1, 1, 1, 0])
    assert column(1800, "query_episode_count_log")[0] == pytest.approx(np.log(3))
    assert column(7200, "query_episode_count_log")[0] == pytest.approx(np.log(2))
    assert column(1800, "query_boundary_gap_log_hours")[0] == pytest.approx(
        np.log1p(1_800_001 / 3_600_000)
    )


def test_graph_normalizations_share_pools_and_ignore_candidate_composition(tmp_path):
    from test_graph_signals import graph_fixture

    graphs = {}
    for family in ("symmetric", "forward"):
        root = tmp_path / family
        graph_fixture(root)
        graphs[family] = GraphSignals(root, family)
    signals = NormalizedGraphSignals(graphs)
    prefix = Prefix(1, np.array([1, 2, 1]), np.array([101, 102, 103]), np.array([0, 1, 2]))
    aids = np.array([3, 999])
    values = signals.transform(prefix, aids)
    raw, norm = values["raw_graph"], values["normalized_graph"]

    def value(mode, pool, reduction):
        family = "raw_graph" if mode == "raw" else "normalized_graph"
        name = f"domain_norm_forward_time_{mode}_{pool}_{reduction}"
        return values[family][0, feature_names(family).index(name)]

    assert value("raw", "recent_unique", "mean") == 3.5  # not event-weighted 3
    assert value("row", "last", "mean") == pytest.approx(2 / 3)
    assert value("row", "recent_unique", "mean") == pytest.approx((2 / 3 + 5 / 6) / 2)
    assert value("degree", "last", "max") == pytest.approx(2 / np.sqrt(3 * 8))
    assert value("raw", "purchase_unique", "mean") == 3.5
    assert raw.shape == (2, 36) and norm.shape == (2, 72)
    for family in values:
        assert not values[family][1].any()
        np.testing.assert_array_equal(
            signals.transform(prefix, np.array([3]))[family], values[family][:1]
        )
    clicks = Prefix(1, np.array([999]), np.array([101]), np.array([0]))
    assert all(not block.any() for block in signals.transform(clicks, aids).values())
    with pytest.raises(ValueError, match="cutoff"):
        signals.transform(Prefix(1, np.array([1]), np.array([99]), np.array([0])), aids)
    with pytest.raises(ValueError, match="unique"):
        signals.transform(prefix, np.array([3, 3]))
    assert [len(feature_names(f)) for f in DOMAIN_FAMILIES] == [40, 24, 36, 72]


def test_complete_domain_comparison_replays_baseline_and_recovers(tmp_path, monkeypatch):
    from test_directed_retrieval import test_complete_retrieval_study_preserves_roles_and_recovery

    from otto_recsys.research import domain_feature_study as study
    from otto_recsys.research.dataset import Queries

    test_complete_retrieval_study_preserves_roles_and_recovery(tmp_path, monkeypatch)
    inputs, previous, output = tmp_path / "inputs", tmp_path / "study", tmp_path / "domain"
    for family in ("symmetric", "forward"):
        shutil.copytree(previous / f"wide_{family}/retrieval", inputs / f"graphs/{family}")
    shutil.copytree(previous / "models/baseline", inputs / "reference")
    config = {
        "baseline_features": ["source_time_score", "repeat_all_n3_count", "hist_clicks_all"],
        "screening_rows": 1000,
        "training": {"rounds": 2, "patience": 1, "threads": 1, "seed": 2},
        "expected_baseline_score": json.loads((previous / "results.json").read_text())["arms"][
            "baseline"
        ]["weighted_recall_at_20"],
        "arms": {k: list(v) for k, v in study.ARM_FAMILIES.items()},
        "reference_files": {
            n: sha256_file(inputs / "reference" / n) for n in study.REFERENCE_FILES
        },
    }

    def development_queries(corpus, role):
        assert role in {"fit", "selection"}, "evaluation access forbidden"
        return Queries(corpus, role)

    monkeypatch.setattr(study, "Queries", development_queries)
    args = {"logger": logging.getLogger("domain-test"), "publish": lambda path: None}
    result = study.run_study(inputs, output, config, **args)
    assert result["status"] == "passed" and result["evaluation_access"] is False
    assert set(result["arms"]) == {"baseline", *study.ARM_FAMILIES}
    assert result["arms"]["baseline"]["reused_reference"] is True
    for role in ("fit", "selection"):
        old = pl.read_parquet(inputs / f"{role}_cache/part-0000.parquet")
        new = pl.read_parquet(output / f"{role}_cache/part-0000.parquet")
        assert new.select(old.columns).equals(old)
        assert new.width == old.width + 172
    from otto_recsys.research.domain_feature_audit import audit

    launch = {
        "task": "domain_features",
        "study": config,
        "source_commit": "test-source",
        "graphs": {
            f: [
                {
                    "path": "manifest.json",
                    "sha256": sha256_file(inputs / f"graphs/{f}/manifest.json"),
                }
            ]
            for f in ("symmetric", "forward")
        },
        "corpus": [
            {
                "key": "corpus/queries.parquet",
                "sha256": sha256_file(inputs / "corpus/queries.parquet"),
            }
        ],
    }
    (output / "launch.json").write_text(json.dumps(launch))
    shutil.copyfile(inputs / "corpus/queries.parquet", output / "queries.parquet")
    checked = audit(output)
    assert checked["status"] == "passed" and checked["arms_verified"] == 5
    assert checked["models_verified"] == 15
    altered = json.loads((output / "results.json").read_text())
    altered["arms"]["sequence"]["weighted_recall_at_20"] += 0.01
    (output / "results.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="weighted Recall"):
        audit(output)
    (output / "results.json").write_text(json.dumps(result))
    times = {p: p.stat().st_mtime_ns for p in output.glob("*_cache/part-*.parquet")}
    assert study.run_study(inputs, output, config, **args) == result
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in times.items())
    with pytest.raises(ValueError, match="contract differs"):
        study.run_study(inputs, output, {**config, "screening_rows": 22}, **args)
    (inputs / "reference/clicks/model.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="reference checksum"):
        study.run_study(inputs, output, config, **args)
