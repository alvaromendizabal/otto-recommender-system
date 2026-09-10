"""Frozen schemas, bounded advancement and actual cross-stage model recovery."""

import json
import logging

import numpy as np
import polars as pl
import pytest
from test_task_feature_pilot import fixture

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.dataset import OBJECTIVES
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.shared_feature_validation import advancement, frozen_arms, run


def shared_fixture(root):
    config = fixture(root)
    for role in ("fit", "selection"):
        path = root / f"{role}_cache/manifest.json"
        manifest = json.loads(path.read_text())
        for part in manifest["parts"]:
            file = path.parent / part
            frame = pl.read_parquet(file)
            for j, objective in enumerate(OBJECTIVES):
                frame = frame.with_columns(
                    (pl.col("aid") == 35 + j).cast(pl.Int8).alias(f"target_{objective}"),
                    (pl.col("aid") >= 30 + j)
                    .cast(pl.Float64)
                    .alias(["domain_click", "domain_cart", "domain_order"][j]),
                )
            frame.write_parquet(file)
            manifest["files"][part] = sha256_file(file)
        atomic_json(path, manifest)
        config["source_manifests"][role] = sha256_file(path)
    path = root / "corpus/queries.parquet"
    pl.read_parquet(path).with_columns(
        pl.col("session").cast(pl.Int64).alias("query_ts"),
        (pl.col("session") % 8 + 1).cast(pl.Int64).alias("observed_events"),
    ).write_parquet(path)
    config["queries_sha256"] = sha256_file(path)
    manifest = json.loads((root / "corpus/manifest.json").read_text())
    manifest["files"]["queries.parquet"] = config["queries_sha256"]
    atomic_json(root / "corpus/manifest.json", manifest)
    config["corpus_manifest_sha256"] = sha256_file(root / "corpus/manifest.json")
    config["added_features"] = ["domain_click", "domain_cart", "domain_order"]
    config["ablation_blocks"] = {
        name: [column]
        for name, column in zip(
            ("click_signal", "cart_signal", "order_signal"), config["added_features"], strict=True
        )
    }
    config["minimum_weighted_gain"] = 0.001
    config["source_study_id"] = "synthetic_prior"
    atomic_json(
        root / "frozen_screening.json",
        {
            "study_id": config["source_study_id"],
            "shared": [*config["baseline_features"], *config["added_features"]],
        },
    )
    config["source_screening_sha256"] = sha256_file(root / "frozen_screening.json")
    return config


def test_frozen_schema_and_blocks_reject_silent_selection_changes(tmp_path):
    config = shared_fixture(tmp_path)
    arms = frozen_arms(tmp_path, config)
    assert arms["without_click_signal"] == ["base", "domain_cart", "domain_order"]
    config["ablation_blocks"]["cart_signal"].append("domain_click")
    with pytest.raises(ValueError, match="schema"):
        frozen_arms(tmp_path, config)
    (tmp_path / "frozen_screening.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        frozen_arms(tmp_path, config)


def test_gate_requires_both_weighted_gain_and_preserved_orders():
    base = {"weighted_recall_at_20": 0.5, "objectives": {"orders": {"recall_at_20": 0.6}}}
    shared = {"weighted_recall_at_20": 0.51, "objectives": {"orders": {"recall_at_20": 0.6}}}
    arms = {"baseline": base, "shared": shared}
    assert advancement(arms, 0.001)["advance_to_ablations"]
    shared["objectives"]["orders"]["recall_at_20"] = 0.59
    assert not advancement(arms, 0.001)["advance_to_ablations"]
    shared["objectives"]["orders"]["recall_at_20"] = 0.6
    shared["weighted_recall_at_20"] = 0.5005
    assert not advancement(arms, 0.001)["advance_to_ablations"]
    with pytest.raises(ValueError, match="nonnegative"):
        advancement(arms, -1)
    shared["weighted_recall_at_20"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        advancement(arms, 0.001)


def test_two_stage_recovery_reuses_controls_and_preserves_complete_targets(tmp_path, monkeypatch):
    import lightgbm as lgb

    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    config = shared_fixture(inputs)
    logger = logging.getLogger("shared_test")
    with pytest.raises(ValueError, match="completed validation"):
        run(inputs, output, config, logger=logger, phase="ablation")
    result = run(inputs, output, config, logger=logger)
    assert list(result["arms"]) == ["baseline", "shared"]
    assert result["decision"]["advance_to_ablations"]
    assert result["candidate_ceiling"]["objectives"]["orders"]["recall_at_20"] == 0.5
    baseline = result["arms"]["baseline"]
    assert sum(baseline["slices"][f"time_q{i}"]["sessions"] for i in range(1, 5)) == 32
    for objective in OBJECTIVES:
        assert (
            sum(
                baseline["slices"][f"time_q{i}"]["objectives"][objective]["hits"]
                for i in range(1, 5)
            )
            == baseline["objectives"][objective]["hits"]
        )
    train, calls = lgb.train, []

    def counted(*args, **kwargs):
        calls.append(1)
        return train(*args, **kwargs)

    monkeypatch.setattr(lgb, "train", counted)
    ablated = run(inputs, output, config, logger=logger, phase="ablation")
    assert len(calls) == 9  # Three synthetic block removals, three tasks; controls reused.
    assert ablated["arms"]["baseline"] == result["arms"]["baseline"]
    assert ablated["arms"]["shared"] == result["arms"]["shared"]

    def forbidden(*args, **kwargs):
        raise AssertionError("completed checkpoints must not retrain")

    monkeypatch.setattr(lgb, "train", forbidden)
    assert run(inputs, output, config, logger=logger, phase="ablation")["arms"] == ablated["arms"]
    assert run(inputs, output, config, logger=logger)["arms"] == result["arms"]
    with pytest.raises(ValueError, match="different experiment"):
        run(inputs, output, {**config, "seed": 10}, logger=logger)
    previous = json.loads((output / "validation_results.json").read_text())
    previous["arms"]["shared"]["weighted_recall_at_20"] = 0
    atomic_json(output / "validation_results.json", previous)
    with pytest.raises(ValueError, match="passed validation gate"):
        run(inputs, output, config, logger=logger, phase="ablation")
