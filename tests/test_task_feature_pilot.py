"""Sampling, task-specific signals, metric denominators and model recovery."""

import json
import logging

import numpy as np
import polars as pl
import pytest

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.research.dataset import OBJECTIVES
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.task_feature_pilot import (
    load_sample,
    run,
    sample_parts,
    screen,
    select_columns,
)


def fixture(root):
    names = ["base", "domain_click", "domain_cart", "domain_order", "domain_noise"]
    rng = np.random.default_rng(10)
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    ledger, manifests = [], {}
    for role, offset in (("fit", 0), ("selection", 1000)):
        directory = root / f"{role}_cache"
        directory.mkdir()
        parts, hashes = [], {}
        for p in range(4):
            ids = np.arange(offset + p * 16, offset + (p + 1) * 16)
            candidate = np.tile(np.arange(40), 16)
            values = {
                "session": np.repeat(ids, 40),
                "aid": candidate,
                "candidate_position": candidate,
                "base": rng.normal(size=640),
                "domain_noise": rng.normal(size=640),
            }
            for j, objective in enumerate(OBJECTIVES):
                values[f"target_{objective}"] = (candidate == j).astype(np.int8)
                values[f"baseline_{objective}"] = -candidate.astype(float)
                values[names[j + 1]] = (candidate == j) + rng.normal(0, 0.1, 640)
                ledger.extend(
                    {
                        "session": int(s),
                        "split_role": role,
                        "objective": objective,
                        "recall_denominator": 2 if objective == "orders" else 1,
                    }
                    for s in ids
                )
            name = f"part-{p:04d}.parquet"
            pl.DataFrame(values).write_parquet(directory / name)
            parts.append(name)
            hashes[name] = sha256_file(directory / name)
        contract = {"corpus_id": "fixture", "role": role, "features": names}
        atomic_json(directory / "contract.json", contract)
        manifest = {
            "status": "passed",
            "role": role,
            "features": names,
            "parts": parts,
            "files": hashes,
            "input_id": canonical_json_sha256(contract),
        }
        atomic_json(directory / "manifest.json", manifest)
        manifests[role] = sha256_file(directory / "manifest.json")
    pl.DataFrame(ledger).write_parquet(corpus / "queries.parquet")
    atomic_json(
        corpus / "manifest.json",
        {
            "input_id": "fixture",
            "files": {"queries.parquet": sha256_file(corpus / "queries.parquet")},
        },
    )
    return {
        "source_manifests": manifests,
        "part_counts": {"fit": 2, "selection": 2},
        "sessions_per_part": 16,
        "candidate_budget": 40,
        "corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
        "queries_sha256": sha256_file(corpus / "queries.parquet"),
        "baseline_features": ["base"],
        "quality_rows": 1280,
        "seed": 9,
        "threads": 1,
        "screen_rounds": 3,
        "maximum_added": 1,
        "training": {"threads": 1, "rounds": 3, "patience": 3, "seed": 9},
    }


def test_sampling_spans_period_and_omits_partial_tail():
    assert sample_parts([str(i) for i in range(79)], 8) == [str(i) for i in range(0, 78, 11)]
    with pytest.raises(ValueError):
        sample_parts(["0", "1"], 3)


def test_task_signals_survive_shared_order_weighting_at_equal_budget():
    names = ("base", "click", "cart", "order")
    gains = np.zeros((3, 3, 4))
    gains[:, :, 0] = 1
    for j in range(3):
        gains[:, j, j + 1] = 10
    result = select_columns(names, ("base",), gains, 1)
    assert result["shared"] == ["base", "order"]
    assert result["per_task"] == {o: ["base", names[j + 1]] for j, o in enumerate(OBJECTIVES)}
    gains[1:, 0, 1] = 0
    assert select_columns(names, ("base",), gains, 1)["per_task"]["clicks"] == ["base"]


def test_missing_targets_corruption_and_role_guards(tmp_path):
    config = fixture(tmp_path)
    fit = load_sample(tmp_path, "fit", config)
    valid = load_sample(tmp_path, "selection", config)
    assert fit["ids"].size == valid["ids"].size == 32
    assert np.all(valid["groups"] == 40)
    assert np.all(valid["denominators"][:, 2] == 2)
    assert np.all(valid["coverage"][:, 2] == 1)
    with pytest.raises(ValueError, match="fitting data"):
        screen(valid, tmp_path / "bad", config, "x", logging.getLogger("test"))
    with pytest.raises(ValueError, match="roles"):
        load_sample(tmp_path, "evaluation", config)
    (tmp_path / "fit_cache/part-0000.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_sample(tmp_path, "fit", config)


def test_complete_pilot_and_training_free_recovery(tmp_path, monkeypatch):
    import lightgbm as lgb

    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    config = fixture(inputs)
    result = run(inputs, output, config, logger=logging.getLogger("test"))
    assert result["status"] == "passed"
    assert list(result["arms"]) == ["baseline", "shared", "per_task"]
    assert result["candidate_ceiling"]["objectives"]["orders"]["recall_at_20"] == 0.5
    assert len(json.loads((output / "screening/selection.json").read_text())["pilots"]) == 9

    def unexpected(*args, **kwargs):
        raise AssertionError("completed pilot must not train again")

    monkeypatch.setattr(lgb, "train", unexpected)
    resumed = run(inputs, output, config, logger=logging.getLogger("test"))
    assert resumed["arms"] == result["arms"]
    with pytest.raises(ValueError, match="different experiment"):
        run(inputs, output, {**config, "seed": 11}, logger=logging.getLogger("test"))
    (output / "screening/0-clicks.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="checkpoint"):
        run(inputs, output, config, logger=logging.getLogger("test"))
