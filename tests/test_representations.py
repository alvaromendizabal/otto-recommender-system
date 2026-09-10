"""Temporal isolation, semantic similarities, complete-query alignment and native recovery."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
import pytest

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.features import Prefix
from otto_recsys.research.representation_study import augment_cache, screen_features
from otto_recsys.research.representations import (
    Representation,
    feature_names,
    prepare_sequences,
    train_representation,
)

LOGGER = logging.getLogger("representation-test")


def write_vectors(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "vectors.npz"
    np.savez(
        path,
        ids=np.array([1, 2, 3]),
        vectors=np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32),
        counts=np.array([10, 20, 30]),
    )
    (directory / "manifest.json").write_text(json.dumps({"sha256": sha256_file(path)}))
    return Representation(path)


def test_similarity_features_match_geometry_and_are_independent_of_candidate_sampling(tmp_path):
    vectors = write_vectors(tmp_path)
    prefix = Prefix(9, np.array([1, 2]), np.array([100, 200]), np.array([0, 1]))
    candidates = np.array([1, 2, 3, 999])
    actual = vectors.transform(prefix, candidates)
    names = feature_names("all")
    assert actual.shape == (4, len(names))
    assert len(set(names)) == 74
    np.testing.assert_array_equal(actual[:, names.index("embedding_all_all_n5_last")], [0, 1, 0, 0])
    np.testing.assert_allclose(
        actual[:, names.index("embedding_all_all_n5_centroid")], [2**-0.5, 2**-0.5, -(2**-0.5), 0]
    )
    assert actual[3, 0] == 0 and actual[3, 1] == 0
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(vectors.transform(prefix, candidates[[1, 3]]), actual[[1, 3]])
    # Missing action histories carry explicit zero coverage and finite zero scores.
    assert np.all(actual[:, names.index("embedding_all_orders_n50_coverage")] == 0)
    unknown = Prefix(10, np.array([777]), np.array([200]), np.array([0]))
    assert not vectors.transform(unknown, candidates)[:, 2:].any()
    (tmp_path / "vectors.npz").write_bytes(b"partial checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        Representation(tmp_path / "vectors.npz")


def test_sequences_sort_events_separate_intent_and_reject_future_history(tmp_path):
    history = tmp_path / "history.parquet"
    pl.DataFrame(
        {
            "session": [2, 1, 1, 2],
            "aid": [30, 20, 10, 40],
            "ts": [200, 200, 100, 300],
            "event_index": [0, 1, 0, 1],
            "event_type": [1, 1, 0, 2],
        }
    ).write_parquet(history)
    args = dict(
        history=history,
        output=tmp_path / "sequences",
        cutoff=400,
        expected_sha256=sha256_file(history),
        threads=1,
        logger=LOGGER,
    )
    receipt = prepare_sequences(**args)
    assert (tmp_path / "sequences/all.txt").read_text().splitlines() == ["10 20", "30 40"]
    assert (tmp_path / "sequences/intent.txt").read_text().splitlines() == ["20", "30 40"]
    assert prepare_sequences(**args) == receipt
    with pytest.raises(ValueError, match="strictly before"):
        prepare_sequences(**{**args, "output": tmp_path / "future", "cutoff": 300})
    with pytest.raises(ValueError, match="different history"):
        prepare_sequences(**{**args, "cutoff": 500})


def test_native_embedding_epoch_recovery_preserves_the_realized_vectors(tmp_path):
    corpus = tmp_path / "sequences.txt"
    corpus.write_text("1 2 3 1 2\n2 3 4 2 3\n" * 100)
    output = tmp_path / "model"
    config = {"dimensions": 8, "window": 3, "negative": 2, "epochs": 2, "workers": 1, "seed": 8}
    receipt = train_representation(corpus, output, config, logger=LOGGER)
    assert receipt["epochs"] == 2 and receipt["vocabulary"] == 4
    first = Representation(output / "vectors.npz").vectors.copy()
    assert train_representation(corpus, output, config, logger=LOGGER) == receipt
    # Recover a missing export from the last verified native epoch without retraining.
    (output / "vectors.npz").unlink()
    restored = train_representation(corpus, output, config, logger=LOGGER)
    np.testing.assert_array_equal(Representation(output / "vectors.npz").vectors, first)
    assert restored["sha256"] == receipt["sha256"]
    with pytest.raises(ValueError, match="contract differs"):
        train_representation(corpus, output, {**config, "seed": 9}, logger=LOGGER)


def test_quality_screen_uses_fit_only_and_rejects_duplicate_constant_and_nonfinite_columns():
    a = np.arange(200, dtype=np.float32)
    b = np.sin(a)
    matrix = np.column_stack([a, a * 2, b, np.ones(200), np.full(200, np.nan)])
    result = screen_features(
        matrix, ("base", "duplicate", "useful", "constant", "bad"), ("base",), rows=150
    )
    assert result["features"] == ["base", "useful"]
    assert [r["decision"] for r in result["decisions"]] == [
        "redundant_on_fit_sample",
        "retained",
        "constant_on_fit_sample",
        "nonfinite",
    ]


def test_augmentation_preserves_rows_labels_and_complete_groups_and_recovers(tmp_path):
    vectors = write_vectors(tmp_path / "vectors")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    frame = pl.DataFrame(
        {
            "base": [1.0, 2.0, 3.0, 4.0],
            "session": [7, 7, 8, 8],
            "aid": [1, 3, 2, 999],
            "candidate_position": [0, 1, 0, 1],
            **{f"target_{o}": [1, 0, 1, 0] for o in ("clicks", "carts", "orders")},
            **{f"baseline_{o}": [0.4, 0.2, 0.4, 0.2] for o in ("clicks", "carts", "orders")},
        }
    )
    part = source / "part-0000.parquet"
    frame.write_parquet(part)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "input_id": "fixture",
                "role": "selection",
                "rows": 4,
                "sessions": 2,
                "parts": [part.name],
                "files": {part.name: sha256_file(part)},
            }
        )
    )

    class QueriesFixture:
        def __init__(self):
            self.role = "selection"
            self.session = np.array([7, 8])
            self.manifest = {"input_id": "fixture"}

        def prefix(self, index):
            return Prefix(
                int(self.session[index]), np.array([1, 2]), np.array([100, 200]), np.array([0, 1])
            )

    args = dict(
        source=source,
        output=output,
        queries=QueriesFixture(),
        representations={"all": vectors, "intent": vectors},
        vector_hashes={"all": "fixture", "intent": "fixture"},
        base_names=("base",),
        logger=LOGGER,
    )
    receipt = augment_cache(**args)
    actual = pl.read_parquet(output / part.name)
    assert actual.select(frame.columns).equals(frame)
    assert actual.width == frame.width + 148 and receipt["rows"] == 4
    (output / part.name).write_bytes(b"interrupted")
    assert augment_cache(**args)["files"] == receipt["files"]
    args["queries"].role = "evaluation"
    with pytest.raises(ValueError, match=r"temporal role|evaluation queries"):
        augment_cache(**args)


def test_complete_learned_feature_study_trains_matched_models_without_evaluation(
    tmp_path, monkeypatch
):
    from otto_recsys.research import representation_study

    inputs, output = tmp_path / "inputs", tmp_path / "study"
    corpus = inputs / "corpus"
    corpus.mkdir(parents=True)
    history = corpus / "history.parquet"
    pl.DataFrame(
        {
            "session": np.repeat(np.arange(80), 40),
            "aid": np.tile(np.arange(1, 41), 80),
            "event_index": np.tile(np.arange(40), 80),
            "ts": np.tile(np.arange(40), 80),
            "event_type": np.tile(np.arange(40) % 3, 80),
        }
    ).write_parquet(history)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "input_id": "synthetic",
                "files": {"history.parquet": sha256_file(history)},
            }
        )
    )
    (corpus / "contract.json").write_text(json.dumps({"protocol": {"history_end": 100}}))

    class QueryFixture:
        def __init__(self, path, role):
            assert role in ("fit", "selection"), "evaluation access is forbidden"
            self.role = role
            self.session = np.arange(120) + (200 if role == "selection" else 0)
            self.denominators = np.ones((120, 3), dtype=np.int64)
            self.manifest = {"input_id": "synthetic"}

        def prefix(self, index):
            return Prefix(
                int(self.session[index]), np.array([1, 2]), np.array([101, 102]), np.array([0, 1])
            )

    monkeypatch.setattr(representation_study, "Queries", QueryFixture)
    rng = np.random.default_rng(4)
    for role in ("fit", "selection"):
        directory = inputs / f"{role}_cache"
        directory.mkdir()
        ids = QueryFixture(corpus, role).session
        labels = np.tile(np.r_[1, np.zeros(39)], 120).astype(np.int8)
        frame = pl.DataFrame(
            {
                "base": rng.random(4800).astype(np.float32),
                "session": np.repeat(ids, 40),
                "aid": np.tile(np.arange(1, 41), 120),
                "candidate_position": np.tile(np.arange(40), 120),
                **{f"target_{o}": labels for o in ("clicks", "carts", "orders")},
                **{
                    f"baseline_{o}": rng.random(4800).astype(np.float32)
                    for o in ("clicks", "carts", "orders")
                },
            }
        )
        part = directory / "part-0000.parquet"
        frame.write_parquet(part)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "role": role,
                    "input_id": role,
                    "rows": 4800,
                    "sessions": 120,
                    "features": ["base"],
                    "parts": [part.name],
                    "files": {part.name: sha256_file(part)},
                }
            )
        )
    config = {
        "baseline_features": ["base"],
        "screening_rows": 1000,
        "embedding": {
            "dimensions": 8,
            "window": 3,
            "negative": 2,
            "epochs": 1,
            "workers": 1,
            "seed": 8,
        },
        "training": {"rounds": 3, "patience": 2, "threads": 1, "seed": 4},
    }
    report = representation_study.run_study(inputs, output, config, logger=LOGGER)
    assert report["status"] == "passed"
    assert report["fit_sessions"] == report["selection_sessions"] == 120
    assert set(report["variants"]) == {"baseline", "with_all", "with_intent", "with_both"}
    assert report["evaluation_access"] is False and report["kaggle_promotion"] is False
    assert len(list((output / "models").glob("*/*/model.txt"))) == 12
    assert report["candidate_ceiling"]["weighted_recall_at_20"] == pytest.approx(1)
    repeated = representation_study.run_study(inputs, output, config, logger=LOGGER)
    assert repeated["variants"] == report["variants"]
