from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from evaluate import export_predictions
from otto_two_tower.checkpoint import write_json_atomic
from otto_two_tower.config import ModelConfig
from otto_two_tower.data import writable_vectors
from otto_two_tower.evaluation import (
    commit_part,
    exact_search,
    identity,
    sha256_file,
    verified_part,
)
from otto_two_tower.model import TwoTowerModel


def test_exact_search_matches_exhaustive_ranking_with_boundary_ties() -> None:
    candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    ids = torch.tensor([40, 10, 20, 30, 50])
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    scores, found = exact_search(queries, candidates, ids, k=3, chunk_size=2)
    expected = np.array(
        [np.lexsort((ids.numpy(), -row))[:3] for row in (queries @ candidates.T).numpy()]
    )
    assert np.array_equal(found.numpy(), ids.numpy()[expected])
    assert np.allclose(
        scores.numpy(), np.take_along_axis((queries @ candidates.T).numpy(), expected, axis=1)
    )


def test_nonfinite_search_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        exact_search(
            torch.tensor([[float("nan"), 1.0]]),
            torch.ones(2, 2),
            torch.arange(2),
            k=1,
            chunk_size=1,
        )


def test_readonly_array_becomes_owned_writable_tensor() -> None:
    source = np.ones((3, 4), dtype=np.float32)
    source.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensor = writable_vectors(source)
    tensor[0, 0] = 7
    assert source[0, 0] == 1
    assert tensor[0, 0] == 7


def test_artifact_receipt_requires_complete_matching_bytes(tmp_path: Path) -> None:
    path = tmp_path / "part.npy"
    temporary = tmp_path / "part.npy.tmp"
    temporary.write_bytes(b"complete")
    assert verified_part(path, "run") is None
    commit_part(temporary, path, "run", rows=3)
    assert verified_part(path, "run")["rows"] == 3
    with pytest.raises(ValueError, match="different evaluation"):
        verified_part(path, "other")
    path.write_bytes(b"interrupted")
    assert verified_part(path, "run") is None


def fixture_inputs(tmp_path: Path) -> argparse.Namespace:
    ranking, items, trained, output = (
        tmp_path / name for name in ("ranking", "items", "trained", "output")
    )
    for path in (ranking, items, trained, output):
        path.mkdir()
    np.save(items / "item_ids.npy", np.arange(12, dtype=np.int32))
    np.save(items / "aid_to_index.npy", np.arange(12, dtype=np.int32))
    vectors = np.random.default_rng(7).normal(size=(12, 4)).astype(np.float32)
    np.save(items / "item_vectors.npy", vectors)
    item_manifest = {
        "items": 12,
        **{
            name + "_sha256": sha256_file(items / (name + ".npy"))
            for name in ("item_ids", "item_vectors", "aid_to_index")
        },
    }
    write_json_atomic(item_manifest, items / "manifest.json")
    pq.write_table(
        pa.table({"session": [10, 11, 12], "fold": [0, 1, 0]}), ranking / "examples.parquet"
    )
    pq.write_table(
        pa.table(
            {
                "session": [10, 11, 12],
                "aid": [1, 2, 3],
                "ts": [1000] * 3,
                "event_type": [0] * 3,
                "event_index": [0] * 3,
                "fold": [0, 1, 0],
            }
        ),
        ranking / "events.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "session": [10, 11, 12],
                "aid": [2, 3, 4],
                "objective": ["clicks"] * 3,
                "fold": [0, 1, 0],
            }
        ),
        ranking / "labels.parquet",
    )
    ranking_manifest = {
        "validation_manifest_id": "test-validation",
        "config": {"buckets": 2},
        "fold_session_counts": [2, 1],
        **{
            name + "_sha256": sha256_file(ranking / (name + ".parquet"))
            for name in ("events", "examples", "labels")
        },
    }
    write_json_atomic(ranking_manifest, ranking / "manifest.json")
    model_config = ModelConfig(embedding_dim=4, hidden_dim=8, time_buckets=4)
    model = TwoTowerModel(
        torch.tensor(vectors), padding_index=13, config=model_config, max_seq_len=2
    )
    torch.save(model.state_dict(), trained / "best_model.pt")
    from dataclasses import asdict

    manifest = {
        "input_id": "trained-input",
        "validation_manifest_id": "test-validation",
        "validation_fold": 0,
        "config": {"model": asdict(model_config), "data": {"max_seq_len": 2, "validation_fold": 0}},
    }
    write_json_atomic(manifest, trained / "training_manifest.json")
    return argparse.Namespace(
        ranking_cache=ranking,
        item_data=items,
        model_dir=trained,
        output_dir=output,
        k=5,
        batch_size=1,
        chunk_size=5,
        allow_cpu=True,
        expected_ranking_id=identity(ranking_manifest),
        expected_items_id=identity(item_manifest),
        training_input_id="trained-input",
        code_commit="test-commit",
    )


def test_export_is_held_out_complete_and_resumes_after_corruption(tmp_path: Path) -> None:
    args = fixture_inputs(tmp_path)
    first = export_predictions(args, {})
    assert first["sessions"] == 2
    assert len(first["parts"]) == 6
    path = args.output_dir / first["parts"][0]["path"]
    original = path.read_bytes()
    untouched = args.output_dir / first["parts"][1]["path"]
    untouched_time = untouched.stat().st_mtime_ns
    for part in first["parts"]:
        assert set(pq.read_table(args.output_dir / part["path"])["session"].to_pylist()) <= {10, 12}
    path.write_bytes(b"incomplete")
    second = export_predictions(args, {})
    assert second["input_id"] == first["input_id"]
    assert path.read_bytes() == original
    assert untouched.stat().st_mtime_ns == untouched_time
    args.code_commit = "changed"
    with pytest.raises(ValueError, match="different evaluation"):
        export_predictions(args, {})


def test_export_rejects_changed_input_bytes(tmp_path: Path) -> None:
    args = fixture_inputs(tmp_path)
    (args.ranking_cache / "labels.parquet").write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum"):
        export_predictions(args, {})


def test_export_rejects_training_identity_mismatch(tmp_path: Path) -> None:
    args = fixture_inputs(tmp_path)
    args.training_input_id = "wrong"
    with pytest.raises(ValueError, match="training input"):
        export_predictions(args, {})
