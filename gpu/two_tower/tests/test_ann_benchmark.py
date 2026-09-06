from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from evaluate import export_predictions
from otto_two_tower.ann_benchmark import _check_inputs
from otto_two_tower.ann_cli import hyperparameters_to_argv, parse_args
from otto_two_tower.benchmark_artifacts import BenchmarkArtifacts
from otto_two_tower.checkpoint import write_json_atomic
from otto_two_tower.config import ModelConfig
from otto_two_tower.data import ItemVocabulary, PackedSessionStore
from otto_two_tower.evaluation import identity, sha256_file
from otto_two_tower.logging_utils import configure_logging
from otto_two_tower.model import TwoTowerModel
from otto_two_tower.ranking_metrics import summarize_ranking


def fixture(tmp_path: Path) -> argparse.Namespace:
    ranking, items, model_dir, reference = [
        tmp_path / x for x in ("ranking", "items", "trained", "reference")
    ]
    for path in (ranking, items, model_dir, reference):
        path.mkdir()
    rng = np.random.default_rng(27)
    vectors = rng.normal(size=(160, 8)).astype(np.float32)
    item_ids = (rng.permutation(160) * 3 + 5).astype(np.int32)
    inverse = np.full(int(item_ids.max()) + 1, -1, dtype=np.int32)
    inverse[item_ids] = np.arange(160, dtype=np.int32)
    for name, value in {
        "item_ids": item_ids,
        "item_vectors": vectors,
        "aid_to_index": inverse,
    }.items():
        np.save(items / (name + ".npy"), value)
    item_manifest = {
        "items": 160,
        **{
            name + "_sha256": sha256_file(items / (name + ".npy"))
            for name in ("item_ids", "item_vectors", "aid_to_index")
        },
    }
    write_json_atomic(item_manifest, items / "manifest.json")
    sessions = np.arange(100, 124)
    folds = sessions % 2
    pq.write_table(pa.table({"session": sessions, "fold": folds}), ranking / "examples.parquet")
    pq.write_table(
        pa.table(
            {
                "session": sessions,
                "fold": folds,
                "aid": item_ids[sessions % 160],
                "ts": np.full(24, 1000),
                "event_type": np.zeros(24, dtype=int),
                "event_index": np.zeros(24, dtype=int),
            }
        ),
        ranking / "events.parquet",
    )
    labels = [
        {"session": int(s), "fold": int(s % 2), "objective": o, "aid": int(item_ids[(s + i) % 160])}
        for s in sessions
        for i, o in enumerate(("clicks", "carts", "orders"))
    ]
    pq.write_table(pa.Table.from_pylist(labels), ranking / "labels.parquet")
    rank_manifest = {
        "validation_manifest_id": "validation",
        "config": {"buckets": 3},
        "fold_session_counts": [12, 12],
        **{
            name + "_sha256": sha256_file(ranking / (name + ".parquet"))
            for name in ("events", "examples", "labels")
        },
    }
    write_json_atomic(rank_manifest, ranking / "manifest.json")
    torch.manual_seed(17)
    config = ModelConfig(embedding_dim=8, hidden_dim=16, time_buckets=4)
    model = TwoTowerModel(torch.tensor(vectors), padding_index=161, config=config, max_seq_len=2)
    torch.save(model.state_dict(), model_dir / "best_model.pt")
    write_json_atomic(
        {
            "input_id": "trained",
            "validation_manifest_id": "validation",
            "validation_fold": 0,
            "config": {"model": asdict(config), "data": {"max_seq_len": 2, "validation_fold": 0}},
        },
        model_dir / "training_manifest.json",
    )
    args = argparse.Namespace(
        ranking_cache=ranking,
        item_data=items,
        model_dir=model_dir,
        output_dir=reference,
        k=40,
        batch_size=4,
        chunk_size=64,
        allow_cpu=True,
        expected_ranking_id=identity(rank_manifest),
        expected_items_id=identity(item_manifest),
        training_input_id="trained",
        code_commit="reference-code",
    )
    manifest = export_predictions(args, {})
    return parse_args(
        [
            "--ranking-cache",
            str(ranking),
            "--item-data",
            str(items),
            "--model-dir",
            str(model_dir),
            "--reference-dir",
            str(reference),
            "--output-dir",
            str(tmp_path / "benchmark"),
            "--run-id",
            "run",
            "--code-commit",
            "code",
            "--reference-input-id",
            manifest["input_id"],
            "--reference-manifest-sha256",
            sha256_file(reference / "prediction_manifest.json"),
            "--sample-sessions",
            "8",
            "--nlist",
            "4",
            "--train-items",
            "160",
            "--train-iterations",
            "2",
            "--probes",
            "2,4",
            "--candidate-depth",
            "40",
            "--target-overlap",
            "0.98",
            "--batch-size",
            "2",
            "--index-shard-rows",
            "64",
            "--threads",
            "1",
            "--latency-queries",
            "2",
            "--latency-repeats",
            "2",
            "--warmup-queries",
            "1",
            "--heartbeat-seconds",
            "0.01",
            "--allow-cpu",
        ]
    )


def test_real_model_index_metrics_and_process_restart(
    tmp_path: Path, toolkit_argv: Callable[[dict[str, str]], list[str]]
) -> None:
    args = fixture(tmp_path)
    ids = np.load(args.item_data / "item_ids.npy")
    assert np.any(np.diff(ids) < 0) and np.min(np.diff(np.sort(ids))) > 1
    parameters = {
        key.replace("_", "-"): str(value)
        for key, value in vars(args).items()
        if key not in {"allow_cpu", "probes", "checkpoint_uri"}
    }
    parameters["probes"] = ",".join(str(p) for p in args.probes)
    argv = toolkit_argv(parameters)
    assert argv[argv.index("--export-fold-predictions") + 1] == "True"
    assert hyperparameters_to_argv({**parameters, "sagemaker_program": "benchmark.py"}) == argv
    command = [sys.executable, "benchmark.py", *argv, "--allow-cpu"]

    def run() -> subprocess.CompletedProcess:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed

    run()
    report = json.loads((args.output_dir / "metrics.json").read_text())
    assert report["status"] == "passed"
    assert report["full_reference_ranking"]["sessions"] == 12
    assert report["selected_nprobe"] in (2, 4)
    assert report["confirmation"]["ranking"]["sessions"] == 4
    assert report["confirmation_fidelity_passed"] is True
    assert report["full_ann_ranking"]["sessions"] == 12
    prediction_root = args.output_dir / "prediction_export"
    manifest = json.loads((prediction_root / "prediction_manifest.json").read_text())
    assert manifest["input_id"] == "run"
    assert manifest["search"]["k"] == 40
    assert manifest["training_input_id"] == "trained"
    assert len(manifest["parts"]) == 9
    totals = []
    for objective in ("clicks", "carts", "orders"):
        counts = []
        for path in sorted((prediction_root / "counts" / objective).glob("*.npz")):
            with np.load(path, allow_pickle=False) as part:
                counts.append(part["counts"])
        totals.append(np.concatenate(counts))
    assert summarize_ranking(np.stack(totals, axis=1)) == report["full_ann_ranking"]
    for part in manifest["parts"]:
        assert sha256_file(prediction_root / part["path"]) == part["sha256"]
    assert "base-exclusive" in report["base_union_status"]
    retained = args.output_dir / "indices/clicks/shards/part-00000000.faiss"
    timestamp = retained.stat().st_mtime_ns
    damaged = next((args.output_dir / "tuning").rglob("part-*.npz"))
    with np.load(damaged, allow_pickle=False) as part:
        original_ids = part["aids"]
    damaged.write_bytes(b"interrupted")
    exported = prediction_root / manifest["parts"][0]["path"]
    export_ids = pq.read_table(exported)["aids"].to_pylist()
    exported.write_bytes(b"interrupted full-fold export")
    for path in (args.output_dir / "metrics.json", args.output_dir / "metrics.json.json"):
        path.unlink()
    run()
    assert retained.stat().st_mtime_ns == timestamp
    with np.load(damaged, allow_pickle=False) as part:
        assert np.array_equal(part["aids"], original_ids)
    assert pq.read_table(exported)["aids"].to_pylist() == export_ids
    rows = [
        json.loads(line)
        for line in (args.output_dir / "logs/two_tower_ann.jsonl").read_text().splitlines()
    ]
    assert any(row["message"] == "heartbeat" for row in rows)
    assert any(row["message"] == "artifact_reused" for row in rows)
    assert all(row["timestamp"].endswith("+00:00") for row in rows)
    assert rows[-1]["message"] == "ann_attempt_complete"
    assert rows[-1]["elapsed_seconds"] > 0
    assert rows[-1]["status"] == "passed"


def test_reference_checksum_is_validated_before_benchmark(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    path = args.reference_dir / "prediction_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="reference manifest checksum mismatch"):
        _check_inputs(args, {})


def test_selected_session_loading_matches_full_store(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    vocabulary = ItemVocabulary.load(args.item_data)
    selected = np.array([100, 104])
    options = {"max_seq_len": 2, "time_buckets": 4}
    full = PackedSessionStore.from_parquet(args.ranking_cache, vocabulary, **options)
    subset = PackedSessionStore.from_parquet(
        args.ranking_cache, vocabulary, selected_sessions=selected, **options
    )
    assert np.array_equal(subset.session_ids, selected)
    assert torch.equal(
        full.batch(selected, torch.device("cpu")).item_indices,
        subset.batch(selected, torch.device("cpu")).item_indices,
    )


def test_real_model_resumes_all_stages_through_boto3_http_transport(
    tmp_path: Path, s3_http: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faiss

    from otto_two_tower import ann_benchmark

    args = fixture(tmp_path)
    logger = configure_logging("two_tower_ann", args.output_dir / "logs")
    local = BenchmarkArtifacts(args.output_dir, args.run_id, logger)
    expected = ann_benchmark.run_benchmark(args, local, logger, {})
    for path in args.output_dir.rglob("*"):
        if path.is_file():
            name = path.relative_to(args.output_dir).as_posix()
            if name not in {"metrics.json", "metrics.json.json"}:
                s3_http.objects["benchmark/run/" + name] = path.read_bytes()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("completed reference counts, query embeddings or centroids were recomputed")

    monkeypatch.setattr(ann_benchmark, "ranking_counts", forbidden)
    monkeypatch.setattr(ann_benchmark, "load_encoder", forbidden)
    monkeypatch.setattr(faiss.IndexIVFFlat, "train", forbidden)
    args.output_dir = tmp_path / "fresh-worker"
    logger = configure_logging("two_tower_ann", args.output_dir / "logs")
    restored = BenchmarkArtifacts(
        args.output_dir,
        args.run_id,
        logger,
        uri="s3://test-bucket/benchmark/run",
        client=s3_http.client,
    )
    actual = ann_benchmark.run_benchmark(args, restored, logger, {})
    assert actual["full_ann_ranking"] == expected["full_ann_ranking"]
    assert actual["selected_nprobe"] == expected["selected_nprobe"]
    assert actual["confirmation_fidelity_passed"] is True
    for prefix in (
        "reference/",
        "queries/",
        "indices/",
        "tuning/",
        "confirmation/",
        "prediction_export/",
    ):
        assert any(key.startswith("benchmark/run/" + prefix) for key in s3_http.reads)
    assert [key for key in s3_http.writes if "/logs/" not in key] == [
        "benchmark/run/metrics.json",
        "benchmark/run/metrics.json.json",
    ]
    rows = [
        json.loads(line)
        for line in (args.output_dir / "logs/two_tower_ann.jsonl").read_text().splitlines()
    ]
    completed = [row for row in rows if row["message"] == "artifact_restored"]
    assert len(completed) > 30
    assert all(
        row["elapsed_seconds"] >= 0 and row["timestamp"].endswith("+00:00") for row in completed
    )
