"""Export held-out full-catalogue predictions from a saved best model."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from otto_two_tower.checkpoint import write_json_atomic
from otto_two_tower.config import OBJECTIVES, ModelConfig
from otto_two_tower.data import ItemVocabulary, PackedSessionStore, writable_vectors
from otto_two_tower.evaluation import (
    commit_part,
    exact_search,
    identity,
    read_json,
    sha256_file,
    verified_part,
)
from otto_two_tower.logging_utils import configure_logging
from otto_two_tower.model import TwoTowerModel
from otto_two_tower.telemetry import TrainingHeartbeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-cache", type=Path, default=Path("/opt/ml/input/data/ranking"))
    parser.add_argument("--item-data", type=Path, default=Path("/opt/ml/input/data/items"))
    parser.add_argument("--model-dir", type=Path, default=Path("/opt/ml/input/data/trained"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/ml/checkpoints"))
    parser.add_argument("--expected-ranking-id", required=True)
    parser.add_argument("--expected-items-id", required=True)
    parser.add_argument("--training-input-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--k", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def export_predictions(args: argparse.Namespace, progress: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    logger = configure_logging("two_tower_evaluation", args.output_dir / "logs")
    if min(args.k, args.batch_size, args.chunk_size) <= 0:
        raise ValueError("search sizes must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA required; --allow-cpu is reserved for contract tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The reference measures exhaustive FP32 inner products; no ANN or BF16 search.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260906)
    ranking = read_json(args.ranking_cache / "manifest.json")
    items = read_json(args.item_data / "manifest.json")
    training = read_json(args.model_dir / "training_manifest.json")
    if identity(ranking) != args.expected_ranking_id or identity(items) != args.expected_items_id:
        raise ValueError("input manifests differ from the original training run")
    if training["input_id"] != args.training_input_id:
        raise ValueError("training input identity mismatch")
    if ranking["validation_manifest_id"] != training["validation_manifest_id"]:
        raise ValueError("validation manifest mismatch")
    model_config = ModelConfig(**training["config"]["model"])
    data_config = training["config"]["data"]
    fold = int(training["validation_fold"])
    if fold != data_config["validation_fold"]:
        raise ValueError("training fold mismatch")
    for root, manifest, names in (
        (args.ranking_cache, ranking, ("events", "examples", "labels")),
        (args.item_data, items, ("item_ids", "item_vectors", "aid_to_index")),
    ):
        for name in names:
            suffix = ".parquet" if root == args.ranking_cache else ".npy"
            progress.update(stage="verify_inputs", file=name)
            if sha256_file(root / (name + suffix)) != manifest[name + "_sha256"]:
                raise ValueError(f"input checksum mismatch: {name}")
    model_path = args.model_dir / "best_model.pt"
    contract = {
        "schema_version": 1,
        "code_commit": args.code_commit,
        "training_input_id": args.training_input_id,
        "training_manifest": training,
        "model_sha256": sha256_file(model_path),
        "ranking_manifest": ranking,
        "item_manifest": items,
        "search": {
            "method": "exhaustive_inner_product",
            "dtype": "float32",
            "tf32": False,
            "tie_break": "ascending_aid",
            "k": args.k,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
        },
        "torch_version": str(torch.__version__),
        "device": str(device),
    }
    input_id = identity(contract)
    contract_path = args.output_dir / "evaluation_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("output directory belongs to a different evaluation")
    write_json_atomic(contract, contract_path)
    progress.clear()
    progress.update(stage="load_model")
    vocabulary = ItemVocabulary.load(args.item_data)
    if args.k > len(vocabulary.item_ids):
        raise ValueError("k exceeds catalogue size")
    if len(np.unique(vocabulary.item_ids)) != len(vocabulary.item_ids):
        raise ValueError("duplicate catalogue IDs")
    examples = pq.read_table(
        args.ranking_cache / "examples.parquet", filters=[("fold", "=", fold)], columns=["session"]
    )
    sessions = np.sort(examples["session"].to_numpy().astype(np.int64))
    if len(sessions) != ranking["fold_session_counts"][fold] or len(np.unique(sessions)) != len(
        sessions
    ):
        raise ValueError("held-out session coverage mismatch")
    event_sessions = pq.read_table(
        args.ranking_cache / "events.parquet", filters=[("fold", "=", fold)], columns=["session"]
    )
    if not np.array_equal(np.unique(event_sessions["session"].to_numpy()), sessions):
        raise ValueError("held-out event/session coverage mismatch")
    store = PackedSessionStore.from_parquet(
        args.ranking_cache,
        vocabulary,
        max_seq_len=data_config["max_seq_len"],
        time_buckets=model_config.time_buckets,
    )
    model = TwoTowerModel(
        writable_vectors(vocabulary.vectors),
        padding_index=vocabulary.padding_index,
        config=model_config,
        max_seq_len=data_config["max_seq_len"],
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=True
    )
    model.to(device).eval()
    item_ids = torch.tensor(np.array(vocabulary.item_ids, dtype=np.int64), device=device)
    buckets = int(ranking["config"]["buckets"])
    receipts: list[dict[str, Any]] = []
    with torch.inference_mode():
        for objective_id, objective in enumerate(OBJECTIVES):
            progress.update(stage="encode_catalogue", objective=objective)
            directory = args.output_dir / "embeddings" / objective
            directory.mkdir(parents=True, exist_ok=True)
            embeddings = []
            for start in range(0, len(item_ids), args.chunk_size):
                end = min(start + args.chunk_size, len(item_ids))
                path = directory / f"part-{start:08d}.npy"
                if verified_part(path, input_id) is None:
                    indices = torch.arange(start, end, device=device)
                    objectives = torch.full_like(indices, objective_id)
                    vectors = model.encode_candidates(indices, objectives).float().cpu().numpy()
                    if not np.isfinite(vectors).all():
                        raise ValueError("non-finite candidate embedding")
                    temporary = path.with_suffix(".npy.tmp")
                    with temporary.open("wb") as handle:
                        np.save(handle, vectors, allow_pickle=False)
                    commit_part(temporary, path, input_id, rows=end - start)
                embeddings.append(torch.from_numpy(np.load(path, allow_pickle=False)))
                progress.update(examples=end)
            candidates = torch.cat(embeddings).to(device)
            del embeddings
            if candidates.shape != (len(item_ids), model_config.embedding_dim):
                raise ValueError("candidate embedding shape mismatch")
            if not torch.isfinite(candidates).all():
                raise ValueError("non-finite candidate embeddings")
            for bucket in range(buckets):
                progress.update(stage="retrieve", objective=objective, bucket=bucket, examples=0)
                path = args.output_dir / "predictions" / objective / f"part-{bucket:03d}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                existing = verified_part(path, input_id)
                bucket_sessions = sessions[sessions % buckets == bucket]
                if existing is not None:
                    if existing["rows"] != len(bucket_sessions):
                        raise ValueError("prediction receipt row count mismatch")
                    receipts.append({"path": str(path.relative_to(args.output_dir)), **existing})
                    logger.info("part_reused", extra={"stage": objective, "bucket": bucket})
                    continue
                bucket_started = time.perf_counter()
                predictions, similarities, batch_seconds = [], [], []
                for start in range(0, len(bucket_sessions), args.batch_size):
                    batch = bucket_sessions[start : start + args.batch_size]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    batch_started = time.perf_counter()
                    query = model.encode_session(
                        store.batch(batch, device),
                        torch.full((len(batch),), objective_id, device=device),
                    )
                    scores, aids = exact_search(
                        query.float(),
                        candidates,
                        item_ids,
                        k=args.k,
                        chunk_size=args.chunk_size,
                        validate_candidates=False,
                    )
                    predictions.append(aids.cpu().numpy().astype(np.int32))
                    similarities.append(scores.cpu().numpy())
                    batch_seconds.append(time.perf_counter() - batch_started)
                    progress.update(examples=start + len(batch))
                ids = (
                    np.concatenate(predictions) if predictions else np.empty((0, args.k), np.int32)
                )
                scores_array = (
                    np.concatenate(similarities)
                    if similarities
                    else np.empty((0, args.k), np.float32)
                )
                table = pa.table(
                    {
                        "session": pa.array(bucket_sessions),
                        "aids": pa.array(ids.tolist(), type=pa.list_(pa.int32())),
                        "scores": pa.array(scores_array.tolist(), type=pa.list_(pa.float32())),
                    }
                )
                temporary = path.with_suffix(".parquet.tmp")
                pq.write_table(table, temporary, compression="zstd")
                elapsed = time.perf_counter() - bucket_started
                commit_part(
                    temporary,
                    path,
                    input_id,
                    rows=len(bucket_sessions),
                    elapsed_seconds=elapsed,
                    batch_seconds=batch_seconds,
                )
                receipt = verified_part(path, input_id)
                assert receipt is not None
                receipts.append({"path": str(path.relative_to(args.output_dir)), **receipt})
                logger.info(
                    "part_complete",
                    extra={
                        "stage": objective,
                        "bucket": bucket,
                        "examples": len(bucket_sessions),
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )
            del candidates
    if len(receipts) != buckets * len(OBJECTIVES):
        raise ValueError("incomplete prediction parts")
    result = {
        "schema_version": 1,
        "status": "passed",
        "input_id": input_id,
        "training_input_id": args.training_input_id,
        "validation_fold": fold,
        "validation_manifest_id": training["validation_manifest_id"],
        "model_sha256": contract["model_sha256"],
        "code_commit": args.code_commit,
        "ranking_manifest": ranking,
        "model_config": asdict(model_config),
        "catalogue_items": len(item_ids),
        "sessions": len(sessions),
        "search": contract["search"],
        "parts": receipts,
        "elapsed_seconds_this_attempt": time.perf_counter() - started,
        "completed_retrieval_seconds": sum(r["elapsed_seconds"] for r in receipts),
        "approximation": "none; exhaustive reference; ANN serving benchmark pending",
    }
    write_json_atomic(result, args.output_dir / "prediction_manifest.json")
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("two_tower_evaluation", args.output_dir / "logs")
    progress: dict[str, Any] = {"stage": "initialization"}
    started = time.perf_counter()
    try:
        with TrainingHeartbeat(
            logger,
            stage="evaluation",
            interval_seconds=args.heartbeat_seconds,
            progress_provider=progress.copy,
        ):
            result = export_predictions(args, progress)
        logger.info("OTTO_TWO_TOWER_PREDICTIONS_PASSED", extra={"examples": result["sessions"]})
        return 0
    except Exception:
        logger.exception("evaluation_failed")
        raise
    finally:
        logger.info(
            "evaluation_complete",
            extra={"elapsed_seconds": round(time.perf_counter() - started, 3)},
        )
        if os.environ.get("SM_MODEL_DIR"):
            destination = Path(os.environ["SM_MODEL_DIR"])
            destination.mkdir(parents=True, exist_ok=True)
            manifest = args.output_dir / "prediction_manifest.json"
            if manifest.is_file():
                (destination / manifest.name).write_bytes(manifest.read_bytes())


if __name__ == "__main__":
    raise SystemExit(main())
