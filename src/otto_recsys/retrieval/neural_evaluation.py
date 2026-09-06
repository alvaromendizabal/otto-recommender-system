"""Paired held-out comparison against the frozen candidate discovery pool."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb
import faiss
import numpy as np
import pyarrow.parquet as pq
from gensim.models import KeyedVectors  # type: ignore[import-untyped]

from otto_recsys.cloud.comparison_checkpoints import S3ComparisonCheckpoints
from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.retrieval.candidate_union import (
    append_item2vec_candidates,
    configure_connection,
    create_covisit_source_candidates,
    create_validation_tables,
)
from otto_recsys.runtime import Heartbeat

OBJECTIVES = ("clicks", "carts", "orders")
DEPTHS = (20, 50, 100, 200, 400, 800)
WEIGHTS = np.array([0.1, 0.3, 0.6])


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def session_counts(
    truth: set[int], base: set[int], neural: list[int], depths: tuple[int, ...] = DEPTHS
) -> list[int]:
    """Top-20 ceilings; raw exclusive hits are separate from capped gains."""
    if len(neural) != len(set(neural)) or any(aid < 0 for aid in neural):
        raise ValueError("neural predictions contain duplicate or invalid aids")
    if not depths or min(depths) < 1 or max(depths) > len(neural):
        raise ValueError("prediction depth is insufficient")
    base_hits = base & truth
    counts = [min(20, len(truth)), min(20, len(base_hits))]
    for k in depths:
        hits = set(neural[:k]) & truth
        counts.extend((min(20, len(hits)), min(20, len(base_hits | hits)), len(hits - base_hits)))
    return counts


def summarize_counts(
    counts: np.ndarray,
    *,
    depths: tuple[int, ...] = DEPTHS,
    iterations: int = 500,
    seed: int = 20260906,
) -> dict[str, Any]:
    """Resample complete sessions, preserving all objectives and both systems."""
    if iterations < 2 or counts.ndim != 3 or counts.shape[1:] != (3, 2 + 3 * len(depths)):
        raise ValueError("invalid bootstrap shape or iteration count")
    totals = counts.sum(axis=0, dtype=np.float64)
    if np.any(totals[:, 0] <= 0):
        raise ValueError("each objective needs held-out labels")
    rng = np.random.default_rng(seed)
    delta_samples = np.empty((iterations, 3, len(depths)))
    for iteration in range(iterations):
        # Same sampled session indices for every objective and candidate depth.
        sample = counts[rng.integers(0, len(counts), size=len(counts))].sum(
            axis=0, dtype=np.float64
        )
        if np.any(sample[:, 0] == 0):
            raise ValueError("bootstrap sample lacks an objective; enlarge evaluation cohort")
        delta_samples[iteration] = (sample[:, 3::3] - sample[:, 1:2]) / sample[:, 0:1]
    rows = []
    for index, k in enumerate(depths):
        neural = totals[:, 2 + 3 * index] / totals[:, 0]
        base = totals[:, 1] / totals[:, 0]
        union = totals[:, 3 + 3 * index] / totals[:, 0]
        per_objective = {}
        for objective_id, objective in enumerate(OBJECTIVES):
            per_objective[objective] = {
                "denominator": int(totals[objective_id, 0]),
                "base_ceiling": float(base[objective_id]),
                "neural_ceiling": float(neural[objective_id]),
                "union_ceiling": float(union[objective_id]),
                "incremental_ceiling": float(union[objective_id] - base[objective_id]),
                "neural_only_positive_hits": int(totals[objective_id, 4 + 3 * index]),
                "incremental_ci95": np.quantile(
                    delta_samples[:, objective_id, index], [0.025, 0.975]
                ).tolist(),
            }
        weighted_samples = delta_samples[:, :, index] @ WEIGHTS
        rows.append(
            {
                "neural_k": k,
                "objectives": per_objective,
                "weighted_base_ceiling": float(base @ WEIGHTS),
                "weighted_neural_ceiling": float(neural @ WEIGHTS),
                "weighted_union_ceiling": float(union @ WEIGHTS),
                "weighted_incremental_ceiling": float((union - base) @ WEIGHTS),
                "weighted_incremental_ci95": np.quantile(weighted_samples, [0.025, 0.975]).tolist(),
            }
        )
    return {
        "points": rows,
        "bootstrap": {
            "unit": "session",
            "method": "paired percentile",
            "iterations": iterations,
            "seed": seed,
            "confidence": 0.95,
        },
    }


def evaluate_neural_retrieval(
    ranking_dir: Path,
    predictions_dir: Path,
    covisit_dir: Path,
    vectors_path: Path,
    index_path: Path,
    output_dir: Path,
    *,
    logger: logging.Logger,
    source_k: int = 1200,
    ann_k: int = 800,
    ef_search: int = 1024,
    threads: int = 4,
    memory_limit: str = "8GB",
    heartbeat_seconds: float = 30.0,
    iterations: int = 500,
    seed: int = 20260906,
    checkpoint_store: S3ComparisonCheckpoints | None = None,
) -> dict[str, Any]:
    if min(source_k, ann_k, ef_search, threads) <= 0 or iterations < 2:
        raise ValueError("invalid evaluation settings")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = read_json(predictions_dir / "prediction_manifest.json")
    ranking = read_json(ranking_dir / "manifest.json")
    if prediction["status"] != "passed" or prediction["ranking_manifest"] != ranking:
        raise ValueError("prediction/ranking contract mismatch")
    if prediction["search"]["k"] < max(DEPTHS):
        raise ValueError("neural export does not cover all candidate depths")
    fold = int(prediction["validation_fold"])
    buckets = int(ranking["config"]["buckets"])
    # Hash the actual baseline files, including separately stored Gensim arrays.
    input_files = [ranking_dir / f"{name}.parquet" for name in ("items", "labels", "examples")]
    input_files += [
        covisit_dir / name
        for name in (
            "time.parquet",
            "type.parquet",
            "buy.parquet",
            "time.json",
            "type.json",
            "buy.json",
        )
    ]
    input_files += sorted(vectors_path.parent.glob(vectors_path.name + "*"))
    input_files += [
        vectors_path.parent / "manifest.json",
        index_path,
        index_path.parent / "manifest.json",
    ]
    checksums = {str(path.resolve()): sha256_file(path) for path in input_files}
    for name in ("items", "labels", "examples"):
        if (
            checksums[str((ranking_dir / (name + ".parquet")).resolve())]
            != ranking[name + "_sha256"]
        ):
            raise ValueError(f"ranking checksum mismatch: {name}")
    # Exclude filesystem locations from identity, preserving ordered content hashes.
    contract = {
        "schema_version": 1,
        "comparison_source_sha256": sha256_file(Path(__file__)),
        "baseline_source_sha256": sha256_file(Path(__file__).with_name("candidate_union.py")),
        "runtime": {
            "numpy": np.__version__,
            "faiss": faiss.__version__,
            "duckdb": duckdb.__version__,
            "threads": threads,
            "memory_limit": memory_limit,
        },
        "prediction_input_id": prediction["input_id"],
        "baseline_checksums": list(checksums.values()),
        "depths": list(DEPTHS),
        "source_k": source_k,
        "ann_k": ann_k,
        "ef_search": ef_search,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "metric": "per-session hits and denominators capped at 20",
    }
    input_id = canonical_json_sha256(contract)
    contract_path = output_dir / "comparison_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("output directory belongs to a different comparison")
    write_json(contract_path, contract)
    expected_paths = {
        f"predictions/{objective}/part-{bucket:03d}.parquet"
        for objective in OBJECTIVES
        for bucket in range(buckets)
    }
    parts = {part["path"]: part for part in prediction["parts"]}
    if set(parts) != expected_paths or len(parts) != len(prediction["parts"]):
        raise ValueError("missing, duplicate, or unexpected prediction parts")
    for relative, receipt in parts.items():
        if (
            receipt["input_id"] != prediction["input_id"]
            or sha256_file(predictions_dir / relative) != receipt["sha256"]
        ):
            raise ValueError(f"prediction part checksum or identity mismatch: {relative}")
    sessions = np.sort(
        pq.read_table(
            ranking_dir / "examples.parquet", filters=[("fold", "=", fold)], columns=["session"]
        )["session"]
        .to_numpy()
        .astype(np.int64)
    )
    if len(sessions) != prediction["sessions"] or len(np.unique(sessions)) != len(sessions):
        raise ValueError("held-out session coverage mismatch")
    if checkpoint_store is not None:
        checkpoint_store.restore(output_dir, input_id)
    # Shared baseline code now reads only this held-out cohort.
    items_path = output_dir / "items.parquet"
    labels_path = output_dir / "labels.parquet"
    for name, path in (("items", items_path), ("labels", labels_path)):
        table = pq.read_table(ranking_dir / (name + ".parquet"), filters=[("fold", "=", fold)])
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)
    vectors = KeyedVectors.load(str(vectors_path), mmap="r")
    index = faiss.read_index(str(index_path))
    faiss.omp_set_num_threads(threads)
    all_counts = np.zeros((len(sessions), 3, 2 + 3 * len(DEPTHS)), dtype=np.int32)
    progress: dict[str, Any] = {"bucket": 0, "buckets": buckets}
    with Heartbeat(
        logger,
        stage="neural_comparison",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress.copy,
    ):
        for bucket in range(buckets):
            bucket_started = time.perf_counter()
            subset = sessions[sessions % buckets == bucket]
            path = output_dir / "parts" / f"part-{bucket:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path = path.with_suffix(".json")
            try:
                receipt = read_json(receipt_path) if receipt_path.exists() else {}
            except (OSError, ValueError):
                receipt = {}
                logger.warning("comparison_receipt_invalid", extra={"bucket": bucket})
            reuse = (
                path.exists()
                and receipt.get("input_id") == input_id
                and receipt.get("sha256") == sha256_file(path)
            )
            if reuse:
                with np.load(path, allow_pickle=False) as stored:
                    if not np.array_equal(stored["sessions"], subset):
                        raise ValueError("saved comparison session mismatch")
                    counts = stored["counts"]
                if counts.shape != (len(subset), 3, all_counts.shape[2]):
                    raise ValueError("saved comparison shape mismatch")
            else:
                counts = np.zeros((len(subset), 3, all_counts.shape[2]), dtype=np.int32)
                with duckdb.connect() as connection:
                    configure_connection(
                        connection,
                        threads=threads,
                        memory_limit=memory_limit,
                        temp_directory=output_dir / "temporary" / str(bucket),
                    )
                    create_validation_tables(
                        connection, items_path=items_path, labels_path=labels_path, bucket=bucket
                    )
                    create_covisit_source_candidates(
                        connection, covisit_dir=covisit_dir, source_k=source_k
                    )
                    append_item2vec_candidates(
                        connection,
                        items_path=items_path,
                        vectors=vectors,
                        index=index,
                        bucket=bucket,
                        ann_k=ann_k,
                        ef_search=ef_search,
                    )
                    labels = connection.execute(
                        "SELECT DISTINCT session, objective, aid FROM vlabels"
                    ).fetchall()
                    base_rows = connection.execute("""SELECT DISTINCT l.session, l.objective, l.aid
                        FROM vlabels l JOIN source_candidates c
                        USING (session, objective, aid)""").fetchall()
                truth: dict[tuple[int, str], set[int]] = {}
                base: dict[tuple[int, str], set[int]] = {}
                for source, destination in ((labels, truth), (base_rows, base)):
                    for session, objective, aid in source:
                        destination.setdefault((int(session), str(objective)), set()).add(int(aid))
                for objective_id, objective in enumerate(OBJECTIVES):
                    table = pq.read_table(
                        predictions_dir / f"predictions/{objective}/part-{bucket:03d}.parquet"
                    )
                    actual_sessions = table["session"].to_numpy()
                    if not np.array_equal(actual_sessions, subset):
                        raise ValueError(
                            "prediction sessions missing, reordered, or outside holdout"
                        )
                    for row, (session, aids) in enumerate(
                        zip(subset, table["aids"].to_pylist(), strict=True)
                    ):
                        key = (int(session), objective)
                        counts[row, objective_id] = session_counts(
                            truth.get(key, set()), base.get(key, set()), aids
                        )
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, sessions=subset, counts=counts)
                temporary.replace(path)
                write_json(
                    receipt_path,
                    {
                        "input_id": input_id,
                        "sha256": sha256_file(path),
                        "elapsed_seconds": time.perf_counter() - bucket_started,
                    },
                )
            if checkpoint_store is not None:
                checkpoint_store.publish_part(output_dir, bucket, input_id)
            all_counts[np.searchsorted(sessions, subset)] = counts
            progress["bucket"] = bucket + 1
            logger.info(
                "comparison_part_reused" if reuse else "comparison_part_complete",
                extra={
                    "stage": "neural_comparison",
                    "bucket": bucket,
                    "elapsed_seconds": round(time.perf_counter() - bucket_started, 3),
                },
            )
        progress["stage"] = "bootstrap"
        summary = summarize_counts(all_counts, iterations=iterations, seed=seed)
    result = {
        "schema_version": 1,
        "status": "passed",
        "input_id": input_id,
        "prediction_input_id": prediction["input_id"],
        "validation_fold": fold,
        "sessions": len(sessions),
        "contract": contract,
        **summary,
        "elapsed_seconds_this_attempt": round(time.perf_counter() - started, 3),
        "completed_bucket_compute_seconds": sum(
            read_json(output_dir / "parts" / f"part-{bucket:03d}.json")["elapsed_seconds"]
            for bucket in range(buckets)
        ),
        "interpretation": (
            "Fixed base discovery pool plus neural top-K; "
            "union is an ideal top-20 ceiling, not ranked Recall@20."
        ),
        "generalization": (
            "Fold used for checkpoint selection; exploratory validation, not an untouched test set."
        ),
        "prediction_search": prediction["search"],
        "ann_serving_benchmark": "Latency is evaluated separately by the ANN benchmark.",
    }
    write_json(output_dir / "metrics.json", result)
    if checkpoint_store is not None:
        checkpoint_store.publish_metrics(output_dir)
    logger.info(
        "neural_comparison_complete",
        extra={"elapsed_seconds": result["elapsed_seconds_this_attempt"]},
    )
    return result
