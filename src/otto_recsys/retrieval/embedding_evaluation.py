from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from gensim.models import KeyedVectors  # type: ignore[import-untyped]

from otto_recsys.runtime import Heartbeat

_OBJECTIVES = ("clicks", "carts", "orders")
_WEIGHTS = {
    "clicks": np.asarray([1.0, 2.0, 2.5], dtype=np.float32),
    "carts": np.asarray([1.0, 4.0, 3.0], dtype=np.float32),
    "orders": np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
}
_METRIC_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


@dataclass(frozen=True)
class EmbeddingEvaluationManifest:
    buckets: int
    ks: tuple[int, ...]
    ann_k: int
    ef_search: int
    sessions: int
    elapsed_seconds: float
    metrics: dict[str, float]


def _labels_for_bucket(
    labels_path: Path,
    bucket: int,
) -> dict[tuple[int, str], set[int]]:
    table = pq.read_table(
        labels_path,
        filters=[("bucket", "=", bucket)],
        columns=["session", "objective", "aid"],
    )
    result: dict[tuple[int, str], set[int]] = {}

    sessions = table.column("session").to_pylist()
    objectives = table.column("objective").to_pylist()
    aids = table.column("aid").to_pylist()

    for session, objective, aid in zip(
        sessions,
        objectives,
        aids,
        strict=True,
    ):
        result.setdefault(
            (int(session), str(objective)),
            set(),
        ).add(int(aid))

    return result


def _queries_for_bucket(
    items_path: Path,
    vectors: KeyedVectors,
    bucket: int,
    objective: str,
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        items_path,
        filters=[("bucket", "=", bucket)],
        columns=["session", "aid", "event_type", "recency_rank"],
    )

    sessions = table.column("session").to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)
    aids = table.column("aid").to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)
    event_types = table.column("event_type").to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)
    recency_rank = table.column("recency_rank").to_numpy(
        zero_copy_only=False,
    ).astype(np.float32, copy=False)

    if sessions.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(
            (0, vectors.vector_size),
            dtype=np.float32,
        )

    unique_sessions, inverse = np.unique(sessions, return_inverse=True)

    vector_rows: list[int] = []
    keep_mask = np.zeros(aids.shape[0], dtype=bool)
    for index, aid in enumerate(aids.tolist()):
        vector_index = vectors.key_to_index.get(int(aid))
        if vector_index is not None:
            vector_rows.append(int(vector_index))
            keep_mask[index] = True

    if not vector_rows:
        return unique_sessions, np.zeros(
            (unique_sessions.size, vectors.vector_size),
            dtype=np.float32,
        )

    kept_inverse = inverse[keep_mask]
    kept_types = event_types[keep_mask]
    kept_recency = recency_rank[keep_mask]
    item_vectors = np.asarray(
        vectors.vectors[np.asarray(vector_rows, dtype=np.int64)],
        dtype=np.float32,
    )

    recency_weight = 1.0 / (1.0 + 0.10 * (kept_recency - 1.0))
    action_weight = _WEIGHTS[objective][kept_types]
    row_weight = recency_weight * action_weight

    queries = np.zeros(
        (unique_sessions.size, vectors.vector_size),
        dtype=np.float32,
    )
    weight_sum = np.zeros(unique_sessions.size, dtype=np.float32)

    np.add.at(queries, kept_inverse, item_vectors * row_weight[:, None])
    np.add.at(weight_sum, kept_inverse, row_weight)

    valid = weight_sum > 0
    queries[valid] /= weight_sum[valid, None]
    faiss.normalize_L2(queries)

    return unique_sessions, queries


def evaluate_embedding_retrieval(
    validation_cache_dir: str | Path,
    vectors_path: str | Path,
    index_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    ks: tuple[int, ...] = (20, 50, 100, 200),
    ann_k: int = 200,
    ef_search: int = 256,
    heartbeat_seconds: float = 30.0,
) -> EmbeddingEvaluationManifest:
    """Evaluate target-conditioned pooled Item2Vec queries against FAISS."""
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain positive integers")
    if ann_k < max(normalized_ks):
        raise ValueError("ann_k must be at least max(ks)")

    cache = Path(validation_cache_dir).resolve()
    items_path = cache / "items.parquet"
    labels_path = cache / "labels.parquet"
    vectors = KeyedVectors.load(str(Path(vectors_path).resolve()), mmap="r")
    index = faiss.read_index(str(Path(index_path).resolve()))
    faiss.ParameterSpace().set_index_parameter(
        index,
        "efSearch",
        ef_search,
    )

    hits = {
        objective: {k: 0 for k in normalized_ks}
        for objective in _OBJECTIVES
    }
    denominators = {
        objective: {k: 0 for k in normalized_ks}
        for objective in _OBJECTIVES
    }

    progress: dict[str, int] = {
        "bucket": 0,
        "buckets": buckets,
        "sessions": 0,
    }
    started = time.perf_counter()

    logger.info(
        "embedding_evaluation_start",
        extra={
            "event": "embedding_evaluation_start",
            "stage": "embedding_evaluation",
            "buckets": buckets,
            "ann_k": ann_k,
            "ef_search": ef_search,
        },
    )

    with Heartbeat(
        logger,
        stage="embedding_evaluation",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress.copy,
    ):
        for bucket in range(buckets):
            labels = _labels_for_bucket(labels_path, bucket)
            bucket_sessions: set[int] = set()

            for objective in _OBJECTIVES:
                session_ids, queries = _queries_for_bucket(
                    items_path,
                    vectors,
                    bucket,
                    objective,
                )

                if queries.shape[0] == 0:
                    continue

                _, neighbors = index.search(queries, ann_k)

                for row_index, session in enumerate(session_ids.tolist()):
                    bucket_sessions.add(int(session))
                    truth = labels.get((int(session), objective), set())
                    if not truth:
                        continue

                    prediction_row = [
                        int(aid)
                        for aid in neighbors[row_index].tolist()
                        if int(aid) >= 0
                    ]

                    for k in normalized_ks:
                        hits[objective][k] += len(
                            set(prediction_row[:k]) & truth
                        )
                        denominators[objective][k] += min(k, len(truth))

            progress["bucket"] = bucket + 1
            progress["sessions"] += len(bucket_sessions)

            logger.info(
                "embedding_bucket_complete",
                extra={
                    "event": "embedding_bucket_complete",
                    "stage": "embedding_evaluation",
                    "bucket": bucket,
                    "buckets": buckets,
                    "sessions": len(bucket_sessions),
                    "elapsed_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                },
            )

    metrics: dict[str, float] = {}
    for k in normalized_ks:
        weighted = 0.0
        for objective in _OBJECTIVES:
            denominator = denominators[objective][k]
            recall = hits[objective][k] / denominator if denominator else 0.0
            metrics[f"{objective}.recall_{k}"] = recall
            weighted += _METRIC_WEIGHTS[objective] * recall
        metrics[f"weighted_recall_{k}"] = weighted

    elapsed = round(time.perf_counter() - started, 3)
    manifest = EmbeddingEvaluationManifest(
        buckets=buckets,
        ks=normalized_ks,
        ann_k=ann_k,
        ef_search=ef_search,
        sessions=progress["sessions"],
        elapsed_seconds=elapsed,
        metrics=metrics,
    )

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "metrics.json"
    temp_path = destination / ".metrics.json.tmp"
    temp_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, manifest_path)

    logger.info(
        "embedding_evaluation_complete",
        extra={
            "event": "embedding_evaluation_complete",
            "stage": "embedding_evaluation",
            "status": "passed",
            "sessions": manifest.sessions,
            "elapsed_seconds": elapsed,
            "weighted_recall_20": metrics.get("weighted_recall_20", 0.0),
        },
    )

    return manifest
