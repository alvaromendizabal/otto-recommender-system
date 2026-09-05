from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from gensim.models import KeyedVectors  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class FaissConfig:
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 256
    batch_size: int = 100_000
    threads: int = 4


@dataclass(frozen=True)
class FaissManifest:
    vector_size: int
    items: int
    config: FaissConfig
    index_sha256: str
    elapsed_seconds: float


def build_faiss_index(
    vectors_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    config: FaissConfig,
    heartbeat_seconds: float = 30.0,
) -> FaissManifest:
    """Build a cosine-similarity HNSW index with bounded batch memory."""
    if config.m <= 0:
        raise ValueError("m must be positive")
    if config.ef_construction <= 0 or config.ef_search <= 0:
        raise ValueError("FAISS ef values must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.threads <= 0:
        raise ValueError("threads must be positive")

    source = Path(vectors_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    vectors = KeyedVectors.load(str(source), mmap="r")
    dimension = int(vectors.vector_size)
    total_items = len(vectors)

    if total_items <= 0:
        raise RuntimeError("Item2Vec vectors are empty")

    faiss.omp_set_num_threads(config.threads)

    base = faiss.IndexHNSWFlat(
        dimension,
        config.m,
        faiss.METRIC_INNER_PRODUCT,
    )
    base.hnsw.efConstruction = config.ef_construction
    base.hnsw.efSearch = config.ef_search
    index = faiss.IndexIDMap2(base)

    progress: dict[str, int] = {
        "events": 0,
        "total_items": total_items,
    }
    started = time.perf_counter()

    logger.info(
        "faiss_build_start",
        extra={
            "event": "faiss_build_start",
            "stage": "faiss_build",
            "events": 0,
            "items": total_items,
            "dimension": dimension,
        },
    )

    with Heartbeat(
        logger,
        stage="faiss_build",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress.copy,
    ):
        for start in range(0, total_items, config.batch_size):
            end = min(start + config.batch_size, total_items)
            batch = np.asarray(
                vectors.vectors[start:end],
                dtype=np.float32,
            ).copy()
            faiss.normalize_L2(batch)

            ids = np.asarray(
                vectors.index_to_key[start:end],
                dtype=np.int64,
            )
            index.add_with_ids(batch, ids)
            progress["events"] = end

            logger.info(
                "faiss_batch_complete",
                extra={
                    "event": "faiss_batch_complete",
                    "stage": "faiss_build",
                    "events": end,
                    "items": total_items,
                    "elapsed_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                },
            )

    if index.ntotal != total_items:
        raise RuntimeError(
            f"FAISS index contains {index.ntotal} items; expected {total_items}"
        )

    output_path = destination / "item.index"
    temp_path = destination / ".item.index.tmp"
    temp_path.unlink(missing_ok=True)

    faiss.write_index(index, str(temp_path))
    verification = faiss.read_index(str(temp_path))

    if verification.ntotal != total_items:
        raise RuntimeError("persisted FAISS index failed ntotal verification")

    os.replace(temp_path, output_path)
    elapsed = round(time.perf_counter() - started, 3)

    manifest = FaissManifest(
        vector_size=dimension,
        items=total_items,
        config=config,
        index_sha256=sha256_file(output_path),
        elapsed_seconds=elapsed,
    )

    manifest_path = destination / "manifest.json"
    temp_manifest = destination / ".manifest.json.tmp"
    temp_manifest.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_manifest, manifest_path)

    logger.info(
        "faiss_build_complete",
        extra={
            "event": "faiss_build_complete",
            "stage": "faiss_build",
            "status": "passed",
            "events": total_items,
            "elapsed_seconds": elapsed,
        },
    )

    return manifest
