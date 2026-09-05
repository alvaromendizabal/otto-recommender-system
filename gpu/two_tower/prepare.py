from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from otto_two_tower.logging_utils import configure_logging
from otto_two_tower.telemetry import TrainingHeartbeat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat-seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("two_tower_item_export", args.output_dir / "logs")
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "step": 0,
        "examples": 0,
        "loss": None,
        "mrr": None,
        "hit10": None,
    }
    logger.info(
        "item_export_start",
        extra={
            "event": "item_export_start",
            "stage": "two_tower_item_export",
        },
    )

    with TrainingHeartbeat(
        logger,
        stage="two_tower_item_export",
        interval_seconds=args.heartbeat_seconds,
        progress_provider=lambda: dict(progress),
    ):
        from gensim.models import KeyedVectors

        vectors = KeyedVectors.load(str(args.vectors), mmap="r")
        progress["step"] = 1
        item_ids = np.fromiter(
            (int(value) for value in vectors.index_to_key),
            dtype=np.int32,
            count=len(vectors.index_to_key),
        )
        matrix = np.asarray(vectors.vectors, dtype=np.float32)
        if matrix.shape[0] != item_ids.shape[0]:
            raise RuntimeError("Item2Vec vocabulary and vector rows differ")
        progress["examples"] = int(item_ids.shape[0])
        progress["step"] = 2
        max_aid = int(item_ids.max())
        aid_to_index = np.full(max_aid + 1, -1, dtype=np.int32)
        aid_to_index[item_ids] = np.arange(item_ids.shape[0], dtype=np.int32)

        np.save(args.output_dir / "item_ids.npy", item_ids, allow_pickle=False)
        np.save(args.output_dir / "item_vectors.npy", matrix, allow_pickle=False)
        np.save(args.output_dir / "aid_to_index.npy", aid_to_index, allow_pickle=False)
        progress["step"] = 3
        manifest = {
            "items": int(item_ids.shape[0]),
            "dimension": int(matrix.shape[1]),
            "max_aid": max_aid,
            "source_vectors_sha256": sha256_file(args.vectors),
            "item_ids_sha256": sha256_file(args.output_dir / "item_ids.npy"),
            "item_vectors_sha256": sha256_file(args.output_dir / "item_vectors.npy"),
            "aid_to_index_sha256": sha256_file(args.output_dir / "aid_to_index.npy"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json_atomic(manifest, args.output_dir / "manifest.json")

    logger.info(
        "item_export_complete",
        extra={
            "event": "item_export_complete",
            "stage": "two_tower_item_export",
            "status": "passed",
            "examples": int(item_ids.shape[0]),
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("OTTO_TWO_TOWER_ITEM_EXPORT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
