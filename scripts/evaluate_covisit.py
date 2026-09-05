from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.bulk_evaluation import evaluate_covisit_retrieval

MIN_TOTAL_RAM_GIB = 16.0
MIN_AVAILABLE_RAM_GIB = 10.0
MIN_FREE_DISK_GIB = 25.0


def _gib(value: int) -> float:
    return value / (1024**3)


def resource_preflight() -> None:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.home())
    total = _gib(memory.total)
    available = _gib(memory.available)
    free_disk = _gib(disk.free)

    print(
        f"host_ram_gib={total:.1f} "
        f"available_ram_gib={available:.1f} "
        f"free_disk_gib={free_disk:.1f}",
        flush=True,
    )

    if total < MIN_TOTAL_RAM_GIB:
        raise RuntimeError(
            f"Retrieval evaluation requires at least "
            f"{MIN_TOTAL_RAM_GIB:.0f} GiB host RAM"
        )

    if available < MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError(
            f"Only {available:.1f} GiB RAM is available. "
            "Close other workloads before evaluation."
        )

    if free_disk < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"Only {free_disk:.1f} GiB disk is free. "
            "At least 25 GiB is required for DuckDB spill."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--covisit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[20, 50, 100, 200, 500, 1200],
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("data/interim/duckdb_evaluation"),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    resource_preflight()

    logger = configure_logging("covisit_evaluation")
    started = time.perf_counter()

    result = evaluate_covisit_retrieval(
        args.validation_cache,
        args.covisit_dir,
        args.output_dir,
        logger=logger,
        buckets=args.buckets,
        ks=args.ks,
        rrf_k=args.rrf_k,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_root=args.temp_root,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    elapsed = round(time.perf_counter() - started, 3)
    print(
        json.dumps(
            {
                "input_id": result.input_id,
                "config": asdict(result.config),
                "completed_buckets": result.completed_buckets,
                "elapsed_seconds": result.elapsed_seconds,
                "metrics": result.metrics,
                "incremental_hits": result.incremental_hits,
                "candidate_stats": result.candidate_stats,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wall_elapsed_seconds={elapsed}")
    print("OTTO_COVISIT_EVALUATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
