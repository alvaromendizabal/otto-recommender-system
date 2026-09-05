from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import psutil

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.candidate_budget import evaluate_candidate_budget

_MIN_HOST_RAM_GIB = 12.0
_MIN_AVAILABLE_RAM_GIB = 8.0
_MIN_FREE_DISK_GIB = 20.0


def _gib(value: int) -> float:
    return value / (1024**3)


def resource_preflight(path: Path) -> None:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(path)
    total_ram = _gib(memory.total)
    available_ram = _gib(memory.available)
    free_disk = _gib(disk.free)

    print(
        f"host_ram_gib={total_ram:.1f} "
        f"available_ram_gib={available_ram:.1f} "
        f"free_disk_gib={free_disk:.1f}",
        flush=True,
    )
    if total_ram < _MIN_HOST_RAM_GIB:
        raise RuntimeError(
            f"Host RAM is {total_ram:.1f} GiB; at least "
            f"{_MIN_HOST_RAM_GIB:.0f} GiB is required."
        )
    if available_ram < _MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError(
            f"Available RAM is {available_ram:.1f} GiB; close other workloads."
        )
    if free_disk < _MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"Free disk is {free_disk:.1f} GiB; at least "
            f"{_MIN_FREE_DISK_GIB:.0f} GiB is required."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Item2Vec marginal recall versus candidate cost and select "
            "objective-aware ANN quotas."
        )
    )
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--covisit-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--source-k", type=int, default=1200)
    parser.add_argument(
        "--item2vec-quotas",
        nargs="+",
        type=int,
        default=[0, 10, 20, 50, 100, 150, 200],
    )
    parser.add_argument("--capture-fraction", type=float, default=0.95)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("data/interim/duckdb_candidate_budget"),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    started = time.perf_counter()
    resource_preflight(Path.home())
    logger = configure_logging("candidate_budget")

    result = evaluate_candidate_budget(
        args.validation_cache,
        args.covisit_dir,
        args.vectors,
        args.index,
        args.output_dir,
        logger=logger,
        buckets=args.buckets,
        source_k=args.source_k,
        item2vec_quotas=tuple(args.item2vec_quotas),
        capture_fraction=args.capture_fraction,
        ef_search=args.ef_search,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_root=args.temp_root,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    print(
        json.dumps(
            result.recommended_item2vec_k,
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(json.dumps(result.candidate_stats, indent=2, sort_keys=True))
    print(
        f"wall_elapsed_seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    print("OTTO_CANDIDATE_BUDGET_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
