from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from otto_recsys.logging_utils import configure_logging
from otto_recsys.ranking.hard_negatives import mine_hard_negatives

_MIN_TOTAL_RAM_GIB = 16.0
_MIN_AVAILABLE_RAM_GIB = 12.0
_MIN_FREE_DISK_GIB = 25.0


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
    if total_ram < _MIN_TOTAL_RAM_GIB:
        raise RuntimeError("hard-negative mining requires at least 16 GiB RAM")
    if available_ram < _MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError("hard-negative mining requires at least 12 GiB available RAM")
    if free_disk < _MIN_FREE_DISK_GIB:
        raise RuntimeError("hard-negative mining requires at least 25 GiB free disk")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mine false-negative-safe hard negatives from co-visitation and Item2Vec "
            "for two-tower and ranking training."
        )
    )
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--covisit-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--source-k", type=int, default=1200)
    parser.add_argument("--item2vec-k", type=int, default=800)
    parser.add_argument("--hard-negatives", type=int, default=64)
    parser.add_argument("--ef-search", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("data/interim/duckdb_hard_negatives"),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    started = time.perf_counter()
    resource_preflight(Path.home())
    logger = configure_logging("hard_negative_mining")
    manifest = mine_hard_negatives(
        args.training_cache,
        args.covisit_dir,
        args.vectors,
        args.index,
        args.output_dir,
        logger=logger,
        buckets=args.buckets,
        source_k=args.source_k,
        item2vec_k=args.item2vec_k,
        hard_negatives=args.hard_negatives,
        ef_search=args.ef_search,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_root=args.temp_root,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"wall_elapsed_seconds={time.perf_counter() - started:.3f}", flush=True)
    print("OTTO_HARD_NEGATIVE_MINING_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
