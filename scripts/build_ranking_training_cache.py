from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from otto_recsys.logging_utils import configure_logging
from otto_recsys.ranking.training_cache import build_ranking_training_cache

_MIN_TOTAL_RAM_GIB = 8.0
_MIN_AVAILABLE_RAM_GIB = 4.0
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
    if total_ram < _MIN_TOTAL_RAM_GIB:
        raise RuntimeError("host RAM is below the ranking-cache minimum")
    if available_ram < _MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError("available RAM is below the ranking-cache minimum")
    if free_disk < _MIN_FREE_DISK_GIB:
        raise RuntimeError("free disk is below the ranking-cache minimum")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic supervised prefixes from the frozen pre-validation "
            "training universe for ranker and neural-retriever training."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--sample-denominator", type=int, default=8)
    parser.add_argument("--sample-remainder", type=int, default=0)
    parser.add_argument("--min-prefix-events", type=int, default=2)
    parser.add_argument("--max-prefix-events", type=int, default=50)
    parser.add_argument("--flush-examples", type=int, default=5_000)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    started = time.perf_counter()
    resource_preflight(Path.home())
    logger = configure_logging("ranking_training_cache")
    manifest = build_ranking_training_cache(
        args.source,
        args.source_manifest,
        args.output_dir,
        logger=logger,
        buckets=args.buckets,
        seed=args.seed,
        sample_denominator=args.sample_denominator,
        sample_remainder=args.sample_remainder,
        min_prefix_events=args.min_prefix_events,
        max_prefix_events=args.max_prefix_events,
        flush_examples=args.flush_examples,
        max_examples=args.max_examples,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"wall_elapsed_seconds={time.perf_counter() - started:.3f}", flush=True)
    print("OTTO_RANKING_TRAINING_CACHE_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
