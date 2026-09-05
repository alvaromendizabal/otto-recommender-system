from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import psutil

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.covisit import build_covisit_matrix

MATRIX_SPECS = {
    "time": 80,
    "type": 120,
    "buy": 120,
}

MIN_HOST_RAM_GIB = 12.0
MIN_AVAILABLE_RAM_GIB = 8.0
MIN_FREE_DISK_GIB = 25.0


def _gib(value: int) -> float:
    return value / (1024**3)


def safe_duckdb_memory_limit() -> str:
    """Return a conservative DuckDB memory budget for the current host."""
    memory = psutil.virtual_memory()
    total_gib = _gib(memory.total)
    available_gib = _gib(memory.available)

    if total_gib < MIN_HOST_RAM_GIB:
        raise RuntimeError(
            f"Host RAM is only {total_gib:.1f} GiB. "
            "Full co-visitation requires a larger instance."
        )

    if available_gib < MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError(
            f"Only {available_gib:.1f} GiB RAM is currently available. "
            "Close other workloads before building co-visitation."
        )

    # Intentionally conservative. DuckDB can spill to disk.
    # Keep substantial memory available to JupyterLab, the OS,
    # page cache, Python, and transient execution state.
    budget_gib = min(
        8,
        max(4, int(total_gib * 0.25)),
    )

    return f"{budget_gib}GB"


def resource_preflight(temp_directory: Path) -> str:
    """Validate host resources before launching a graph build."""
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(temp_directory.parent)

    total_ram_gib = _gib(memory.total)
    available_ram_gib = _gib(memory.available)
    free_disk_gib = _gib(disk.free)

    print(
        f"host_ram_gib={total_ram_gib:.1f} "
        f"available_ram_gib={available_ram_gib:.1f} "
        f"free_disk_gib={free_disk_gib:.1f}",
        flush=True,
    )

    if free_disk_gib < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"Only {free_disk_gib:.1f} GiB disk is free. "
            f"At least {MIN_FREE_DISK_GIB:.0f} GiB is required."
        )

    return safe_duckdb_memory_limit()


def completed_matrix_is_valid(
    output_path: Path,
    manifest_path: Path,
) -> bool:
    """Verify that an existing matrix is fully committed and intact."""
    if not output_path.is_file() or not manifest_path.is_file():
        return False

    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    expected_hash = payload.get("output_sha256")
    rows = payload.get("rows")

    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or not isinstance(rows, int)
        or rows <= 0
    ):
        return False

    return sha256_file(output_path) == expected_hash


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session-items",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--matrix",
        choices=tuple(MATRIX_SPECS),
        required=True,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--position-window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--temp-directory",
        type=Path,
        default=Path("data/interim/duckdb"),
    )

    args = parser.parse_args()

    started = time.perf_counter()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.temp_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = configure_logging(
        f"covisit_{args.matrix}"
    )

    output_path = (
        args.output_dir
        / f"{args.matrix}.parquet"
    )
    manifest_path = (
        args.output_dir
        / f"{args.matrix}.json"
    )

    print("\n=== RESOURCE PREFLIGHT ===", flush=True)

    memory_limit = resource_preflight(
        args.temp_directory
    )

    print(
        f"duckdb_memory_limit={memory_limit}",
        flush=True,
    )
    print(
        f"threads={args.threads}",
        flush=True,
    )

    if completed_matrix_is_valid(
        output_path,
        manifest_path,
    ):
        elapsed = round(
            time.perf_counter() - started,
            3,
        )

        print(
            f"{args.matrix}: existing matrix passed SHA-256 verification",
            flush=True,
        )
        print(
            f"total_elapsed_seconds={elapsed}",
            flush=True,
        )
        print(
            "OTTO_COVISIT_MATRIX_PASSED "
            f"matrix={args.matrix} status=already_complete",
            flush=True,
        )

        return 0

    temporary = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temporary.exists():
        print(
            f"Removing incomplete temporary artifact: {temporary}",
            flush=True,
        )
        temporary.unlink()

    manifest = build_covisit_matrix(
        args.matrix,
        args.session_items,
        output_path,
        manifest_path,
        logger=logger,
        top_k=MATRIX_SPECS[args.matrix],
        position_window=args.position_window,
        threads=args.threads,
        memory_limit=memory_limit,
        temp_directory=args.temp_directory,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    if not completed_matrix_is_valid(
        output_path,
        manifest_path,
    ):
        raise RuntimeError(
            f"{args.matrix} matrix failed post-build integrity verification"
        )

    elapsed = round(
        time.perf_counter() - started,
        3,
    )

    print(
        json.dumps(
            {
                "matrix": args.matrix,
                "rows": manifest.rows,
                "sha256": manifest.output_sha256,
                "duckdb_memory_limit": memory_limit,
                "threads": args.threads,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "OTTO_COVISIT_MATRIX_PASSED "
        f"matrix={args.matrix} status=built",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
