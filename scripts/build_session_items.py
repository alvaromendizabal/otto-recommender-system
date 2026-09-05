from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.session_items import (
    build_session_item_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-pattern",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--memory-limit",
        default="2GB",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    logger = configure_logging("session_items")
    started = time.perf_counter()

    manifest = build_session_item_cache(
        args.processed_pattern,
        args.validation_manifest,
        args.output,
        args.manifest,
        logger=logger,
        max_items_per_session=args.max_items,
        threads=args.threads,
        memory_limit=args.memory_limit,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    elapsed = round(
        time.perf_counter() - started,
        3,
    )

    print(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        )
    )
    print(f"total_elapsed_seconds={elapsed}")
    print("OTTO_SESSION_ITEMS_PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
