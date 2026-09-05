from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.validation_cache import (
    build_retrieval_validation_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--flush-sessions", type=int, default=10_000)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    logger = configure_logging("retrieval_validation")
    started = time.perf_counter()

    manifest = build_retrieval_validation_cache(
        args.validation_dir,
        args.output_dir,
        logger=logger,
        buckets=args.buckets,
        flush_sessions=args.flush_sessions,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    elapsed = round(time.perf_counter() - started, 3)
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"total_elapsed_seconds={elapsed}")
    print("OTTO_RETRIEVAL_VALIDATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
