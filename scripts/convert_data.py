from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.data.convert import convert_jsonl_to_parquet
from otto_recsys.logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)

    parser.add_argument(
        "--events-per-part",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--max-sessions", type=int)

    args = parser.parse_args()

    logger = configure_logging("convert_data")
    started = time.perf_counter()

    manifest = convert_jsonl_to_parquet(
        args.input,
        args.output,
        args.raw_manifest,
        logger=logger,
        events_per_part=args.events_per_part,
        heartbeat_seconds=args.heartbeat_seconds,
        max_sessions=args.max_sessions,
    )

    elapsed = round(time.perf_counter() - started, 3)

    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"total_elapsed_seconds={elapsed}")
    print(
        "OTTO_CONVERSION_PASSED "
        f"status={manifest.status}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
