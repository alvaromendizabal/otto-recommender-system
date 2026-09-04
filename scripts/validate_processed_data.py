from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.data.processed_validation import (
    validate_processed_dataset,
)
from otto_recsys.logging_utils import configure_logging


def write_summary_atomic(
    summary: dict[str, object],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, destination)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--expected-status",
        choices=("running", "partial", "complete"),
    )
    parser.add_argument("--summary-output", type=Path)

    args = parser.parse_args()

    logger = configure_logging("validate_processed_data")
    started = time.perf_counter()

    summary = validate_processed_dataset(
        args.root,
        logger=logger,
        heartbeat_seconds=args.heartbeat_seconds,
        expected_status=args.expected_status,
    )

    payload = asdict(summary)
    elapsed = round(time.perf_counter() - started, 3)

    if args.summary_output is not None:
        write_summary_atomic(payload, args.summary_output)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"total_elapsed_seconds={elapsed}")
    print("OTTO_PROCESSED_VALIDATION_PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
