from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.data.processed_validation import (
    validate_processed_dataset,
)
from otto_recsys.logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    logger = configure_logging("validate_processed_data")
    started = time.perf_counter()

    summary = validate_processed_dataset(
        args.root,
        logger=logger,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    elapsed = round(time.perf_counter() - started, 3)

    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(f"total_elapsed_seconds={elapsed}")
    print("OTTO_PROCESSED_VALIDATION_PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
