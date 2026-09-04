from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.validation.kaggle_split import build_validation


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--processed-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)

    args = parser.parse_args()

    summary = json.loads(
        args.processed_summary.read_text(encoding="utf-8")
    )

    if not isinstance(summary, dict):
        raise ValueError("processed summary must contain a JSON object")

    max_ts = summary.get("max_ts")

    if not isinstance(max_ts, int):
        raise ValueError("processed summary has no integer max_ts")

    logger = configure_logging("build_validation")
    started = time.perf_counter()

    manifest = build_validation(
        args.source,
        args.raw_manifest,
        args.output,
        max_ts=max_ts,
        days=args.days,
        seed=args.seed,
        logger=logger,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    elapsed = round(time.perf_counter() - started, 3)

    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"total_elapsed_seconds={elapsed}")
    print(
        "OTTO_VALIDATION_PASSED "
        f"manifest_id={manifest.manifest_id}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
