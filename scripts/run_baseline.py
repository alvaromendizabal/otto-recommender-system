from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.baseline import (
    evaluate,
    fit_popularity,
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--top-popularity",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    logger = configure_logging("baseline")
    started = time.perf_counter()

    train_path = (
        args.validation_dir / "train_sessions.jsonl"
    )
    test_path = (
        args.validation_dir / "test_sessions.jsonl"
    )
    labels_path = (
        args.validation_dir / "test_labels.jsonl"
    )

    popularity = fit_popularity(
        train_path,
        logger=logger,
        top_n=args.top_popularity,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    popularity.write_json(
        args.output_dir / "popularity.json"
    )

    metrics = evaluate(
        test_path,
        labels_path,
        popularity,
        logger=logger,
        k=20,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    elapsed = round(
        time.perf_counter() - started,
        3,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )
    print(f"total_elapsed_seconds={elapsed}")
    print("OTTO_BASELINE_PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
