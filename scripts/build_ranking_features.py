"""Prepare durable observed features and explicit exploratory nested split metadata."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from otto_recsys.cloud.ranking_checkpoints import S3FeatureCheckpoints
from otto_recsys.logging_utils import configure_logging
from otto_recsys.ranking.feature_cache import build_feature_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking-cache", type=Path, default=Path("data/interim/ranking_training_cache")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/ranking_features"))
    parser.add_argument("--checkpoint-uri", required=True, help="Durable S3 project prefix")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--inner-seed", type=int, default=20260907)
    parser.add_argument("--heartbeat-seconds", type=float, default=15)
    args = parser.parse_args()
    logger = configure_logging("ranking_features", log_dir=args.output_dir / "logs")
    checkpoints = S3FeatureCheckpoints(args.checkpoint_uri, region=args.region, logger=logger)
    started = time.perf_counter()
    failed = False
    try:
        result = build_feature_cache(
            args.ranking_cache,
            args.output_dir,
            logger=logger,
            inner_seed=args.inner_seed,
            heartbeat_seconds=args.heartbeat_seconds,
            checkpoints=checkpoints,
        )
        print(json.dumps(result, indent=2))
        print("OTTO_RANKING_FEATURES_PASSED", flush=True)
        return 0
    except Exception:
        failed = True
        logger.exception("ranking_features_failed", extra={"status": "failed"})
        raise
    finally:
        logger.info(
            "ranking_features_attempt_complete",
            extra={
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )
        try:
            checkpoints.publish_logs(args.output_dir)
        except Exception:
            if not failed:
                raise
            logger.exception("ranking_failure_log_upload_failed")


if __name__ == "__main__":
    raise SystemExit(main())
