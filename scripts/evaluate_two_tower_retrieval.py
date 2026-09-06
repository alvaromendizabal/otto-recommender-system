from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluate_incremental_recall import resource_preflight

from otto_recsys.cloud.comparison_checkpoints import S3ComparisonCheckpoints
from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.neural_evaluation import evaluate_neural_retrieval, write_json
from otto_recsys.runtime import Heartbeat


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved neural predictions with frozen retrieval"
    )
    for name in ("ranking-cache", "predictions", "covisit-dir", "vectors", "index", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-k", type=int, default=1200)
    parser.add_argument("--ann-k", type=int, default=800)
    parser.add_argument("--ef-search", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--publish-report", action="store_true")
    parser.add_argument("--checkpoint-uri", help="S3 prefix for recovery on another workspace")
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resource_preflight(args.output_dir)
    logger = configure_logging("two_tower_comparison", log_dir=args.output_dir / "logs")
    checkpoint_store = (
        S3ComparisonCheckpoints(args.checkpoint_uri, region=args.region, logger=logger)
        if args.checkpoint_uri
        else None
    )
    started = time.perf_counter()
    try:
        with Heartbeat(logger, stage="comparison_total", interval_seconds=args.heartbeat_seconds):
            result = evaluate_neural_retrieval(
                args.ranking_cache,
                args.predictions,
                args.covisit_dir,
                args.vectors,
                args.index,
                args.output_dir,
                logger=logger,
                source_k=args.source_k,
                ann_k=args.ann_k,
                ef_search=args.ef_search,
                threads=args.threads,
                memory_limit=args.memory_limit,
                heartbeat_seconds=args.heartbeat_seconds,
                checkpoint_store=checkpoint_store,
            )
        if args.publish_report:
            write_json(
                Path(f"reports/metrics/two_tower_fold{result['validation_fold']}_retrieval.json"),
                result,
            )
        print(json.dumps(result, indent=2))
        print("OTTO_TWO_TOWER_RETRIEVAL_EVALUATION_PASSED", flush=True)
        return 0
    except Exception:
        logger.exception("comparison_failed")
        raise
    finally:
        logger.info(
            "comparison_complete",
            extra={"elapsed_seconds": round(time.perf_counter() - started, 3)},
        )
        if checkpoint_store is not None:
            try:
                checkpoint_store.publish_logs(args.output_dir)
            except Exception:
                logger.exception("comparison_log_upload_failed")


if __name__ == "__main__":
    raise SystemExit(main())
