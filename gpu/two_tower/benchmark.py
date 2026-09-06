"""Run the durable ANN benchmark and preserve attempt diagnostics."""

from __future__ import annotations

import json
import time

from otto_two_tower.ann_cli import parse_args
from otto_two_tower.logging_utils import configure_logging, utc_now_iso


def main() -> int:
    args = parse_args()
    from otto_two_tower.ann_benchmark import run_benchmark
    from otto_two_tower.benchmark_artifacts import BenchmarkArtifacts
    from otto_two_tower.telemetry import TrainingHeartbeat

    logger = configure_logging("two_tower_ann", args.output_dir / "logs")
    progress = {"stage": "initialization"}
    started = time.perf_counter()
    status = "failed"
    artifacts = BenchmarkArtifacts(
        args.output_dir, args.run_id, logger, uri=args.checkpoint_uri, region=args.region
    )
    log_path = args.output_dir / "logs/two_tower_ann.jsonl"
    try:
        with TrainingHeartbeat(
            logger,
            stage="ann_benchmark",
            interval_seconds=args.heartbeat_seconds,
            progress_provider=lambda: dict(progress),
        ):
            result = run_benchmark(args, artifacts, logger, progress)
        status = "passed"
        logger.info(
            "OTTO_TWO_TOWER_ANN_BENCHMARK_PASSED",
            extra={
                "reference_weighted_recall_at_20": result["full_reference_ranking"][
                    "weighted_recall_at_20"
                ],
                "ann_weighted_recall_at_20": (result["full_ann_ranking"] or {}).get(
                    "weighted_recall_at_20"
                ),
                "selected_nprobe": result["selected_nprobe"],
                "confirmation_fidelity_passed": result["confirmation_fidelity_passed"],
            },
        )
        return 0
    except Exception:
        logger.exception("ann_benchmark_failed")
        raise
    finally:
        logger.info(
            "ann_attempt_complete",
            extra={"status": status, "elapsed_seconds": round(time.perf_counter() - started, 3)},
        )
        # Attempt logs have unique names, so a retry never overwrites earlier diagnostics.
        name = utc_now_iso().replace(":", "-")
        try:
            artifacts.upload_log(log_path, f"logs/attempt-{name}.jsonl")
        except Exception:
            logger.exception("attempt_log_upload_failed")
            if status == "passed":
                raise


def entrypoint() -> int:
    """Record bootstrap failures even before heavy imports or S3 initialization."""
    started, exit_code = time.perf_counter(), 1
    print(json.dumps({"timestamp": utc_now_iso(), "message": "ann_worker_start"}), flush=True)
    try:
        exit_code = main()
        return exit_code
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
        raise
    finally:
        print(
            json.dumps(
                {
                    "timestamp": utc_now_iso(),
                    "message": "ann_worker_complete",
                    "exit_code": exit_code,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(entrypoint())
