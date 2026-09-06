"""Audit the downloaded ANN report and count checkpoints without paid compute."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.ann_audit import audit_ann
from otto_recsys.runtime import Heartbeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    logger = configure_logging("two_tower_ann_audit", log_dir=args.benchmark_dir / "audit_logs")
    started, status = time.perf_counter(), "failed"
    try:
        with Heartbeat(logger, stage="ann_audit", interval_seconds=15):
            result = audit_ann(
                args.benchmark_dir, expected_run_id=args.expected_run_id, logger=logger
            )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.report)
        status = "passed"
        print("OTTO_TWO_TOWER_ANN_AUDIT_PASSED", flush=True)
        return 0
    except Exception:
        logger.exception("ann_audit_failed")
        raise
    finally:
        logger.info(
            "ann_audit_complete",
            extra={"status": status, "elapsed_seconds": round(time.perf_counter() - started, 3)},
        )


if __name__ == "__main__":
    raise SystemExit(main())
