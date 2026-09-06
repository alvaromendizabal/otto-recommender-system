"""Audit a downloaded comparison using only the saved count checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.comparison_audit import audit_comparison
from otto_recsys.runtime import Heartbeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--expected-input-id", required=True)
    parser.add_argument("--buckets", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    logger = configure_logging("two_tower_comparison_audit", log_dir=args.comparison_dir / "logs")
    started = time.perf_counter()
    status = "failed"
    try:
        with Heartbeat(logger, stage="comparison_audit", interval_seconds=15):
            result = audit_comparison(
                args.comparison_dir,
                expected_input_id=args.expected_input_id,
                buckets=args.buckets,
                logger=logger,
            )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.report)
        status = "passed"
        print("OTTO_TWO_TOWER_COMPARISON_AUDIT_PASSED", flush=True)
        return 0
    except Exception:
        logger.exception("comparison_audit_failed")
        raise
    finally:
        logger.info(
            "comparison_audit_complete",
            extra={
                "status": status,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
