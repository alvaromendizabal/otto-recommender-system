"""Audit downloaded graph-study artifacts without starting training or inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.graph_feature_audit import audit
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger = configure_logging("graph_feature_audit")
    with Heartbeat(logger, stage="graph_feature_audit", interval_seconds=15):
        result = audit(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    logger.info("graph_feature_audit_passed", extra={"sessions": result["sessions_per_arm"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
