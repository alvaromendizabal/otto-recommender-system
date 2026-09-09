"""Audit a completed frozen replication and its native predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.robustness_audit import verify_replication
from otto_recsys.research.window_audit import verify_window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="original Parquet directory for earlier windows"
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    launch = json.loads(args.launch.read_text())
    logger = configure_logging("robustness_verification", log_dir=args.root / "logs")
    if launch.get("task") == "window_study":
        if args.source is None:
            parser.error("--source is required to reconstruct earlier-window events")
        result = verify_window(repository, args.root, args.source, launch, logger=logger)
    else:
        result = verify_replication(repository, args.root, launch, logger=logger)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cell_id": result["cell_id"],
                "audit_id": result["audit_id"],
                "sessions": result["statistics"]["sessions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
