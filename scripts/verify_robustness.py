"""Audit a completed frozen reference-window replication and its native predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.robustness_audit import verify_replication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    result = verify_replication(
        Path(__file__).resolve().parents[1],
        args.root,
        json.loads(args.launch.read_text()),
        logger=configure_logging("robustness_verification", log_dir=args.root / "logs"),
    )
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
