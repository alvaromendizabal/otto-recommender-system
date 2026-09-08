"""Audit the frozen temporal study against original events and native models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.audit import audit_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--source", type=Path, default=Path("data/source_audit/processed/train"))
    parser.add_argument("--sessions", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=12)
    args = parser.parse_args()
    report = audit_study(
        args.root,
        args.source,
        sessions=args.sessions,
        threads=args.threads,
        memory_gib=args.memory_gib,
        logger=configure_logging("research_audit", log_dir=args.root / "logs"),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
