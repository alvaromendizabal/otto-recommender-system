"""Prepare a guarded launch for one predeclared reference-window replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_recsys.research.robustness import seed_launch, verify_seed_launch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    launch = seed_launch(
        root, args.cell, source_commit=args.source_commit, source_sha256=args.source_sha256
    )
    verify_seed_launch(root, launch)
    payload = json.dumps(launch, sort_keys=True, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise ValueError("launch output already contains a different immutable replication")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(payload)
    temporary.replace(args.output)
    print(json.dumps({"status": "prepared", **launch["robustness"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
