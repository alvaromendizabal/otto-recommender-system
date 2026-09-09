"""Prepare a guarded launch for one predeclared temporal replication or its audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_recsys.research.robustness import (
    cells,
    load_plan,
    seed_launch,
    verification_launch,
    verify_seed_launch,
    verify_verification_launch,
)
from otto_recsys.research.temporal_robustness import (
    verify_window_launch,
    verify_window_verification,
    window_launch,
    window_verification_launch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--cell")
    action.add_argument("--verify-launch", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify_launch:
        training = json.loads(args.verify_launch.read_text())
        earlier = training.get("task") == "window_study"
        builder = window_verification_launch if earlier else verification_launch
        launch = builder(
            root,
            training,
            source_commit=args.source_commit,
            source_sha256=args.source_sha256,
        )
        (verify_window_verification if earlier else verify_verification_launch)(root, launch)
    else:
        plan, _ = load_plan(root)
        matches = [cell for cell in cells(plan) if cell["cell_id"] == args.cell]
        if len(matches) != 1:
            parser.error("--cell must name a cell in configs/robustness.toml")
        earlier = matches[0]["window"] != plan["reference_window"]
        launch = (window_launch if earlier else seed_launch)(
            root, args.cell, source_commit=args.source_commit, source_sha256=args.source_sha256
        )
        (verify_window_launch if earlier else verify_seed_launch)(root, launch)
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
