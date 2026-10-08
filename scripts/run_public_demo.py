#!/usr/bin/env python3
"""Run the public synthetic demo without installing the project or dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otto_recsys.public_demo import run_demo, verify_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/public_demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="Verify an existing completed demo")
    args = parser.parse_args()
    reused = False
    try:
        manifest = args.output / "manifest.json"
        if manifest.is_file() and not manifest.is_symlink():
            reused = json.loads(manifest.read_bytes()).get("status") == "completed"
        result = (
            verify_output(args.output, args.seed)
            if args.check
            else run_demo(args.output, args.seed)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "passed" if args.check else "completed",
                "reused": reused,
                "output_dir": str(args.output),
                "result_path": str(args.output / "result.json"),
                "manifest_path": str(args.output / "manifest.json"),
                "report_path": str(args.output / "report.html"),
                "weighted_recall_at_20": result["metrics"]["weighted_recall_at_20"],
                "synthetic": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
