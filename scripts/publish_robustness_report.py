"""Rebuild the notebook comparison from verified replication evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from otto_recsys.research.robustness_report import comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = comparison(root)
    destination = root / "reports/robustness/comparison.json"
    payload = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.check:
        if destination.read_text() != payload:
            raise ValueError("published robustness comparison differs from verified evidence")
    else:
        destination.write_text(payload)
    print(json.dumps({k: data[k] for k in ("verified_cells", "planned_cells", "verified_windows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
