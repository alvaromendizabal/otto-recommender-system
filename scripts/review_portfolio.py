"""Read and verify the compact public review with Python alone; no cloud access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from otto_recsys.portfolio_review import review_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    snapshot = review_snapshot(args.root)
    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        release = snapshot["verified_release"]
        controlled = snapshot["controlled_study"]
        diagnostic = snapshot["candidate_diagnostic"]
        print(f"OTTO PUBLIC REVIEW — evidence as of {snapshot['as_of_utc']}")
        print(f"Release: {release['private_score']:.5f} private / "
              f"{release['public_score']:.5f} public; {release['evaluation_setting']}.")
        print(f"Delivery: {release['sessions']:,} sessions; {release['rows']:,} rows.")
        print(f"Offline study: {controlled['weighted_recall_at_20']['selected']:.6f}; "
              f"{controlled['sessions']:,} sessions; "
              f"+{100 * controlled['selected_minus_core']:.3f} percentage points over compact.")
        print(f"Later challenger: {snapshot['research_status']}.")
        print(f"Separate cart diagnostic: {diagnostic['ranked_hits']:,} ranked hits; "
              f"{diagnostic['coverage_hits']:,} candidate ceiling; "
              f"{diagnostic['denominator']:,} capped targets.")
        print(f"Archived research question ({snapshot['as_of_utc']}): "
              f"{snapshot['next_research_question']}.")
        print("OTTO_PUBLIC_REVIEW_PASSED — recorded evidence checked; no new model or score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
