"""Show published OTTO evidence and the next modeling task without starting jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import utc_now_iso


def read_report(root: Path, name: str) -> dict[str, Any] | None:
    path = root / "reports/metrics" / name
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Expected a report object: {name}")
    return report


def project_status(root: Path) -> dict[str, Any]:
    """Use versioned evidence, so status works without local data or pointers."""
    training = read_report(root, "two_tower_fold0_training.json")
    exact = read_report(root, "two_tower_fold0_retrieval.json")
    ann = read_report(root, "two_tower_fold0_ann.json")
    comparison = read_report(root, "two_tower_fold0_ann_comparison.json")
    audit = read_report(root, "two_tower_fold0_ann_comparison_audit.json")
    completed = False
    if comparison is not None:
        if (
            audit is None
            or ann is None
            or comparison.get("status") != "passed"
            or audit.get("status") != "passed"
            or comparison.get("input_id") != audit.get("input_id")
            or comparison.get("prediction_input_id") != ann.get("input_id")
            or sha256_file(root / "reports/metrics/two_tower_fold0_ann_comparison.json")
            != audit.get("metrics_sha256")
        ):
            raise ValueError("Published ANN comparison evidence does not match its audit")
        completed = True
    result: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "scope": "Published evidence; live AWS jobs are not queried.",
        "fold0_training": training.get("status", "unknown") if training else "pending",
        "exact_comparison": exact.get("status", "unknown") if exact else "pending",
        "ann_benchmark": ann.get("status", "unknown") if ann else "pending",
        "ann_comparison": "passed" if completed else "pending",
        "next_task": "Implement nested validation and the candidate/feature pipeline for ranking."
        if completed
        else "Complete and audit the frozen-baseline ANN comparison.",
        "ranking_evaluation": "not yet measured",
        "kaggle_submission": "not yet generated",
        "paid_compute_started": False,
    }
    if completed and comparison is not None:
        point = next(row for row in comparison["points"] if row["neural_k"] == 800)
        result["ann_comparison_run_id"] = comparison["input_id"]
        result["ann_prediction_run_id"] = comparison["prediction_input_id"]
        result["sessions"] = comparison["sessions"]
        result["candidate_ceiling_k800"] = {
            "base": point["weighted_base_ceiling"],
            "union": point["weighted_union_ceiling"],
            "gain": point["weighted_incremental_ceiling"],
            "gain_ci95": point["weighted_incremental_ci95"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    result = project_status(args.root)
    git = subprocess.run(
        ["git", "-C", str(args.root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    result["commit"] = git.stdout.strip() if git.returncode == 0 else None
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['timestamp']}] OTTO PROJECT STATUS commit={result['commit']}")
        for key in (
            "fold0_training",
            "exact_comparison",
            "ann_benchmark",
            "ann_comparison",
            "ranking_evaluation",
            "kaggle_submission",
        ):
            print(f"{key}={result[key]}")
        if "candidate_ceiling_k800" in result:
            value = result["candidate_ceiling_k800"]
            print(
                f"ANN candidate coverage: {value['base']:.3%} -> {value['union']:.3%}; "
                f"gain={100 * value['gain']:.3f} percentage points"
            )
        print(f"Next: {result['next_task']}")
        print(result["scope"])
        print(f"OTTO_PROJECT_STATUS_COMPLETE elapsed_seconds={result['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
