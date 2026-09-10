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
from otto_recsys.ranking.reporting import validate_report


def read_report(root: Path, name: str) -> dict[str, Any] | None:
    path = root / "reports/metrics" / name
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Expected a report object: {name}")
    return report


def research_status(root: Path) -> dict[str, Any]:
    directory = root / "reports/research"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return {"controlled_research": "pending", "competition_prediction": "pending"}
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.name != name or not path.is_file() or sha256_file(path) != expected:
            raise ValueError("Published controlled research checksum mismatch")
    evaluation = json.loads((directory / "evaluation.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    if (
        manifest["status"] != evaluation["status"]
        or audit["status"] != "passed"
        or evaluation["status"] != "passed"
        or manifest["seal_id"] != evaluation["seal_id"]
        or audit["seal_id"] != manifest["seal_id"]
        or audit["evaluation_report_sha256"] != sha256_file(directory / "evaluation.json")
    ):
        raise ValueError("Published controlled research does not match its audit")
    result: dict[str, Any] = {
        "controlled_research": "passed",
        "research_evaluation_sessions": evaluation["sessions"],
        "research_weighted_recall_at_20": evaluation["scores"]["selected"]["weighted_recall_at_20"],
        "research_features": manifest["features"],
        "competition_prediction": "pending",
        "notebook_publication": "pending",
        "next_task": "Complete the full inference notebook and publish its verified replay bundle.",
    }
    replay_path = directory / "inference_replay/manifest.json"
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text())
        if replay["status"] != "passed" or replay["seal_id"] != manifest["seal_id"]:
            raise ValueError("Published inference replay has a different model seal")
        for name, expected in replay["files"].items():
            path = replay_path.parent / name
            if path.name != name or not path.is_file() or sha256_file(path) != expected:
                raise ValueError("Published inference replay checksum mismatch")
        full = replay["full_prediction"]
        if full["sessions"] <= 0 or full["rows"] != full["sessions"] * 3:
            raise ValueError("Published competition prediction coverage is incomplete")
        result.update(
            competition_prediction="format and coverage validated",
            prediction_sessions=full["sessions"],
            prediction_rows=full["rows"],
            prediction_sha256=full["sha256"],
            kaggle_submission="local format validated; no Kaggle upload or score claimed",
            next_task="Publish verified canonical notebook outputs and execution receipts.",
        )
    receipts = root / "notebooks/execution.json"
    if receipts.is_file():
        record = json.loads(receipts.read_text())
        notebooks = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
        if record["status"] != "passed" or record["notebooks"] != len(notebooks):
            raise ValueError("Published notebook inventory is incomplete")
        if {r["notebook"] for r in record["receipts"]} != {p.name for p in notebooks}:
            raise ValueError("Published notebook receipts do not cover the canonical inventory")
        for receipt in record["receipts"]:
            if sha256_file(root / "notebooks" / receipt["notebook"]) != receipt["sha256"]:
                raise ValueError("Published notebook bytes differ from the execution receipt")
        result["notebook_publication"] = "passed"
        if result["competition_prediction"] != "pending":
            result["next_task"] = (
                "Completed research and batch inference release. Review notebooks 09 and 10; "
                "additional seeds/cohorts or online evaluation are optional new studies."
            )
    return result


def project_status(root: Path) -> dict[str, Any]:
    """Use versioned evidence, so status works without local data or pointers."""
    training = read_report(root, "two_tower_fold0_training.json")
    exact = read_report(root, "two_tower_fold0_retrieval.json")
    ann = read_report(root, "two_tower_fold0_ann.json")
    comparison = read_report(root, "two_tower_fold0_ann_comparison.json")
    audit = read_report(root, "two_tower_fold0_ann_comparison_audit.json")
    features = read_report(root, "ranking_features.json")
    feature_audit = read_report(root, "ranking_features_audit.json")
    ranking = read_report(root, "ranking_evaluation.json")
    features_complete = False
    if features is not None:
        if (
            feature_audit is None
            or features.get("status") != "passed"
            or feature_audit.get("status") != "passed"
            or features.get("input_id") != feature_audit.get("input_id")
            or features.get("parts_sha256") != feature_audit.get("parts_sha256")
            or features.get("contract_sha256") != feature_audit.get("feature_contract_sha256")
            or any(feature_audit.get("mismatches", {"missing": 1}).values())
            or sha256_file(root / "reports/metrics/ranking_feature_contract.json")
            != features.get("contract_sha256")
        ):
            raise ValueError("Published ranking feature evidence does not match its audit")
        features_complete = True
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
        "paid_compute_started_by_this_command": False,
    }
    result["observed_ranking_features"] = "passed" if features_complete else "pending"
    if features_complete:
        result["next_task"] = (
            "Run scripts/run_ranking.py to materialize candidates and train/evaluate "
            "LambdaRank under exploratory nested validation; see docs/RANKING.md."
        )
    if ranking is not None:
        validate_report(ranking)
        result.update(
            ranking_evaluation="measured (exploratory)",
            ranking_run_id=ranking["run_id"],
            ranking_weighted_recall_at_20=ranking["learned"]["weighted_recall_at_20"],
            ranking_baseline_recall_at_20=ranking["baseline"]["weighted_recall_at_20"],
            ranking_validation_scope=ranking["validation_scope"],
            next_task=(
                "Compare ablations and certified neural sources under nested validation; "
                "then implement full-test prediction and validate a Kaggle submission."
            ),
        )
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
    result.update(research_status(root))
    comparison_path = root / "reports/robustness/comparison.json"
    if comparison_path.is_file():
        from otto_recsys.research.robustness_report import comparison as verified_comparison

        verified = verified_comparison(root)
        if json.loads(comparison_path.read_text()) != verified:
            raise ValueError("Published temporal comparison differs from audited evidence")
        result["temporal_validation"] = (
            f"{verified['verified_cells']}/{verified['planned_cells']} independently audited"
        )
        result["next_task"] = (
            "Submit the verified official-prefix prediction and record its Kaggle result."
            if verified["verified_cells"] == verified["planned_cells"] else
            "Complete the remaining temporal audits and publish their verified results."
        )
    submission_path = root / "reports/submissions/kaggle_submission.json"
    if submission_path.is_file():
        submission = json.loads(submission_path.read_text())
        from otto_recsys.research.robustness import fingerprint

        if submission["receipt_id"] != fingerprint(
            {k: v for k, v in submission.items() if k != "receipt_id"}
        ):
            raise ValueError("Kaggle receipt checksum mismatch")
        if (submission.get("valid_competition_evaluation") is False
                and result.get("prediction_sha256") == submission["prediction"]["sha256"]):
            result.update(
                competition_prediction="invalidated: full test sessions included future events",
                kaggle_submission="initial score invalidated; official-prefix replacement pending",
                next_task=(
                    "Finish the queued temporal audits and validate the replacement inference "
                    "before submitting the corrected file. "
                    "The 50-file collection and release remain."
                ),
            )
        elif submission.get("valid_competition_input") is True:
            for name, expected in submission["source_files"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root.resolve()) or sha256_file(path) != expected:
                    raise ValueError("Kaggle receipt source checksum mismatch")
            if result.get("prediction_sha256") != submission["prediction"]["sha256"]:
                raise ValueError("Kaggle receipt and prediction differ")
            contract = json.loads(
                (root / "reports/research/competition_prediction_contract.json").read_text()
            )
            attestation = json.loads(
                (root / "reports/submissions/competition_input.json").read_text()
            )
            if contract["competition_input"]["raw_sha256"] != attestation["raw_sha256"]:
                raise ValueError("Kaggle receipt does not use the attested official input")
            result["competition_prediction"] = (
                "official input, full file and native replay verified"
            )
            if (submission["status"] == "complete"
                    and submission.get("valid_competition_evaluation")):
                result.update(kaggle_submission="complete (after deadline)",
                              kaggle_scores=submission["displayed_scores"],
                              next_task="Review the published results and release scope.")
            else:
                if any(v is not None for v in submission["displayed_scores"].values()):
                    raise ValueError("Unsubmitted predictions cannot have a Kaggle score")
                result.update(
                    kaggle_submission="file uploaded; final Submit action awaiting confirmation",
                    next_task="Confirm the prepared Kaggle submission; no training remains.",
                )
    from otto_recsys.research.feature_gate import feature_gate

    result.update(feature_gate(root))
    if result["feature_research_gate"] == "open":
        result["next_task"] = result["feature_gate_next_task"]
    for experiment in ("shared_feature", "task_feature", "domain_feature", "graph_feature"):
        development_path = root / f"reports/research/{experiment}_run.json"
        if not development_path.is_file():
            continue
        development = json.loads(development_path.read_text())
        result["latest_development_observation"] = {
            "experiment": development.get("experiment", experiment),
            "observed": development["observed"],
            "scope": "Saved observation only; use retrieval_status.py for live AWS state.",
        }
        break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-feature-gate", action="store_true",
        help="Exit 2 unless feature coverage and temporal confirmation are complete.",
    )
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
            "observed_ranking_features",
            "ranking_evaluation",
            "kaggle_submission",
            "controlled_research",
            "competition_prediction",
            "temporal_validation",
            "feature_research_gate",
            "final_training_ready",
        ):
            if key in result:
                print(f"{key}={result[key]}")
        if "ranking_weighted_recall_at_20" in result:
            print(
                f"Ranked weighted Recall@20: {result['ranking_weighted_recall_at_20']:.6f}; "
                f"matched baseline={result['ranking_baseline_recall_at_20']:.6f}"
            )
        if "research_weighted_recall_at_20" in result:
            print(
                f"Controlled weighted Recall@20: {result['research_weighted_recall_at_20']:.6f}; "
                f"evaluation sessions={result['research_evaluation_sessions']:,}"
            )
        if "candidate_ceiling_k800" in result:
            value = result["candidate_ceiling_k800"]
            print(
                f"ANN candidate coverage: {value['base']:.3%} -> {value['union']:.3%}; "
                f"gain={100 * value['gain']:.3f} percentage points"
            )
        if "latest_development_observation" in result:
            development = result["latest_development_observation"]
            observed = development["observed"]
            print(
                f"Latest saved development observation: {development['experiment']} "
                f"job={observed['ProcessingJobName']} status={observed['ProcessingJobStatus']}"
            )
        print(f"Next: {result['next_task']}")
        print(result["scope"])
        print(f"OTTO_PROJECT_STATUS_COMPLETE elapsed_seconds={result['elapsed_seconds']:.3f}")
    return 2 if args.require_feature_gate and not result["final_training_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
