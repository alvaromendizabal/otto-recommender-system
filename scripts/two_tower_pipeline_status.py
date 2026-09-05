from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_command(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {rendered}\n{detail}"
        )
    return completed


def aws_json(arguments: list[str]) -> dict[str, Any]:
    completed = run_command(["aws", *arguments, "--output", "json"], check=True)
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("AWS CLI returned a non-object JSON payload")
    return payload


def get_s3_json(bucket: str, key: str) -> dict[str, Any] | None:
    completed = run_command(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", "-", "--only-show-errors"]
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, dict) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("OTTO_BUCKET"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise RuntimeError("--bucket or OTTO_BUCKET is required")
    latest_key = "retrieval/two-tower/pipelines/resume-proof/latest.json"
    latest = get_s3_json(args.bucket, latest_key)
    if latest is None:
        print("OTTO_TWO_TOWER_PIPELINE_NOT_REGISTERED")
        return 1
    pipeline_name = str(latest["pipeline_name"])
    run_id = str(latest["run_id"])
    payload = aws_json(
        [
            "sagemaker",
            "list-pipeline-executions",
            "--pipeline-name",
            pipeline_name,
            "--max-results",
            "10",
        ]
    )
    summaries = payload.get("PipelineExecutionSummaries", [])
    summary = summaries[0] if isinstance(summaries, list) and summaries else None

    print("=" * 72)
    print("OTTO TWO-TOWER MANAGED PIPELINE STATUS")
    print("=" * 72)
    print(f"checked_at={utc_now()}")
    print(f"pipeline_name={pipeline_name}")
    print(f"run_id={run_id}")
    print(f"code_commit={latest.get('code_commit')}")
    if isinstance(summary, dict):
        print(f"execution_status={summary.get('PipelineExecutionStatus')}")
        print(f"execution_arn={summary.get('PipelineExecutionArn')}")
    else:
        print("execution_status=NOT_STARTED")

    proof_key = (
        f"retrieval/two-tower/runs/resume-proof/{run_id}/"
        "checkpoints/resume_proof.json"
    )
    proof = get_s3_json(args.bucket, proof_key)
    if proof is None:
        print("resume_proof=NOT_AVAILABLE_YET")
    else:
        print(f"resume_proof={proof.get('status')}")
        print(f"resumed_from_step={proof.get('resumed_from_step')}")
        print(f"final_step={proof.get('final_step')}")
        print(f"advanced_steps={proof.get('advanced_steps')}")

    if isinstance(summary, dict) and summary.get("PipelineExecutionStatus") == "Succeeded":
        if proof is None or proof.get("status") != "passed":
            print("OTTO_TWO_TOWER_PIPELINE_INCONSISTENT")
            return 2
        print("OTTO_TWO_TOWER_MANAGED_RESUME_PROOF_PASSED")
        return 0
    if isinstance(summary, dict) and summary.get("PipelineExecutionStatus") == "Failed":
        print("OTTO_TWO_TOWER_PIPELINE_FAILED")
        return 3
    print("OTTO_TWO_TOWER_PIPELINE_IN_PROGRESS_OR_NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
