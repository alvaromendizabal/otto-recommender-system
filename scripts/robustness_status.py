"""Show the managed OTTO queue; watching never starts or changes a cloud job."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL = {"Succeeded", "Failed", "Stopped"}


def snapshot(client: Any, arn: str, existing_job: str) -> dict[str, Any]:
    execution = client.describe_pipeline_execution(PipelineExecutionArn=arn)
    pages = client.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=arn
    )
    steps = [step for page in pages for step in page["PipelineExecutionSteps"]]
    previous = client.describe_processing_job(ProcessingJobName=existing_job)
    return {"execution": execution, "steps": {"PipelineExecutionSteps": steps},
            "existing_job": {k: previous.get(k) for k in (
                "ProcessingJobName", "ProcessingJobStatus", "FailureReason"
            )}}


def rows(plan: dict[str, Any], observed: dict[str, Any]) -> list[dict[str, str]]:
    actual = {s["StepName"]: s for s in observed["steps"]["PipelineExecutionSteps"]}
    result = []
    for planned in plan["jobs"]:
        step = actual.get(planned["step"], {})
        # SageMaker adds an execution suffix. Always use its observed ARN.
        arn = step.get("Metadata", {}).get("ProcessingJob", {}).get("Arn", "")
        result.append({
            "cell": planned["cell_id"], "phase": planned["phase"],
            "status": step.get("StepStatus", "Waiting"),
            "job": arn.rsplit("/", 1)[-1],
            "failure": step.get("FailureReason", ""),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--snapshot", type=Path, help="Render a saved API observation offline")
    args = parser.parse_args()
    if not 15 <= args.interval <= 60:
        parser.error("--interval must be between 15 and 60 seconds")
    if args.snapshot and args.watch:
        parser.error("a saved snapshot cannot be watched as live cloud state")
    root = Path(__file__).resolve().parents[1] / "reports/robustness/batch"
    plan = json.loads((root / "plan.json").read_text())
    receipt = json.loads((root / "execution.json").read_text())
    arn = receipt["started"]["PipelineExecutionArn"]
    client = None
    if not args.snapshot:
        sdk = importlib.import_module("boto3")
        configuration = importlib.import_module("botocore.config").Config(
            connect_timeout=5, read_timeout=20,
            retries={"mode": "standard", "total_max_attempts": 3},
        )
        client = sdk.client("sagemaker", region_name=plan["region"], config=configuration)
    start = time.monotonic()
    while True:
        if args.snapshot:
            saved = json.loads(args.snapshot.read_text())
            observed = saved.get("observation", saved)
        else:
            observed = snapshot(client, arn, plan["existing_job"])
        state = observed["execution"]["PipelineExecutionStatus"]
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        print(f"\n{timestamp} pipeline={state} monitor_elapsed={time.monotonic()-start:.0f}s")
        if args.snapshot:
            print("Saved observation; this is not a live cloud refresh.")
        prior = observed.get("existing_job")
        if prior:
            print(f"early_seed_20260909 train {prior['ProcessingJobStatus']}")
        for row in rows(plan, observed):
            print(f"{row['cell']:<23} {row['phase']:<5} {row['status']:<10} {row['job']}")
            if row["failure"]:
                print(f"  failure={row['failure']}")
        print("Cloud completion and publication of audited results are separate.", flush=True)
        if not args.watch or state in TERMINAL:
            return 1 if state in {"Failed", "Stopped"} else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
