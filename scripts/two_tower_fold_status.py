from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def aws_json(arguments: list[str]) -> dict[str, Any]:
    completed = run(["aws", *arguments, "--output", "json"])
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("AWS CLI returned a non-object payload")
    return payload


def s3_json(bucket: str, key: str) -> dict[str, Any] | None:
    completed = run(["aws", "s3", "cp", f"s3://{bucket}/{key}", "-", "--only-show-errors"])
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, dict) else None


def s3_prefix_stats(uri: str) -> tuple[int, int]:
    completed = run(["aws", "s3", "ls", uri, "--recursive", "--summarize"])
    if completed.returncode != 0:
        return 0, 0
    objects = 0
    total_bytes = 0
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Objects:"):
            objects = int(stripped.split(":", maxsplit=1)[1].strip())
        elif stripped.startswith("Total Size:"):
            total_bytes = int(stripped.split(":", maxsplit=1)[1].strip())
    return objects, total_bytes


def training_logs(job_name: str, *, limit: int = 120) -> list[str]:
    streams = aws_json(
        [
            "logs",
            "describe-log-streams",
            "--log-group-name",
            "/aws/sagemaker/TrainingJobs",
            "--log-stream-name-prefix",
            f"{job_name}/",
        ]
    ).get("logStreams", [])
    output: list[str] = []
    if not isinstance(streams, list):
        return output
    for stream in streams:
        if not isinstance(stream, dict) or not stream.get("logStreamName"):
            continue
        payload = aws_json(
            [
                "logs",
                "get-log-events",
                "--log-group-name",
                "/aws/sagemaker/TrainingJobs",
                "--log-stream-name",
                str(stream["logStreamName"]),
                "--no-start-from-head",
                "--limit",
                str(limit),
            ]
        )
        events = payload.get("events", [])
        if isinstance(events, list):
            output.extend(
                str(event.get("message", "")) for event in events if isinstance(event, dict)
            )
    return output[-limit:]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("OTTO_BUCKET"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--show-logs", action="store_true")
    parser.add_argument("--publish-report", action="store_true")
    parser.add_argument("--region")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=21600.0)
    return parser.parse_args()


def report_status(args: argparse.Namespace) -> tuple[int, bool]:
    if not args.bucket:
        raise RuntimeError("--bucket or OTTO_BUCKET is required")
    if not 0 <= args.fold < 5:
        raise ValueError("--fold must be in [0, 4]")

    latest_key = f"retrieval/two-tower/pipelines/folds/fold-{args.fold}/latest.json"
    latest = s3_json(args.bucket, latest_key)
    if latest is None:
        print("OTTO_TWO_TOWER_FOLD_NOT_STARTED")
        return 0, True

    region = args.region or str(latest.get("region", "us-west-2"))
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region
    execution_arn = latest.get("pipeline_execution_arn")
    print("=" * 72)
    print("OTTO TWO-TOWER FOLD STATUS")
    print("=" * 72)
    print(f"checked_at={utc_now()}")
    print(f"validation_fold={latest.get('validation_fold')}")
    print(f"pipeline_name={latest.get('pipeline_name')}")
    print(f"run_id={latest.get('run_id')}")
    print(f"code_commit={latest.get('code_commit')}")

    if not execution_arn:
        print("execution_status=RegisteredNotStarted")
        print("OTTO_TWO_TOWER_FOLD_REGISTERED_NO_GPU_STARTED")
        return 0, True

    execution = aws_json(
        [
            "sagemaker",
            "describe-pipeline-execution",
            "--pipeline-execution-arn",
            str(execution_arn),
        ]
    )
    status = str(execution.get("PipelineExecutionStatus"))
    print(f"execution_status={status}")
    print(f"execution_id={str(execution_arn).rsplit('/', maxsplit=1)[-1]}")
    if execution.get("FailureReason"):
        print(f"pipeline_failure_reason={execution['FailureReason']}")

    steps_payload = aws_json(
        [
            "sagemaker",
            "list-pipeline-execution-steps",
            "--pipeline-execution-arn",
            str(execution_arn),
        ]
    )
    steps = steps_payload.get("PipelineExecutionSteps", [])
    training_job_name: str | None = None
    step_status: str | None = None
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_status = str(step.get("StepStatus"))
            print(
                "step="
                f"{step.get('StepName')} status={step_status} "
                f"started={step.get('StartTime')} ended={step.get('EndTime')}"
            )
            metadata = step.get("Metadata", {})
            if isinstance(metadata, dict):
                training_metadata = metadata.get("TrainingJob", {})
                if isinstance(training_metadata, dict) and training_metadata.get("Arn"):
                    training_job_name = str(training_metadata["Arn"]).rsplit("/", maxsplit=1)[-1]
            if step.get("FailureReason"):
                print(f"step_failure_reason={step['FailureReason']}")

    checkpoint_uri = str(latest.get("checkpoint_s3_uri", ""))
    objects, checkpoint_bytes = s3_prefix_stats(checkpoint_uri) if checkpoint_uri else (0, 0)
    print(f"checkpoint_objects={objects}")
    print(f"checkpoint_bytes={checkpoint_bytes}")

    training: dict[str, Any] | None = None
    if training_job_name:
        training = aws_json(
            [
                "sagemaker",
                "describe-training-job",
                "--training-job-name",
                training_job_name,
            ]
        )
        print(f"training_job={training_job_name}")
        print(f"training_status={training.get('TrainingJobStatus')}")
        print(f"secondary_status={training.get('SecondaryStatus')}")
        print(f"instance_type={training.get('ResourceConfig', {}).get('InstanceType')}")
        print(f"training_start={training.get('TrainingStartTime')}")
        print(f"training_end={training.get('TrainingEndTime')}")
        print(f"billable_seconds={training.get('BillableTimeInSeconds')}")
        if training.get("FailureReason"):
            print(f"training_failure_reason={training['FailureReason']}")

    run_id = str(latest.get("run_id"))
    manifest_key = (
        f"retrieval/two-tower/runs/folds/fold-{args.fold}/{run_id}/"
        "checkpoints/training_manifest.json"
    )
    training_manifest = s3_json(args.bucket, manifest_key)
    if training_manifest:
        print(f"global_step={training_manifest.get('global_step')}")
        print(f"best_valid_loss={training_manifest.get('best_valid_loss')}")
        print(f"training_elapsed_seconds={training_manifest.get('elapsed_seconds')}")
        history = training_manifest.get("history")
        print(f"completed_epochs={len(history) if isinstance(history, list) else 0}")

    if args.show_logs and training_job_name:
        print("--- recent_training_logs ---")
        for line in training_logs(training_job_name):
            print(line)
        print("--- end_recent_training_logs ---")

    if status == "Succeeded" and training_manifest is not None:
        report = {
            "schema_version": 1,
            "status": "passed",
            "validation_fold": args.fold,
            "run_id": run_id,
            "code_commit": latest.get("code_commit"),
            "pipeline_execution_id": str(execution_arn).rsplit("/", maxsplit=1)[-1],
            "instance_type": (
                training.get("ResourceConfig", {}).get("InstanceType")
                if training is not None
                else None
            ),
            "billable_seconds": (
                training.get("BillableTimeInSeconds") if training is not None else None
            ),
            "checkpoint_objects": objects,
            "checkpoint_bytes": checkpoint_bytes,
            "global_step": training_manifest.get("global_step"),
            "best_valid_loss": training_manifest.get("best_valid_loss"),
            "training_elapsed_seconds": training_manifest.get("elapsed_seconds"),
            "completed_epochs": (
                len(training_manifest.get("history", []))
                if isinstance(training_manifest.get("history"), list)
                else 0
            ),
            "input_id": training_manifest.get("input_id"),
            "validation_manifest_id": training_manifest.get("validation_manifest_id"),
            "recorded_at": utc_now(),
        }
        if args.publish_report:
            path = Path(f"reports/metrics/two_tower_fold{args.fold}_training.json")
            write_json_atomic(path, report)
            print(f"public_report={path}")
        print("OTTO_TWO_TOWER_FOLD_TRAINING_PASSED")
        return 0, True

    if status == "Succeeded":
        print("OTTO_TWO_TOWER_FOLD_REPORT_MISSING: training manifest has not been found")
        return 1, True
    if status in {"Failed", "Stopped"}:
        print(f"OTTO_TWO_TOWER_FOLD_{status.upper()}")
        return 1, True

    print("OTTO_TWO_TOWER_FOLD_IN_PROGRESS_OR_NOT_STARTED")
    return 0, False


def main() -> int:
    args = parse_args()
    if not 5 <= args.interval_seconds <= 60:
        raise ValueError("--interval-seconds must be between 5 and 60")
    if args.max_wait_seconds <= 0:
        raise ValueError("--max-wait-seconds must be positive")
    # Bootstrap the S3 read in the requested/default project region. The durable
    # pointer then supplies the training region unless explicitly overridden.
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region
    started = time.perf_counter()
    try:
        while True:
            code, terminal = report_status(args)
            elapsed = time.perf_counter() - started
            print(f"[{utc_now()}] monitor_heartbeat elapsed_seconds={elapsed:.3f}", flush=True)
            if not args.watch or terminal:
                return code
            if elapsed >= args.max_wait_seconds:
                print("OTTO_MONITOR_TIMEOUT: monitoring ended; remote job was not stopped")
                return 2
            time.sleep(min(args.interval_seconds, args.max_wait_seconds - elapsed))
    except KeyboardInterrupt:
        print("OTTO_MONITOR_DETACHED: remote job was not stopped", flush=True)
        return 0
    finally:
        print(
            f"[{utc_now()}] monitor_complete elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
