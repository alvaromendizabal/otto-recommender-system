from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_command(
    command: list[str], *, check: bool = False
) -> subprocess.CompletedProcess[str]:
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


def _execution_from_latest(
    pipeline_name: str, latest: dict[str, Any]
) -> dict[str, Any] | None:
    execution_arn = latest.get("pipeline_execution_arn")
    if isinstance(execution_arn, str) and execution_arn:
        return aws_json(
            [
                "sagemaker",
                "describe-pipeline-execution",
                "--pipeline-execution-arn",
                execution_arn,
            ]
        )

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
    if not isinstance(summaries, list) or not summaries:
        return None
    summary = summaries[0]
    if not isinstance(summary, dict):
        return None
    execution_arn = summary.get("PipelineExecutionArn")
    if not isinstance(execution_arn, str) or not execution_arn:
        return summary
    return aws_json(
        [
            "sagemaker",
            "describe-pipeline-execution",
            "--pipeline-execution-arn",
            execution_arn,
        ]
    )


def _pipeline_steps(execution_arn: str) -> list[dict[str, Any]]:
    payload = aws_json(
        [
            "sagemaker",
            "list-pipeline-execution-steps",
            "--pipeline-execution-arn",
            execution_arn,
            "--max-results",
            "100",
        ]
    )
    steps = payload.get("PipelineExecutionSteps", [])
    return [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []


def _training_job_arn(step: dict[str, Any]) -> str | None:
    metadata = step.get("Metadata")
    if not isinstance(metadata, dict):
        return None
    training_job = metadata.get("TrainingJob")
    if not isinstance(training_job, dict):
        return None
    arn = training_job.get("Arn")
    return str(arn) if arn else None


def _describe_training_job(training_job_arn: str) -> dict[str, Any]:
    training_job_name = training_job_arn.rsplit("/", maxsplit=1)[-1]
    return aws_json(
        [
            "sagemaker",
            "describe-training-job",
            "--training-job-name",
            training_job_name,
        ]
    )


def _checkpoint_summary(bucket: str, run_id: str) -> tuple[int, int]:
    prefix = f"retrieval/two-tower/runs/resume-proof/{run_id}/checkpoints/"
    payload = aws_json(
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
        ]
    )
    contents = payload.get("Contents", [])
    if not isinstance(contents, list):
        return 0, 0
    sizes = [
        int(item.get("Size", 0))
        for item in contents
        if isinstance(item, dict)
    ]
    return len(sizes), sum(sizes)


def _failure_report_from_output(training_job: dict[str, Any]) -> dict[str, Any] | None:
    job_name = training_job.get("TrainingJobName")
    output = training_job.get("OutputDataConfig")
    if not isinstance(job_name, str) or not isinstance(output, dict):
        return None
    base_uri = output.get("S3OutputPath")
    if not isinstance(base_uri, str) or not base_uri.startswith("s3://"):
        return None
    artifact_uri = f"{base_uri.rstrip('/')}/{job_name}/output/output.tar.gz"
    with tempfile.TemporaryDirectory(prefix="otto-output-status-") as tmpdir:
        local_path = Path(tmpdir) / "output.tar.gz"
        completed = run_command(
            ["aws", "s3", "cp", artifact_uri, str(local_path), "--only-show-errors"]
        )
        if completed.returncode != 0:
            return None
        try:
            with tarfile.open(local_path, mode="r:gz") as archive:
                for candidate in ("failure.json", "./failure.json"):
                    try:
                        member = archive.getmember(candidate)
                    except KeyError:
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    parsed = json.loads(handle.read().decode("utf-8"))
                    return parsed if isinstance(parsed, dict) else None
        except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    return None


def _recent_log_messages(training_job_name: str, limit: int = 40) -> list[str]:
    streams = run_command(
        [
            "aws",
            "logs",
            "describe-log-streams",
            "--log-group-name",
            "/aws/sagemaker/TrainingJobs",
            "--log-stream-name-prefix",
            f"{training_job_name}/",
            "--output",
            "json",
        ]
    )
    if streams.returncode != 0:
        return []
    payload = json.loads(streams.stdout or "{}")
    raw_streams = payload.get("logStreams", [])
    messages: list[str] = []
    if not isinstance(raw_streams, list):
        return messages
    for raw_stream in raw_streams:
        if not isinstance(raw_stream, dict):
            continue
        stream_name = raw_stream.get("logStreamName")
        if not isinstance(stream_name, str):
            continue
        events = run_command(
            [
                "aws",
                "logs",
                "get-log-events",
                "--log-group-name",
                "/aws/sagemaker/TrainingJobs",
                "--log-stream-name",
                stream_name,
                "--start-from-head",
                "--limit",
                "10000",
                "--output",
                "json",
            ]
        )
        if events.returncode != 0:
            continue
        event_payload = json.loads(events.stdout or "{}")
        raw_events = event_payload.get("events", [])
        if not isinstance(raw_events, list):
            continue
        messages.extend(
            str(event.get("message", ""))
            for event in raw_events
            if isinstance(event, dict) and event.get("message")
        )
    return messages[-limit:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("OTTO_BUCKET"))
    parser.add_argument("--show-logs", action="store_true")
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
    execution = _execution_from_latest(pipeline_name, latest)

    print("=" * 72)
    print("OTTO TWO-TOWER MANAGED PIPELINE STATUS")
    print("=" * 72)
    print(f"checked_at={utc_now()}")
    print(f"pipeline_name={pipeline_name}")
    print(f"run_id={run_id}")
    print(f"code_commit={latest.get('code_commit')}")

    execution_status = "NOT_STARTED"
    execution_arn: str | None = None
    if isinstance(execution, dict):
        execution_status = str(
            execution.get("PipelineExecutionStatus")
            or execution.get("PipelineExecutionStatusSummary")
            or "UNKNOWN"
        )
        raw_arn = execution.get("PipelineExecutionArn")
        execution_arn = str(raw_arn) if raw_arn else None
        print(f"execution_status={execution_status}")
        print(f"execution_arn={execution_arn}")
        failure_reason = execution.get("FailureReason")
        if failure_reason:
            print(f"pipeline_failure_reason={failure_reason}")
    else:
        print("execution_status=NOT_STARTED")

    steps: list[dict[str, Any]] = []
    if execution_arn:
        steps = _pipeline_steps(execution_arn)
        for step in steps:
            print(
                f"step={step.get('StepName')} status={step.get('StepStatus')} "
                f"started={step.get('StartTime')} ended={step.get('EndTime')}"
            )

    checkpoint_objects, checkpoint_bytes = _checkpoint_summary(args.bucket, run_id)
    print(f"checkpoint_objects={checkpoint_objects}")
    print(f"checkpoint_bytes={checkpoint_bytes}")

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

    failed_steps = [step for step in steps if step.get("StepStatus") == "Failed"]
    if failed_steps:
        failed_step = failed_steps[0]
        print(f"failed_step={failed_step.get('StepName')}")
        if failed_step.get("FailureReason"):
            print(f"step_failure_reason={failed_step.get('FailureReason')}")
        training_job_arn = _training_job_arn(failed_step)
        if training_job_arn:
            training_job = _describe_training_job(training_job_arn)
            job_name = str(training_job.get("TrainingJobName"))
            print(f"training_job={job_name}")
            print(f"training_status={training_job.get('TrainingJobStatus')}")
            print(f"secondary_status={training_job.get('SecondaryStatus')}")
            print(f"instance_type={training_job.get('ResourceConfig', {}).get('InstanceType')}")
            print(f"training_start={training_job.get('TrainingStartTime')}")
            print(f"training_end={training_job.get('TrainingEndTime')}")
            print(f"billable_seconds={training_job.get('BillableTimeInSeconds')}")
            print(f"training_failure_reason={training_job.get('FailureReason')}")

            failure_report = _failure_report_from_output(training_job)
            if failure_report is not None:
                print(f"failure_stage={failure_report.get('stage')}")
                print(f"failure_message={failure_report.get('message')}")
                print(f"failure_return_code={failure_report.get('return_code')}")

            if args.show_logs:
                messages = _recent_log_messages(job_name)
                if messages:
                    print("--- recent_training_logs ---")
                    for message in messages:
                        print(message.rstrip())
                    print("--- end_recent_training_logs ---")

    if execution_status == "Succeeded":
        if proof is None or proof.get("status") != "passed":
            print("OTTO_TWO_TOWER_PIPELINE_INCONSISTENT")
            return 2
        print("OTTO_TWO_TOWER_MANAGED_RESUME_PROOF_PASSED")
        return 0
    if execution_status == "Failed":
        print("OTTO_TWO_TOWER_PIPELINE_FAILED")
        return 3
    if execution_status == "Stopped":
        print("OTTO_TWO_TOWER_PIPELINE_STOPPED")
        return 4
    print("OTTO_TWO_TOWER_PIPELINE_IN_PROGRESS_OR_NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
