"""Register, run, reconnect to, and download a resumable fold evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from launch_two_tower_fold import (
    aws_json,
    ensure_committed_remote_state,
    execution_token,
    pipeline_executions,
    put_json_s3,
    run_command,
    s3_json,
    utc_now,
)
from two_tower_fold_status import training_logs

from otto_recsys.cloud.sagemaker_pipeline import (
    canonical_sha256,
    create_deterministic_source_archive,
    source_s3_uri,
    verify_source_archive,
)
from otto_recsys.cloud.source_preflight import (
    run_exact_source_preflight,
    validate_evaluation_launch,
    verify_uploaded_source_roundtrip,
)
from otto_recsys.cloud.two_tower_evaluation import evaluation_definition
from otto_recsys.cloud.two_tower_fold import write_json_atomic


def watch(pointer: dict[str, Any], *, bucket: str, download: bool, max_wait: float) -> int:
    started = time.perf_counter()
    seen_logs: set[str] = set()
    while True:
        execution = aws_json(
            [
                "sagemaker",
                "describe-pipeline-execution",
                "--pipeline-execution-arn",
                pointer["pipeline_execution_arn"],
            ]
        )
        status = execution["PipelineExecutionStatus"]
        steps = aws_json(
            [
                "sagemaker",
                "list-pipeline-execution-steps",
                "--pipeline-execution-arn",
                pointer["pipeline_execution_arn"],
            ]
        )["PipelineExecutionSteps"]
        job_states: list[str] = []
        for step in steps:
            job = step.get("Metadata", {}).get("TrainingJob", {}).get("Arn")
            if job:
                name = job.rsplit("/", 1)[-1]
                details = aws_json(
                    ["sagemaker", "describe-training-job", "--training-job-name", name]
                )
                job_states.append(str(details.get("SecondaryStatus", details["TrainingJobStatus"])))
                if status in {"Failed", "Stopped"}:
                    print(
                        json.dumps(
                            {
                                "timestamp": utc_now(),
                                "event": "evaluation_job_terminal",
                                "job": name,
                                "status": details["TrainingJobStatus"],
                                "failure_reason": details.get("FailureReason"),
                                "billable_seconds": details.get("BillableTimeInSeconds"),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                for line in training_logs(name, limit=40):
                    if line not in seen_logs:
                        print(line, flush=True)
                        seen_logs.add(line)
        elapsed = time.perf_counter() - started
        print(
            f"[{utc_now()}] evaluation_heartbeat status={status} "
            f"worker_status={','.join(job_states) or 'Pending'} elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        if status == "Succeeded":
            manifest = s3_json(bucket, pointer["checkpoint_key"] + "prediction_manifest.json")
            if manifest["status"] != "passed":
                raise RuntimeError("prediction manifest did not pass")
            if download:
                destination = (
                    Path("data/interim/two_tower")
                    / f"fold-{pointer['validation_fold']}"
                    / pointer["run_id"]
                )
                run_command(
                    [
                        "aws",
                        "s3",
                        "sync",
                        f"s3://{bucket}/{pointer['checkpoint_key']}",
                        str(destination),
                        "--exclude",
                        "embeddings/*",
                        "--only-show-errors",
                    ],
                    check=True,
                )
                write_json_atomic(
                    Path("artifacts/two_tower_evaluation/latest.json"),
                    {**pointer, "predictions_dir": str(destination)},
                )
                print(f"predictions_dir={destination}")
            print("OTTO_TWO_TOWER_EVALUATION_EXPORT_PASSED", flush=True)
            return 0
        if status in {"Failed", "Stopped"}:
            print(json.dumps(execution, indent=2))
            print("OTTO_TWO_TOWER_EVALUATION_FAILED_CHECK_DIAGNOSTICS", flush=True)
            print(
                "Saved training weights are unchanged. A retry with the same source and "
                "configuration reuses verified evaluation parts if any were uploaded.",
                flush=True,
            )
            return 1
        if elapsed >= max_wait:
            print("OTTO_EVALUATION_MONITOR_DETACHED_REMOTE_JOB_CONTINUES", flush=True)
            return 2
        time.sleep(min(30, max_wait - elapsed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    parser.add_argument("--max-wait-seconds", type=float, default=10800)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if (
        not 0 <= args.fold < 5
        or min(args.max_runtime_seconds, args.max_wait_seconds, args.batch_size) <= 0
    ):
        raise ValueError("invalid fold or runtime settings")
    os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_REGION"] = args.region
    pointer_key = f"retrieval/two-tower/evaluations/fold-{args.fold}/latest.json"
    if args.watch and not args.start:
        return watch(
            s3_json(args.bucket, pointer_key),
            bucket=args.bucket,
            download=args.download,
            max_wait=args.max_wait_seconds,
        )
    commit = ensure_committed_remote_state()
    latest = s3_json(
        args.bucket, f"retrieval/two-tower/pipelines/folds/fold-{args.fold}/latest.json"
    )
    execution = aws_json(
        [
            "sagemaker",
            "describe-pipeline-execution",
            "--pipeline-execution-arn",
            latest["pipeline_execution_arn"],
        ]
    )
    if execution["PipelineExecutionStatus"] != "Succeeded":
        raise ValueError("training pipeline must have succeeded")
    training_run = latest["run_id"]
    training_key = f"retrieval/two-tower/runs/folds/fold-{args.fold}/{training_run}/"
    manifest = s3_json(args.bucket, training_key + "checkpoints/training_manifest.json")
    original = s3_json(args.bucket, training_key + "control/run_manifest.json")
    definition = s3_json(args.bucket, training_key + "control/pipeline_definition.json")
    if (
        manifest["code_commit"] != original["code_commit"]
        or manifest["validation_fold"] != args.fold
    ):
        raise ValueError("training provenance mismatch")
    source = Path("gpu/two_tower")
    run_exact_source_preflight(source)
    archive = Path("artifacts/two_tower_evaluation/source.tar.gz")
    source_sha = create_deterministic_source_archive(source, archive)
    verify_source_archive(source, archive)
    uri = source_s3_uri(args.bucket, commit, source_sha)
    contract = {
        "training_run_id": training_run,
        "training_input_id": manifest["input_id"],
        "code_commit": commit,
        "source_sha256": source_sha,
        "training_definition": definition,
        "input_manifests": original["input_manifests"],
        "max_runtime_seconds": args.max_runtime_seconds,
        "batch_size": args.batch_size,
    }
    run_id = canonical_sha256(contract)
    name = f"otto-two-tower-eval-{args.fold}-{run_id[:24]}"
    new_definition = evaluation_definition(
        training_definition=definition,
        bucket=args.bucket,
        training_run_id=training_run,
        evaluation_id=run_id,
        source_uri=uri,
        commit=commit,
        training_manifest=manifest,
        input_manifests=original["input_manifests"],
        max_runtime_seconds=args.max_runtime_seconds,
        batch_size=args.batch_size,
    )
    validate_evaluation_launch(source, new_definition)
    run_command(["aws", "s3", "cp", str(archive), uri, "--only-show-errors"], check=True)
    verify_uploaded_source_roundtrip(source_root=source, source_uri=uri, expected_sha256=source_sha)
    root = Path("artifacts/two_tower_evaluation") / run_id
    path = root / "pipeline_definition.json"
    write_json_atomic(path, new_definition)
    write_json_atomic(root / "run_manifest.json", contract)
    prefix = f"retrieval/two-tower/evaluations/fold-{args.fold}/{run_id}/"
    put_json_s3(contract, bucket=args.bucket, key=prefix + "control/run_manifest.json")
    existing = run_command(["aws", "sagemaker", "describe-pipeline", "--pipeline-name", name])
    action = "update-pipeline" if existing.returncode == 0 else "create-pipeline"
    aws_json(
        [
            "sagemaker",
            action,
            "--pipeline-name",
            name,
            "--pipeline-definition",
            f"file://{path}",
            "--role-arn",
            definition["Steps"][0]["Arguments"]["RoleArn"],
        ]
    )
    executions = pipeline_executions(name)
    retained = next(
        (row for row in executions if row["PipelineExecutionStatus"] in {"Executing", "Stopping"}),
        None,
    )
    if retained is None:
        retained = next(
            (row for row in executions if row["PipelineExecutionStatus"] == "Succeeded"), None
        )
    if retained is None and not args.start:
        print(f"pipeline_name={name}\nOTTO_TWO_TOWER_EVALUATION_REGISTERED_NO_GPU_STARTED")
        return 0
    if retained is None:
        retained = aws_json(
            [
                "sagemaker",
                "start-pipeline-execution",
                "--pipeline-name",
                name,
                "--client-request-token",
                execution_token(run_id, executions),
            ]
        )
    pointer = {
        "run_id": run_id,
        "validation_fold": args.fold,
        "pipeline_name": name,
        "pipeline_execution_arn": retained["PipelineExecutionArn"],
        "region": args.region,
        "checkpoint_key": prefix + "checkpoints/",
    }
    write_json_atomic(Path("artifacts/two_tower_evaluation/latest.json"), pointer)
    put_json_s3(pointer, bucket=args.bucket, key=pointer_key)
    print(json.dumps(pointer, indent=2))
    print("OTTO_TWO_TOWER_EVALUATION_TRACKED_SAFE_TO_DISCONNECT", flush=True)
    if args.watch:
        return watch(
            pointer, bucket=args.bucket, download=args.download, max_wait=args.max_wait_seconds
        )
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("OTTO_EVALUATION_MONITOR_DETACHED_REMOTE_JOB_CONTINUES", flush=True)
    finally:
        print(
            f"[{utc_now()}] evaluation_control_complete "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
