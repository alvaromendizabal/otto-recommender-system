"""Launch or reconnect to the durable ANN benchmark using the proven worker image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
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
    validate_ann_launch,
    verify_uploaded_source_roundtrip,
)
from otto_recsys.cloud.two_tower_ann import ann_definition, load_ann_parameters, stage_ann_source
from otto_recsys.cloud.two_tower_fold import write_json_atomic
from otto_recsys.experiments.manifest import sha256_file


def watch(pointer: dict[str, Any], *, bucket: str, download: bool, max_wait: float) -> int:
    started = time.perf_counter()
    seen: set[str] = set()
    while True:
        execution = aws_json(
            [
                "sagemaker",
                "describe-pipeline-execution",
                "--pipeline-execution-arn",
                pointer["pipeline_execution_arn"],
            ]
        )
        steps = aws_json(
            [
                "sagemaker",
                "list-pipeline-execution-steps",
                "--pipeline-execution-arn",
                pointer["pipeline_execution_arn"],
            ]
        )["PipelineExecutionSteps"]
        state, details = execution["PipelineExecutionStatus"], {}
        for step in steps:
            arn = step.get("Metadata", {}).get("TrainingJob", {}).get("Arn")
            if arn:
                job = arn.rsplit("/", 1)[-1]
                details = aws_json(
                    ["sagemaker", "describe-training-job", "--training-job-name", job]
                )
                for line in training_logs(job, limit=60):
                    if line not in seen:
                        print(line, flush=True)
                        seen.add(line)
        elapsed = time.perf_counter() - started
        print(
            f"[{utc_now()}] ann_monitor_heartbeat status={state} "
            f"worker_status={details.get('SecondaryStatus', 'Pending')} "
            f"elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        if state in {"Succeeded", "Failed", "Stopped"}:
            evidence = {
                "run_id": pointer["run_id"],
                "timestamp": utc_now(),
                "status": state,
                "billable_seconds": details.get("BillableTimeInSeconds"),
                "failure_reason": details.get("FailureReason"),
                "training_seconds": details.get("TrainingTimeInSeconds"),
            }
            print(json.dumps(evidence, indent=2), flush=True)
            put_json_s3(
                evidence,
                bucket=bucket,
                key=pointer["checkpoint_key"].replace("checkpoints/", "control/executions/")
                + pointer["pipeline_execution_arn"].rsplit("/", 1)[-1]
                + ".json",
            )
            if state != "Succeeded":
                print("OTTO_TWO_TOWER_ANN_FAILED_CHECK_DIAGNOSTICS", flush=True)
                return 1
            report = s3_json(bucket, pointer["checkpoint_key"] + "metrics.json")
            if report["status"] != "passed" or report["input_id"] != pointer["run_id"]:
                raise ValueError("ANN completion report identity mismatch")
            if download:
                root = Path("artifacts/two_tower_ann") / pointer["run_id"]
                run_command(
                    [
                        "aws",
                        "s3",
                        "sync",
                        f"s3://{bucket}/{pointer['checkpoint_key']}",
                        str(root),
                        "--exclude",
                        "*",
                        "--include",
                        "metrics.json*",
                        "--include",
                        "contract.json*",
                        "--include",
                        "selection.json*",
                        "--include",
                        "logs/*",
                        "--only-show-errors",
                    ],
                    check=True,
                )
                receipt = json.loads((root / "metrics.json.json").read_text())
                if receipt["input_id"] != pointer["run_id"] or receipt["sha256"] != sha256_file(
                    root / "metrics.json"
                ):
                    raise ValueError("downloaded ANN report checksum mismatch")
                write_json_atomic(
                    Path("artifacts/two_tower_ann/latest.json"),
                    {**pointer, "report_path": str(root / "metrics.json")},
                )
                print(f"report_path={root / 'metrics.json'}", flush=True)
            print(
                json.dumps(
                    {
                        "full_reference_ranking": report["full_reference_ranking"],
                        "full_ann_ranking": report.get("full_ann_ranking"),
                        "selected_nprobe": report["selected_nprobe"],
                        "confirmation_fidelity_passed": report["confirmation_fidelity_passed"],
                    },
                    indent=2,
                )
            )
            if report.get("prediction_export"):
                print(
                    f"prediction_uri=s3://{bucket}/{pointer['checkpoint_key']}prediction_export/",
                    flush=True,
                )
            print("OTTO_TWO_TOWER_ANN_BENCHMARK_COMPLETED", flush=True)
            return 0
        if elapsed >= max_wait:
            print("OTTO_ANN_MONITOR_DETACHED_REMOTE_JOB_CONTINUES", flush=True)
            return 2
        time.sleep(min(30, max_wait - elapsed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--sample-sessions", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/two_tower_ann.toml"))
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    parser.add_argument("--max-wait-seconds", type=float, default=10800)
    for name in ("start", "watch", "download"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    parameters = load_ann_parameters(args.config)
    if args.sample_sessions is None:
        args.sample_sessions = int(parameters["sample-sessions"])
    if (
        not math.isfinite(args.max_wait_seconds)
        or not 0 <= args.fold < 5
        or min(args.max_runtime_seconds, args.max_wait_seconds) <= 0
        or args.sample_sessions < 4
        or args.sample_sessions % 2
    ):
        raise ValueError("invalid benchmark settings")
    os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_REGION"] = args.region
    pointer_key = f"retrieval/two-tower/ann/fold-{args.fold}/latest.json"
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
    export = s3_json(args.bucket, f"retrieval/two-tower/evaluations/fold-{args.fold}/latest.json")
    for pointer in (latest, export):
        status = aws_json(
            [
                "sagemaker",
                "describe-pipeline-execution",
                "--pipeline-execution-arn",
                pointer["pipeline_execution_arn"],
            ]
        )
        if status["PipelineExecutionStatus"] != "Succeeded":
            raise ValueError("training and exact export must both have succeeded")
    training_key = f"retrieval/two-tower/runs/folds/fold-{args.fold}/{latest['run_id']}/"
    training = s3_json(args.bucket, training_key + "checkpoints/training_manifest.json")
    original = s3_json(args.bucket, training_key + "control/run_manifest.json")
    definition = s3_json(args.bucket, training_key + "control/pipeline_definition.json")
    reference = s3_json(args.bucket, export["checkpoint_key"] + "prediction_manifest.json")
    if (
        reference["training_input_id"] != training["input_id"]
        or reference["status"] != "passed"
        or training["validation_fold"] != args.fold
        or args.sample_sessions > reference["sessions"]
    ):
        raise ValueError("reference/training provenance or sample mismatch")
    # Exporter writes canonical indent=2, sort_keys=True JSON with a trailing newline.
    reference_sha = hashlib.sha256(
        (json.dumps(reference, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    source = Path("gpu/two_tower")
    with tempfile.TemporaryDirectory(prefix="otto-ann-source-") as temporary:
        staged = Path(temporary)
        stage_ann_source(source, staged)
        run_exact_source_preflight(staged)
        archive = Path("artifacts/two_tower_ann/source.tar.gz")
        source_sha = create_deterministic_source_archive(staged, archive)
        verify_source_archive(staged, archive)
        uri = source_s3_uri(args.bucket, commit, source_sha)
        contract = {
            "code_commit": commit,
            "source_sha256": source_sha,
            "training_run_id": latest["run_id"],
            "reference_run_id": export["run_id"],
            "reference_input_id": reference["input_id"],
            "reference_manifest_sha256": reference_sha,
            "training_definition": definition,
            "max_runtime_seconds": args.max_runtime_seconds,
            "sample_sessions": args.sample_sessions,
            "parameters": parameters,
        }
        run_id = canonical_sha256(contract)
        new_definition = ann_definition(
            training_definition=definition,
            bucket=args.bucket,
            training_run_id=latest["run_id"],
            run_id=run_id,
            reference_run_id=export["run_id"],
            reference_input_id=reference["input_id"],
            reference_manifest_sha256=reference_sha,
            source_uri=uri,
            commit=commit,
            training_manifest=training,
            input_manifests=original["input_manifests"],
            max_runtime_seconds=args.max_runtime_seconds,
            sample_sessions=args.sample_sessions,
            parameters=parameters,
            region=args.region,
        )
        validate_ann_launch(staged, new_definition)
        run_command(["aws", "s3", "cp", str(archive), uri, "--only-show-errors"], check=True)
        verify_uploaded_source_roundtrip(
            source_root=staged, source_uri=uri, expected_sha256=source_sha
        )
    name = f"otto-ann-fold-{args.fold}-{run_id[:24]}"
    prefix = f"retrieval/two-tower/ann/fold-{args.fold}/{run_id}/"
    path = Path("artifacts/two_tower_ann") / run_id / "pipeline_definition.json"
    write_json_atomic(path, new_definition)
    put_json_s3(contract, bucket=args.bucket, key=prefix + "control/run_manifest.json")
    put_json_s3(new_definition, bucket=args.bucket, key=prefix + "control/pipeline_definition.json")
    existing = run_command(["aws", "sagemaker", "describe-pipeline", "--pipeline-name", name])
    aws_json(
        [
            "sagemaker",
            "update-pipeline" if existing.returncode == 0 else "create-pipeline",
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
        print(f"pipeline_name={name}\nOTTO_ANN_REGISTERED_NO_COMPUTE_STARTED")
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
        "region": args.region,
        "pipeline_name": name,
        "pipeline_execution_arn": retained["PipelineExecutionArn"],
        "checkpoint_key": prefix + "checkpoints/",
    }
    write_json_atomic(Path("artifacts/two_tower_ann/latest.json"), pointer)
    put_json_s3(pointer, bucket=args.bucket, key=pointer_key)
    print(json.dumps(pointer, indent=2), flush=True)
    print("OTTO_ANN_TRACKED_SAFE_TO_DISCONNECT", flush=True)
    return (
        watch(pointer, bucket=args.bucket, download=args.download, max_wait=args.max_wait_seconds)
        if args.watch
        else 0
    )


def entrypoint() -> int:
    started = time.perf_counter()
    try:
        return main()
    except KeyboardInterrupt:
        print("OTTO_ANN_MONITOR_DETACHED_REMOTE_JOB_CONTINUES", flush=True)
        return 2
    finally:
        print(
            f"[{utc_now()}] ann_control_complete "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(entrypoint())
