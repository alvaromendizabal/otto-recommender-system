from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from otto_recsys.cloud.sagemaker_two_tower import (
    ResumeProofConfig,
    build_training_request,
    canonical_sha256,
    derive_role_name_from_sts_arn,
    job_name,
    official_pytorch_image,
    resume_proof_payload,
    run_s3_prefix,
    source_archive_sha256,
    source_s3_prefix,
)

TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}


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


def aws_text(arguments: list[str]) -> str:
    completed = run_command(["aws", *arguments, "--output", "text"], check=True)
    return completed.stdout.strip()


def resolve_role_arn(explicit: str | None) -> str:
    if explicit:
        return explicit
    caller = aws_json(["sts", "get-caller-identity"])
    arn = str(caller["Arn"])
    role_name = derive_role_name_from_sts_arn(arn)
    if role_name is None:
        raise RuntimeError(
            "could not derive a SageMaker execution role from the current AWS identity; "
            "pass --role-arn explicitly"
        )
    try:
        role = aws_json(["iam", "get-role", "--role-name", role_name])
        return str(role["Role"]["Arn"])
    except RuntimeError:
        account = str(caller["Account"])
        return f"arn:aws:iam::{account}:role/{role_name}"


def ensure_clean_git_tree() -> str:
    status = run_command(["git", "status", "--porcelain"], check=True).stdout.strip()
    if status:
        raise RuntimeError("Git working tree must be clean before a paid GPU resume proof")
    return run_command(["git", "rev-parse", "HEAD"], check=True).stdout.strip()


def add_source_tree(archive: tarfile.TarFile, source_root: Path) -> None:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if any(part in ignored for part in relative.parts):
            continue
        if path.is_file() and path.suffix != ".pyc":
            archive.add(path, arcname=str(relative), recursive=False)


def create_source_archive(source_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, mode="w:gz") as archive:
        add_source_tree(archive, source_root)
    os.replace(temporary, destination)


def put_json_s3(payload: dict[str, Any], *, bucket: str, key: str) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(encoded)
        handle.flush()
        run_command(
            [
                "aws",
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--body",
                handle.name,
            ],
            check=True,
        )


def get_json_s3(bucket: str, key: str) -> dict[str, Any] | None:
    completed = run_command(
        [
            "aws",
            "s3",
            "cp",
            f"s3://{bucket}/{key}",
            "-",
            "--only-show-errors",
        ]
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, dict) else None


def head_s3(bucket: str, key: str) -> dict[str, Any] | None:
    completed = run_command(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json"]
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, dict) else None


def describe_job(job: str) -> dict[str, Any]:
    return aws_json(["sagemaker", "describe-training-job", "--training-job-name", job])


def validate_request(request_path: Path) -> None:
    run_command(
        [
            "aws",
            "sagemaker",
            "create-training-job",
            "--cli-input-json",
            f"file://{request_path}",
            "--generate-cli-skeleton",
            "output",
        ],
        check=True,
    )


def create_job(request_path: Path) -> None:
    run_command(
        [
            "aws",
            "sagemaker",
            "create-training-job",
            "--cli-input-json",
            f"file://{request_path}",
        ],
        check=True,
    )


def stop_job(job: str) -> None:
    status = str(describe_job(job)["TrainingJobStatus"])
    if status not in TERMINAL_STATUSES:
        run_command(
            ["aws", "sagemaker", "stop-training-job", "--training-job-name", job],
            check=True,
        )


def wait_terminal(job: str, *, poll_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    while True:
        description = describe_job(job)
        status = str(description["TrainingJobStatus"])
        elapsed = time.perf_counter() - started
        print(
            f"[{utc_now()}] job_heartbeat job={job} status={status} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        if status in TERMINAL_STATUSES:
            return description
        if elapsed > timeout_seconds:
            raise TimeoutError(f"timed out waiting for {job} to stop")
        time.sleep(poll_seconds)


def wait_checkpoint(
    *,
    job: str,
    bucket: str,
    prefix_key: str,
    poll_seconds: int,
    timeout_seconds: int,
    minimum_step: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    progress_key = f"{prefix_key}/progress.json"
    checkpoint_key = f"{prefix_key}/checkpoint.pt"
    while True:
        description = describe_job(job)
        status = str(description["TrainingJobStatus"])
        progress = get_json_s3(bucket, progress_key)
        checkpoint = head_s3(bucket, checkpoint_key)
        step = int(progress.get("global_step", 0)) if progress else 0
        elapsed = time.perf_counter() - started
        checkpoint_bytes = int(checkpoint.get("ContentLength", 0)) if checkpoint else 0
        print(
            f"[{utc_now()}] checkpoint_heartbeat job={job} status={status} "
            f"global_step={step} checkpoint_bytes={checkpoint_bytes} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        if progress is not None and checkpoint_bytes > 0 and step >= minimum_step:
            return progress
        if status in TERMINAL_STATUSES:
            reason = description.get("FailureReason")
            raise RuntimeError(
                f"{job} reached terminal status {status} before checkpoint proof; reason={reason}"
            )
        if elapsed > timeout_seconds:
            raise TimeoutError(f"timed out waiting for checkpoint from {job}")
        time.sleep(poll_seconds)


def active_jobs(name_contains: str) -> list[str]:
    payload = aws_json(
        [
            "sagemaker",
            "list-training-jobs",
            "--name-contains",
            name_contains,
            "--status-equals",
            "InProgress",
            "--max-items",
            "20",
            "--no-paginate",
        ]
    )
    summaries = payload.get("TrainingJobSummaries", [])
    if not isinstance(summaries, list):
        return []
    return [
        str(summary["TrainingJobName"])
        for summary in summaries
        if isinstance(summary, dict) and "TrainingJobName" in summary
    ]


def request_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("OTTO_BUCKET"))
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--role-arn")
    parser.add_argument("--instance-type", default="ml.g6.xlarge")
    parser.add_argument("--volume-size-gb", type=int, default=100)
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--valid-rows", type=int, default=10_000)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-runtime-seconds", type=int, default=1_800)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise RuntimeError("--bucket or OTTO_BUCKET is required")
    overall_started = time.perf_counter()
    config = ResumeProofConfig(
        bucket=args.bucket,
        region=args.region,
        instance_type=args.instance_type,
        volume_size_gb=args.volume_size_gb,
        train_rows=args.train_rows,
        valid_rows=args.valid_rows,
        checkpoint_steps=args.checkpoint_steps,
        poll_seconds=args.poll_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    config.validate()
    commit = ensure_clean_git_tree()
    role_arn = resolve_role_arn(args.role_arn)
    image_uri = official_pytorch_image(config.region)

    source_root = Path("gpu/two_tower")
    source_archive = Path("artifacts/two_tower_resume_proof/source.tar.gz")
    create_source_archive(source_root, source_archive)
    source_sha = source_archive_sha256(source_archive)
    source_prefix = source_s3_prefix(config.bucket, commit, source_sha)
    source_object = f"{source_prefix}source.tar.gz"
    run_command(
        ["aws", "s3", "cp", str(source_archive), source_object, "--only-show-errors"],
        check=True,
    )

    source_head = head_s3(config.bucket, source_object.split(f"s3://{config.bucket}/", 1)[1])
    if source_head is None or int(source_head.get("ContentLength", 0)) <= 0:
        raise RuntimeError("source archive was not durably verified in S3")

    required_keys = [
        "candidates/ranking-training-cache/manifest.json",
        "candidates/hard-negatives/manifest.json",
        "retrieval/two-tower/items/manifest.json",
        "retrieval/two-tower/items/item_vectors.npy",
    ]
    for key in required_keys:
        metadata = head_s3(config.bucket, key)
        if metadata is None or int(metadata.get("ContentLength", 0)) <= 0:
            raise RuntimeError(f"required durable S3 input is missing: s3://{config.bucket}/{key}")

    proof_input = resume_proof_payload(
        commit=commit,
        source_sha256=source_sha,
        image_uri=image_uri,
        config=config,
    )
    run_id = canonical_sha256(proof_input)
    run_prefix = run_s3_prefix(config.bucket, run_id)
    checkpoint_prefix = f"{run_prefix}checkpoints/"
    output_prefix = f"{run_prefix}output/"
    control_prefix = f"retrieval/two-tower/runs/resume-proof/{run_id}/control"
    local_root = Path("artifacts/two_tower_resume_proof") / run_id
    local_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "code_commit": commit,
        "source_sha256": source_sha,
        "source_s3_uri": source_object,
        "checkpoint_s3_uri": checkpoint_prefix,
        "image_uri": image_uri,
        "role_arn": role_arn,
        "config": asdict(config),
    }
    request_file(local_root / "run_manifest.json", manifest)
    put_json_s3(manifest, bucket=config.bucket, key=f"{control_prefix}/run_manifest.json")

    proof_key = f"{control_prefix}/resume_proof.json"

    if args.dry_run:
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        base_name = f"otto-tt-rp-{run_id[:10]}"
        dry_a = job_name(base_name, stamp, "a")
        dry_b = job_name(base_name, stamp, "b")
        dry_request_a = build_training_request(
            job_name_value=dry_a,
            role_arn=role_arn,
            image_uri=image_uri,
            source_prefix=source_prefix,
            checkpoint_prefix=checkpoint_prefix,
            output_prefix=output_prefix,
            commit=commit,
            run_id=run_id,
            config=config,
            resume=False,
        )
        dry_request_b = build_training_request(
            job_name_value=dry_b,
            role_arn=role_arn,
            image_uri=image_uri,
            source_prefix=source_prefix,
            checkpoint_prefix=checkpoint_prefix,
            output_prefix=output_prefix,
            commit=commit,
            run_id=run_id,
            config=config,
            resume=True,
        )
        dry_a_path = local_root / "dry-run-job-a.json"
        dry_b_path = local_root / "dry-run-job-b.json"
        request_file(dry_a_path, dry_request_a)
        request_file(dry_b_path, dry_request_b)
        validate_request(dry_a_path)
        validate_request(dry_b_path)
        active_jobs(base_name)
        dry_payload = {
            "status": "validated",
            "run_id": run_id,
            "code_commit": commit,
            "source_sha256": source_sha,
            "source_s3_uri": source_object,
            "checkpoint_s3_uri": checkpoint_prefix,
            "image_uri": image_uri,
            "role_arn": role_arn,
            "config": asdict(config),
        }
        request_file(local_root / "dry_run.json", dry_payload)
        put_json_s3(
            dry_payload,
            bucket=config.bucket,
            key=f"{control_prefix}/dry_run.json",
        )
        print(json.dumps(dry_payload, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_RESUME_PROOF_DRY_RUN_PASSED")
        return 0

    previous_proof = get_json_s3(config.bucket, proof_key)
    if previous_proof is not None and previous_proof.get("status") == "passed":
        print(json.dumps(previous_proof, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_RESUME_PROOF_ALREADY_PASSED")
        return 0

    checkpoint_key_prefix = (
        f"retrieval/two-tower/runs/resume-proof/{run_id}/checkpoints"
    )
    existing_progress = get_json_s3(
        config.bucket,
        f"{checkpoint_key_prefix}/progress.json",
    )
    existing_checkpoint = head_s3(
        config.bucket,
        f"{checkpoint_key_prefix}/checkpoint.pt",
    )
    base_name = f"otto-tt-rp-{run_id[:10]}"
    current_jobs = active_jobs(base_name)

    progress_a: dict[str, Any]
    final_a: dict[str, Any]
    if (
        existing_progress is not None
        and existing_checkpoint is not None
        and int(existing_progress.get("global_step", 0)) > 0
        and int(existing_checkpoint.get("ContentLength", 0)) > 0
    ):
        for active_job in current_jobs:
            print(f"[{utc_now()}] stopping_stale_job name={active_job}", flush=True)
            stop_job(active_job)
            wait_terminal(
                active_job,
                poll_seconds=config.poll_seconds,
                timeout_seconds=600,
            )
        progress_a = existing_progress
        final_a = {
            "TrainingJobName": "existing-durable-checkpoint",
            "TrainingJobStatus": "CheckpointAvailable",
        }
        print(
            f"[{utc_now()}] reusing_durable_checkpoint "
            f"global_step={progress_a['global_step']}",
            flush=True,
        )
    else:
        active_a = next((name for name in current_jobs if "-a-" in name), None)
        if active_a is None:
            stamp_a = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            job_a = job_name(base_name, stamp_a, "a")
            request_a = build_training_request(
                job_name_value=job_a,
                role_arn=role_arn,
                image_uri=image_uri,
                source_prefix=source_prefix,
                checkpoint_prefix=checkpoint_prefix,
                output_prefix=output_prefix,
                commit=commit,
                run_id=run_id,
                config=config,
                resume=False,
            )
            request_a_path = local_root / f"{job_a}.json"
            request_file(request_a_path, request_a)
            put_json_s3(
                request_a,
                bucket=config.bucket,
                key=f"{control_prefix}/{job_a}.json",
            )
            print(f"[{utc_now()}] launching_job_a name={job_a}", flush=True)
            create_job(request_a_path)
        else:
            job_a = active_a
            print(f"[{utc_now()}] continuing_existing_job_a name={job_a}", flush=True)

        try:
            progress_a = wait_checkpoint(
                job=job_a,
                bucket=config.bucket,
                prefix_key=checkpoint_key_prefix,
                poll_seconds=config.poll_seconds,
                timeout_seconds=config.max_runtime_seconds,
                minimum_step=config.checkpoint_steps,
            )
        except Exception:
            stop_job(job_a)
            raise
        stop_job(job_a)
        final_a = wait_terminal(
            job_a,
            poll_seconds=config.poll_seconds,
            timeout_seconds=600,
        )

    step_a = int(progress_a["global_step"])
    if step_a <= 0:
        raise RuntimeError("durable checkpoint did not contain nonzero global_step")

    for active_job in active_jobs(base_name):
        print(f"[{utc_now()}] stopping_concurrent_job name={active_job}", flush=True)
        stop_job(active_job)
        wait_terminal(
            active_job,
            poll_seconds=config.poll_seconds,
            timeout_seconds=600,
        )

    stamp_b = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    job_b = job_name(base_name, stamp_b, "b")
    request_b = build_training_request(
        job_name_value=job_b,
        role_arn=role_arn,
        image_uri=image_uri,
        source_prefix=source_prefix,
        checkpoint_prefix=checkpoint_prefix,
        output_prefix=output_prefix,
        commit=commit,
        run_id=run_id,
        config=config,
        resume=True,
    )
    request_b_path = local_root / f"{job_b}.json"
    request_file(request_b_path, request_b)
    put_json_s3(
        request_b,
        bucket=config.bucket,
        key=f"{control_prefix}/{job_b}.json",
    )
    print(f"[{utc_now()}] launching_job_b name={job_b} resume_from_step={step_a}", flush=True)
    create_job(request_b_path)
    try:
        progress_b = wait_checkpoint(
            job=job_b,
            bucket=config.bucket,
            prefix_key=checkpoint_key_prefix,
            poll_seconds=config.poll_seconds,
            timeout_seconds=config.max_runtime_seconds,
            minimum_step=step_a + config.checkpoint_steps,
        )
    except Exception:
        stop_job(job_b)
        raise
    stop_job(job_b)
    final_b = wait_terminal(
        job_b,
        poll_seconds=config.poll_seconds,
        timeout_seconds=600,
    )
    step_b = int(progress_b["global_step"])
    if step_b <= step_a:
        raise RuntimeError("Job B did not advance beyond the restored checkpoint")

    proof = {
        "status": "passed",
        "run_id": run_id,
        "code_commit": commit,
        "source_sha256": source_sha,
        "image_uri": image_uri,
        "checkpoint_s3_uri": checkpoint_prefix,
        "job_a": {
            "name": str(final_a.get("TrainingJobName", "checkpoint-baseline")),
            "terminal_status": final_a["TrainingJobStatus"],
            "progress": progress_a,
        },
        "job_b": {
            "name": job_b,
            "terminal_status": final_b["TrainingJobStatus"],
            "progress": progress_b,
        },
        "elapsed_seconds": round(time.perf_counter() - overall_started, 3),
    }
    request_file(local_root / "resume_proof.json", proof)
    put_json_s3(proof, bucket=config.bucket, key=proof_key)
    print(json.dumps(proof, indent=2, sort_keys=True))
    print("OTTO_TWO_TOWER_RESUME_PROOF_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
