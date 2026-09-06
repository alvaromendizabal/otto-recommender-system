from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from otto_recsys.cloud.sagemaker_pipeline import (
    create_deterministic_source_archive,
    official_pytorch_image,
    source_s3_uri,
    verify_source_archive,
)
from otto_recsys.cloud.sagemaker_two_tower import derive_role_name_from_sts_arn
from otto_recsys.cloud.source_preflight import (
    run_command,
    run_exact_source_preflight,
    verify_uploaded_source_roundtrip,
)
from otto_recsys.cloud.two_tower_fold import (
    build_fold_pipeline_definition,
    fold_pipeline_name,
    fold_run_contract,
    fold_run_id,
    fold_run_prefix,
    load_fold_config,
    write_json_atomic,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def aws_json(arguments: list[str]) -> dict[str, Any]:
    completed = run_command(["aws", *arguments, "--output", "json"], check=True)
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("AWS CLI returned a non-object JSON payload")
    return payload


def ensure_committed_remote_state() -> str:
    status = run_command(["git", "status", "--porcelain"], check=True).stdout.strip()
    if status:
        raise RuntimeError("Git working tree must be clean before managed GPU execution")
    run_command(["git", "fetch", "--quiet", "origin", "main"], check=True)
    head = run_command(["git", "rev-parse", "HEAD"], check=True).stdout.strip()
    remote = run_command(["git", "rev-parse", "origin/main"], check=True).stdout.strip()
    if head != remote:
        raise RuntimeError("HEAD must exactly match origin/main before managed GPU execution")
    return head


def resolve_role_arn(explicit: str | None) -> str:
    if explicit:
        return explicit
    caller = aws_json(["sts", "get-caller-identity"])
    arn = str(caller["Arn"])
    role_name = derive_role_name_from_sts_arn(arn)
    if role_name is None:
        raise RuntimeError(
            "could not derive SageMaker execution role; pass --role-arn explicitly"
        )
    role = run_command(
        ["aws", "iam", "get-role", "--role-name", role_name, "--output", "json"]
    )
    if role.returncode == 0:
        payload = json.loads(role.stdout)
        return str(payload["Role"]["Arn"])
    account = str(caller["Account"])
    return f"arn:aws:iam::{account}:role/{role_name}"


def manifest_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def s3_json(bucket: str, key: str) -> dict[str, Any]:
    completed = run_command(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", "-", "--only-show-errors"],
        check=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"s3://{bucket}/{key} is not a JSON object")
    return payload


def head_s3(bucket: str, key: str) -> dict[str, Any] | None:
    completed = run_command(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json"]
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, dict) else None


def put_json_s3(payload: dict[str, Any], *, bucket: str, key: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
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


def pipeline_exists(name: str) -> bool:
    completed = run_command(
        ["aws", "sagemaker", "describe-pipeline", "--pipeline-name", name, "--output", "json"]
    )
    return completed.returncode == 0


def register_pipeline(
    *, name: str, role_arn: str, definition_path: Path, run_id: str, commit: str, fold: int
) -> dict[str, Any]:
    common = [
        "--pipeline-name",
        name,
        "--pipeline-definition",
        f"file://{definition_path}",
        "--pipeline-description",
        f"OTTO two-tower OOF fold={fold} run_id={run_id}",
        "--role-arn",
        role_arn,
    ]
    if pipeline_exists(name):
        return aws_json(["sagemaker", "update-pipeline", *common])
    return aws_json(
        [
            "sagemaker",
            "create-pipeline",
            *common,
            "--tags",
            "Key=Project,Value=otto-recommender-system",
            "Key=Model,Value=two-tower",
            "Key=Purpose,Value=fold-training",
            f"Key=ValidationFold,Value={fold}",
            f"Key=CodeCommit,Value={commit[:40]}",
            f"Key=RunId,Value={run_id[:40]}",
        ]
    )


def active_execution(name: str) -> dict[str, Any] | None:
    payload = aws_json(
        [
            "sagemaker",
            "list-pipeline-executions",
            "--pipeline-name",
            name,
            "--max-results",
            "20",
        ]
    )
    summaries = payload.get("PipelineExecutionSummaries", [])
    if not isinstance(summaries, list):
        return None
    for summary in summaries:
        if isinstance(summary, dict) and summary.get("PipelineExecutionStatus") in {
            "Executing",
            "Stopping",
        }:
            return summary
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("OTTO_BUCKET"))
    parser.add_argument("--config", type=Path, default=Path("configs/two_tower.toml"))
    parser.add_argument("--profile", default="fold0")
    parser.add_argument("--role-arn")
    parser.add_argument("--region")
    parser.add_argument("--instance-type")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise RuntimeError("--bucket or OTTO_BUCKET is required")

    config = load_fold_config(args.config, profile=args.profile, bucket=args.bucket)
    if args.region is not None or args.instance_type is not None:
        from dataclasses import replace

        config = replace(
            config,
            region=args.region or config.region,
            instance_type=args.instance_type or config.instance_type,
        )
    config.validate()

    commit = ensure_committed_remote_state()
    role_arn = resolve_role_arn(args.role_arn)
    image_uri = official_pytorch_image(config.region)
    source_root = Path("gpu/two_tower")

    run_exact_source_preflight(source_root)
    source_archive = Path("artifacts/two_tower_fold/source.tar.gz")
    source_sha = create_deterministic_source_archive(source_root, source_archive)
    local_verification = verify_source_archive(source_root, source_archive)
    print(
        f"[{utc_now()}] source_archive_verified files={local_verification['files']} "
        f"manifest_sha256={local_verification['manifest_sha256']} "
        f"archive_sha256={source_sha}",
        flush=True,
    )

    source_uri = source_s3_uri(config.bucket, commit, source_sha)
    run_command(
        ["aws", "s3", "cp", str(source_archive), source_uri, "--only-show-errors"],
        check=True,
    )
    remote_verification = verify_uploaded_source_roundtrip(
        source_root=source_root,
        source_uri=source_uri,
        expected_sha256=source_sha,
    )
    print(
        f"[{utc_now()}] source_s3_roundtrip_verified "
        f"files={remote_verification['files']} archive_sha256={source_sha}",
        flush=True,
    )

    manifest_keys = {
        "ranking": "candidates/ranking-training-cache/manifest.json",
        "hard_negatives": "candidates/hard-negatives/manifest.json",
        "items": "retrieval/two-tower/items/manifest.json",
    }
    input_manifests = {
        name: manifest_identity(s3_json(config.bucket, key))
        for name, key in manifest_keys.items()
    }

    contract = fold_run_contract(
        commit=commit,
        source_sha256=source_sha,
        source_manifest_sha256=str(local_verification["manifest_sha256"]),
        source_uri=source_uri,
        image_uri=image_uri,
        role_arn=role_arn,
        config=config,
        input_manifests=input_manifests,
    )
    run_id = fold_run_id(contract)
    name = fold_pipeline_name(config.validation_fold, run_id)
    run_prefix = fold_run_prefix(config.bucket, config.validation_fold, run_id)
    control_prefix = (
        f"retrieval/two-tower/runs/folds/fold-{config.validation_fold}/{run_id}/control"
    )
    training_manifest_key = (
        f"retrieval/two-tower/runs/folds/fold-{config.validation_fold}/{run_id}/"
        "checkpoints/training_manifest.json"
    )

    existing_manifest_head = head_s3(config.bucket, training_manifest_key)
    if existing_manifest_head is not None and not args.force:
        existing_manifest = s3_json(config.bucket, training_manifest_key)
        if existing_manifest.get("global_step", 0):
            print(json.dumps(existing_manifest, indent=2, sort_keys=True))
            print("OTTO_TWO_TOWER_FOLD_ALREADY_TRAINED")
            return 0

    definition = build_fold_pipeline_definition(
        role_arn=role_arn,
        image_uri=image_uri,
        source_uri=source_uri,
        commit=commit,
        run_id=run_id,
        config=config,
    )
    local_root = Path("artifacts/two_tower_fold") / run_id
    definition_path = local_root / "pipeline_definition.json"
    manifest_path = local_root / "run_manifest.json"
    write_json_atomic(definition_path, definition)

    public_contract = {
        "schema_version": 1,
        "purpose": "two-tower-fold-training",
        "validation_fold": config.validation_fold,
        "code_commit": commit,
        "source_sha256": source_sha,
        "source_manifest_sha256": local_verification["manifest_sha256"],
        "run_id": run_id,
        "pipeline_name": name,
        "config": {
            "instance_type": config.instance_type,
            "volume_size_gb": config.volume_size_gb,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "max_seq_len": config.max_seq_len,
            "checkpoint_steps": config.checkpoint_steps,
            "heartbeat_seconds": config.heartbeat_seconds,
            "max_runtime_seconds": config.max_runtime_seconds,
            "seed": config.seed,
        },
        "input_manifests": input_manifests,
        "created_at": utc_now(),
    }
    write_json_atomic(manifest_path, public_contract)
    put_json_s3(definition, bucket=config.bucket, key=f"{control_prefix}/pipeline_definition.json")
    put_json_s3(public_contract, bucket=config.bucket, key=f"{control_prefix}/run_manifest.json")

    registration = register_pipeline(
        name=name,
        role_arn=role_arn,
        definition_path=definition_path,
        run_id=run_id,
        commit=commit,
        fold=config.validation_fold,
    )
    print(
        f"[{utc_now()}] fold_pipeline_registered fold={config.validation_fold} "
        f"name={name} arn={registration.get('PipelineArn')}",
        flush=True,
    )

    latest = {
        "pipeline_name": name,
        "run_id": run_id,
        "validation_fold": config.validation_fold,
        "code_commit": commit,
        "checkpoint_s3_uri": f"{run_prefix}checkpoints/",
        "run_manifest_s3_uri": f"s3://{config.bucket}/{control_prefix}/run_manifest.json",
    }
    latest_key = f"retrieval/two-tower/pipelines/folds/fold-{config.validation_fold}/latest.json"
    put_json_s3(latest, bucket=config.bucket, key=latest_key)
    write_json_atomic(Path("artifacts/two_tower_fold/latest.json"), latest)

    if not args.start:
        print(json.dumps(latest, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_FOLD_REGISTERED_NO_GPU_STARTED")
        return 0

    current = active_execution(name)
    if current is not None and not args.force:
        print(json.dumps(current, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_FOLD_ALREADY_RUNNING_SAFE_TO_DISCONNECT")
        return 0

    execution = aws_json(
        [
            "sagemaker",
            "start-pipeline-execution",
            "--pipeline-name",
            name,
            "--pipeline-execution-display-name",
            f"fold-{config.validation_fold}-{run_id[:12]}",
        ]
    )
    execution_arn = str(execution["PipelineExecutionArn"])
    started = {
        **latest,
        "pipeline_execution_arn": execution_arn,
        "pipeline_execution_id": execution_arn.rsplit("/", maxsplit=1)[-1],
        "started_at": utc_now(),
        "status": "Executing",
    }
    write_json_atomic(local_root / "execution.json", started)
    put_json_s3(started, bucket=config.bucket, key=f"{control_prefix}/execution.json")
    put_json_s3(started, bucket=config.bucket, key=latest_key)
    print(json.dumps(started, indent=2, sort_keys=True))
    print("OTTO_TWO_TOWER_FOLD_STARTED_SAFE_TO_DISCONNECT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
