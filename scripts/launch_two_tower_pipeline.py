from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from otto_recsys.cloud.sagemaker_pipeline import (
    ManagedResumeProofConfig,
    build_pipeline_definition,
    canonical_sha256,
    create_deterministic_source_archive,
    official_pytorch_image,
    pipeline_name,
    run_contract_payload,
    run_s3_prefix,
    source_s3_uri,
    verify_source_archive,
)
from otto_recsys.cloud.sagemaker_two_tower import derive_role_name_from_sts_arn


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_command(
    command: list[str], *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


def aws_json(arguments: list[str]) -> dict[str, Any]:
    completed = run_command(["aws", *arguments, "--output", "json"], check=True)
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("AWS CLI returned a non-object JSON payload")
    return payload


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


def _run_preflight_stage(name: str, command: list[str]) -> None:
    started = time.perf_counter()
    print(f"[{utc_now()}] source_preflight_stage_start name={name}", flush=True)
    completed = run_command(command)
    elapsed = time.perf_counter() - started
    if completed.stdout.strip():
        print(completed.stdout.rstrip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"source preflight stage failed: {name} rc={completed.returncode}"
        )
    print(
        f"[{utc_now()}] source_preflight_stage_complete "
        f"name={name} status=passed elapsed_seconds={elapsed:.3f}",
        flush=True,
    )


def run_exact_source_preflight(source_root: Path) -> None:
    """Gate the exact GPU source tree before any SageMaker work is submitted."""
    started = time.perf_counter()
    config_path = source_root / "pyproject.toml"
    if not config_path.is_file():
        raise RuntimeError(f"missing GPU package config: {config_path}")

    _run_preflight_stage(
        "compile",
        [sys.executable, "-m", "compileall", "-q", str(source_root)],
    )
    _run_preflight_stage(
        "ruff",
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--config",
            str(config_path),
            str(source_root),
        ],
    )
    _run_preflight_stage(
        "mypy",
        [
            "uv",
            "run",
            "mypy",
            "--config-file",
            str(config_path),
            "--python-version",
            "3.13",
            str(source_root / "otto_two_tower"),
            str(source_root / "train.py"),
            str(source_root / "prepare.py"),
            str(source_root / "runtime_validation.py"),
            str(source_root / "sagemaker_entrypoint.py"),
        ],
    )
    _run_preflight_stage(
        "cpu_safe_contract_tests",
        [
            "env",
            f"PYTHONPATH={source_root}",
            "uv",
            "run",
            "pytest",
            "-q",
            str(source_root / "tests/test_resume_contract.py"),
            str(source_root / "tests/test_sagemaker_entrypoint.py"),
        ],
    )
    elapsed = time.perf_counter() - started
    print(
        f"[{utc_now()}] source_preflight_complete status=passed "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_uploaded_source_roundtrip(
    *, source_root: Path, source_uri: str, expected_sha256: str
) -> dict[str, Any]:
    """Download the exact S3 source object and prove byte/content parity."""
    with tempfile.TemporaryDirectory(prefix="otto-source-roundtrip-") as tmpdir:
        downloaded = Path(tmpdir) / "source.tar.gz"
        run_command(
            ["aws", "s3", "cp", source_uri, str(downloaded), "--only-show-errors"],
            check=True,
        )
        observed_sha256 = _sha256_file(downloaded)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                "S3 source round-trip SHA-256 mismatch: "
                f"expected={expected_sha256} observed={observed_sha256}"
            )
        verification = verify_source_archive(source_root, downloaded)
    return {
        **verification,
        "archive_sha256": observed_sha256,
        "s3_roundtrip": "passed",
    }


def head_s3(bucket: str, key: str) -> dict[str, Any] | None:
    completed = run_command(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json"]
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, dict) else None


def s3_json(bucket: str, key: str) -> dict[str, Any]:
    completed = run_command(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", "-", "--only-show-errors"],
        check=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"s3://{bucket}/{key} is not a JSON object")
    return payload


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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pipeline_exists(name: str) -> bool:
    completed = run_command(
        ["aws", "sagemaker", "describe-pipeline", "--pipeline-name", name, "--output", "json"]
    )
    return completed.returncode == 0


def register_pipeline(
    *, name: str, role_arn: str, definition_path: Path, run_id: str, commit: str
) -> dict[str, Any]:
    common = [
        "--pipeline-name",
        name,
        "--pipeline-definition",
        f"file://{definition_path}",
        "--pipeline-description",
        f"OTTO two-tower managed resume proof run_id={run_id}",
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
            "Key=Purpose,Value=resume-proof",
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
        if not isinstance(summary, dict):
            continue
        if summary.get("PipelineExecutionStatus") in {"Executing", "Stopping"}:
            return summary
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("OTTO_BUCKET"))
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--role-arn")
    parser.add_argument("--instance-type", default="ml.g6.xlarge")
    parser.add_argument("--volume-size-gb", type=int, default=100)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--valid-rows", type=int, default=10_000)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--job-a-stop-step", type=int, default=40)
    parser.add_argument("--job-b-stop-step", type=int, default=80)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-runtime-seconds", type=int, default=1_800)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise RuntimeError("--bucket or OTTO_BUCKET is required")
    config = ManagedResumeProofConfig(
        bucket=args.bucket,
        region=args.region,
        instance_type=args.instance_type,
        volume_size_gb=args.volume_size_gb,
        validation_fold=args.validation_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        train_rows=args.train_rows,
        valid_rows=args.valid_rows,
        checkpoint_steps=args.checkpoint_steps,
        job_a_stop_step=args.job_a_stop_step,
        job_b_stop_step=args.job_b_stop_step,
        heartbeat_seconds=args.heartbeat_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    config.validate()
    commit = ensure_committed_remote_state()
    role_arn = resolve_role_arn(args.role_arn)
    image_uri = official_pytorch_image(config.region)

    source_root = Path("gpu/two_tower")
    run_exact_source_preflight(source_root)

    source_archive = Path("artifacts/two_tower_pipeline/source.tar.gz")
    source_sha = create_deterministic_source_archive(source_root, source_archive)
    local_source_verification = verify_source_archive(source_root, source_archive)
    print(
        f"[{utc_now()}] source_archive_verified files={local_source_verification['files']} "
        f"manifest_sha256={local_source_verification['manifest_sha256']} "
        f"archive_sha256={source_sha}",
        flush=True,
    )

    source_uri = source_s3_uri(config.bucket, commit, source_sha)
    run_command(
        ["aws", "s3", "cp", str(source_archive), source_uri, "--only-show-errors"],
        check=True,
    )

    source_key = source_uri.split(f"s3://{config.bucket}/", maxsplit=1)[1]
    source_head = head_s3(config.bucket, source_key)
    if source_head is None or int(source_head.get("ContentLength", 0)) <= 0:
        raise RuntimeError("deterministic source archive was not durably verified in S3")

    remote_source_verification = verify_uploaded_source_roundtrip(
        source_root=source_root,
        source_uri=source_uri,
        expected_sha256=source_sha,
    )
    print(
        f"[{utc_now()}] source_s3_roundtrip_verified "
        f"files={remote_source_verification['files']} "
        f"archive_sha256={remote_source_verification['archive_sha256']}",
        flush=True,
    )

    manifest_keys = {
        "ranking": "candidates/ranking-training-cache/manifest.json",
        "hard_negatives": "candidates/hard-negatives/manifest.json",
        "items": "retrieval/two-tower/items/manifest.json",
    }
    input_manifests: dict[str, str] = {}
    for name, key in manifest_keys.items():
        payload = s3_json(config.bucket, key)
        input_manifests[name] = manifest_identity(payload)

    item_vectors = head_s3(config.bucket, "retrieval/two-tower/items/item_vectors.npy")
    if item_vectors is None or int(item_vectors.get("ContentLength", 0)) <= 0:
        raise RuntimeError("durable Item2Vec initialization vectors are missing from S3")

    contract = run_contract_payload(
        commit=commit,
        source_sha256=source_sha,
        source_uri=source_uri,
        image_uri=image_uri,
        role_arn=role_arn,
        config=config,
        input_manifests=input_manifests,
    )
    run_id = canonical_sha256(contract)
    name = pipeline_name(run_id)
    run_prefix = run_s3_prefix(config.bucket, run_id)
    control_key_prefix = f"retrieval/two-tower/runs/resume-proof/{run_id}/control"
    proof_key = f"retrieval/two-tower/runs/resume-proof/{run_id}/checkpoints/resume_proof.json"

    existing_proof = None
    proof_head = head_s3(config.bucket, proof_key)
    if proof_head is not None:
        existing_proof = s3_json(config.bucket, proof_key)
    if (
        existing_proof is not None
        and existing_proof.get("status") == "passed"
        and not args.force
    ):
        print(json.dumps(existing_proof, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_MANAGED_RESUME_PROOF_ALREADY_PASSED")
        return 0

    definition = build_pipeline_definition(
        role_arn=role_arn,
        image_uri=image_uri,
        source_uri=source_uri,
        commit=commit,
        run_id=run_id,
        config=config,
    )
    local_root = Path("artifacts/two_tower_pipeline") / run_id
    definition_path = local_root / "pipeline_definition.json"
    manifest_path = local_root / "run_manifest.json"
    write_json_atomic(definition_path, definition)
    run_manifest = {
        "status": "registered" if not args.start else "starting",
        "created_at": utc_now(),
        "pipeline_name": name,
        "run_id": run_id,
        "code_commit": commit,
        "source_sha256": source_sha,
        "source_s3_uri": source_uri,
        "source_verification": {
            "local": local_source_verification,
            "s3_roundtrip": remote_source_verification,
        },
        "checkpoint_s3_uri": f"{run_prefix}checkpoints/",
        "pipeline_definition_s3_uri": (
            f"s3://{config.bucket}/{control_key_prefix}/pipeline_definition.json"
        ),
        "role_arn": role_arn,
        "image_uri": image_uri,
        "config": config.__dict__,
        "input_manifests": input_manifests,
    }
    write_json_atomic(manifest_path, run_manifest)
    put_json_s3(
        definition,
        bucket=config.bucket,
        key=f"{control_key_prefix}/pipeline_definition.json",
    )
    put_json_s3(run_manifest, bucket=config.bucket, key=f"{control_key_prefix}/run_manifest.json")

    registration = register_pipeline(
        name=name,
        role_arn=role_arn,
        definition_path=definition_path,
        run_id=run_id,
        commit=commit,
    )
    print(
        f"[{utc_now()}] pipeline_registered name={name} "
        f"arn={registration.get('PipelineArn')}",
        flush=True,
    )

    latest = {
        "pipeline_name": name,
        "run_id": run_id,
        "code_commit": commit,
        "run_manifest_s3_uri": f"s3://{config.bucket}/{control_key_prefix}/run_manifest.json",
        "checkpoint_s3_uri": f"{run_prefix}checkpoints/",
    }
    put_json_s3(
        latest,
        bucket=config.bucket,
        key="retrieval/two-tower/pipelines/resume-proof/latest.json",
    )
    write_json_atomic(Path("artifacts/two_tower_pipeline/latest.json"), latest)

    if not args.start:
        print(json.dumps(latest, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_PIPELINE_REGISTERED_NO_GPU_STARTED")
        return 0

    current = active_execution(name)
    if current is not None and not args.force:
        print(json.dumps(current, indent=2, sort_keys=True))
        print("OTTO_TWO_TOWER_PIPELINE_ALREADY_RUNNING_SAFE_TO_DISCONNECT")
        return 0

    execution = aws_json(
        [
            "sagemaker",
            "start-pipeline-execution",
            "--pipeline-name",
            name,
            "--pipeline-execution-display-name",
            f"resume-proof-{run_id[:12]}",
        ]
    )
    execution_arn = str(execution["PipelineExecutionArn"])
    start_manifest = {
        **latest,
        "started_at": utc_now(),
        "pipeline_execution_arn": execution_arn,
        "status": "Executing",
    }
    write_json_atomic(local_root / "execution.json", start_manifest)
    put_json_s3(start_manifest, bucket=config.bucket, key=f"{control_key_prefix}/execution.json")
    put_json_s3(
        start_manifest,
        bucket=config.bucket,
        key="retrieval/two-tower/pipelines/resume-proof/latest.json",
    )
    print(json.dumps(start_manifest, indent=2, sort_keys=True))
    print("OTTO_TWO_TOWER_PIPELINE_STARTED_SAFE_TO_DISCONNECT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
