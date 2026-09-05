from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DLC_ACCOUNT_ID = "763104351884"
DLC_TAG = "2.13.0-cu133-amzn2023-sagemaker"
CHECKPOINT_LOCAL_PATH = "/opt/ml/checkpoints"

_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")


@dataclass(frozen=True)
class ResumeProofConfig:
    bucket: str
    region: str = "us-west-2"
    instance_type: str = "ml.g6.xlarge"
    volume_size_gb: int = 100
    validation_fold: int = 0
    epochs: int = 8
    batch_size: int = 256
    max_seq_len: int = 50
    train_rows: int = 100_000
    valid_rows: int = 10_000
    checkpoint_steps: int = 20
    heartbeat_seconds: float = 30.0
    max_runtime_seconds: int = 1_800
    poll_seconds: int = 20

    def validate(self) -> None:
        if not self.bucket:
            raise ValueError("bucket is required")
        if not self.region:
            raise ValueError("region is required")
        if not self.instance_type.startswith("ml."):
            raise ValueError("instance_type must be a SageMaker ml.* instance")
        if self.volume_size_gb < 30:
            raise ValueError("volume_size_gb must be at least 30")
        if not 0 <= self.validation_fold < 5:
            raise ValueError("validation_fold must be in [0, 4]")
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("max_seq_len", self.max_seq_len),
            ("train_rows", self.train_rows),
            ("valid_rows", self.valid_rows),
            ("checkpoint_steps", self.checkpoint_steps),
            ("max_runtime_seconds", self.max_runtime_seconds),
            ("poll_seconds", self.poll_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")


def official_pytorch_image(region: str) -> str:
    if not region:
        raise ValueError("region is required")
    return (
        f"{DLC_ACCOUNT_ID}.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch:{DLC_TAG}"
    )


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_s3_prefix(bucket: str, commit: str, source_sha256: str) -> str:
    return (
        f"s3://{bucket}/retrieval/two-tower/source/"
        f"{commit}/{source_sha256}/"
    )


def run_s3_prefix(bucket: str, run_id: str) -> str:
    return f"s3://{bucket}/retrieval/two-tower/runs/resume-proof/{run_id}/"


def channel_uris(bucket: str) -> dict[str, str]:
    return {
        "ranking": f"s3://{bucket}/candidates/ranking-training-cache/",
        "hard-negatives": f"s3://{bucket}/candidates/hard-negatives/",
        "items": f"s3://{bucket}/retrieval/two-tower/items/",
    }


def job_name(prefix: str, run_id: str, suffix: str) -> str:
    candidate = f"{prefix}-{run_id[:12]}-{suffix}".lower()
    if not _JOB_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid SageMaker job name: {candidate}")
    return candidate


def entrypoint_command(
    *,
    commit: str,
    config: ResumeProofConfig,
    resume: bool,
) -> str:
    arguments = [
        "python -u sagemaker_entrypoint.py",
        f"--code-commit {commit}",
        f"--validation-fold {config.validation_fold}",
        f"--epochs {config.epochs}",
        f"--batch-size {config.batch_size}",
        f"--max-seq-len {config.max_seq_len}",
        f"--train-rows {config.train_rows}",
        f"--valid-rows {config.valid_rows}",
        f"--checkpoint-steps {config.checkpoint_steps}",
        f"--heartbeat-seconds {config.heartbeat_seconds}",
    ]
    if resume:
        arguments.append("--resume")
    training_command = " ".join(arguments)
    return (
        "mkdir -p /opt/ml/code && "
        "tar -xzf /opt/ml/input/data/source/source.tar.gz -C /opt/ml/code && "
        "cd /opt/ml/code && "
        f"{training_command}"
    )


def build_training_request(
    *,
    job_name_value: str,
    role_arn: str,
    image_uri: str,
    source_prefix: str,
    checkpoint_prefix: str,
    output_prefix: str,
    commit: str,
    run_id: str,
    config: ResumeProofConfig,
    resume: bool,
) -> dict[str, Any]:
    config.validate()
    if not _JOB_NAME_PATTERN.fullmatch(job_name_value):
        raise ValueError(f"invalid SageMaker job name: {job_name_value}")
    if not role_arn.startswith("arn:aws"):
        raise ValueError("role_arn must be an AWS ARN")
    if not image_uri:
        raise ValueError("image_uri is required")

    channels = channel_uris(config.bucket)
    inputs = [
        {
            "ChannelName": "source",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": source_prefix,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    ]
    for channel_name, s3_uri in channels.items():
        inputs.append(
            {
                "ChannelName": channel_name,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        )

    return {
        "TrainingJobName": job_name_value,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
            "ContainerEntrypoint": ["bash", "-lc"],
            "ContainerArguments": [
                entrypoint_command(commit=commit, config=config, resume=resume)
            ],
        },
        "RoleArn": role_arn,
        "InputDataConfig": inputs,
        "OutputDataConfig": {"S3OutputPath": output_prefix},
        "ResourceConfig": {
            "InstanceType": config.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": config.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": config.max_runtime_seconds},
        "CheckpointConfig": {
            "S3Uri": checkpoint_prefix,
            "LocalPath": CHECKPOINT_LOCAL_PATH,
        },
        "EnableManagedSpotTraining": False,
        "Environment": {
            "OTTO_CODE_COMMIT": commit,
            "OTTO_RUN_ID": run_id,
            "PYTHONUNBUFFERED": "1",
        },
        "Tags": [
            {"Key": "Project", "Value": "otto-recommender-system"},
            {"Key": "Model", "Value": "two-tower"},
            {"Key": "Purpose", "Value": "resume-proof"},
            {"Key": "CodeCommit", "Value": commit[:40]},
            {"Key": "RunId", "Value": run_id[:40]},
            {"Key": "ResumeMode", "Value": "resume" if resume else "fresh"},
        ],
    }


def resume_proof_payload(
    *,
    commit: str,
    source_sha256: str,
    image_uri: str,
    config: ResumeProofConfig,
) -> dict[str, Any]:
    return {
        "code_commit": commit,
        "source_sha256": source_sha256,
        "image_uri": image_uri,
        "config": asdict(config),
        "channels": channel_uris(config.bucket),
    }


def derive_role_name_from_sts_arn(arn: str) -> str | None:
    marker = ":assumed-role/"
    if marker not in arn:
        return None
    resource = arn.split(marker, maxsplit=1)[1]
    role_name = resource.split("/", maxsplit=1)[0]
    return role_name or None
