from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DLC_ACCOUNT_ID = "763104351884"
DLC_TAG = "2.13.0-cu133-amzn2023-sagemaker"
CHECKPOINT_LOCAL_PATH = "/opt/ml/checkpoints"
PIPELINE_VERSION = "2020-12-01"

_PIPELINE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,255}$")


_SOURCE_ARCHIVE_IGNORED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def _source_files(source_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
        and path.suffix != ".pyc"
        and not any(
            part in _SOURCE_ARCHIVE_IGNORED_DIRS
            for part in path.relative_to(source_root).parts
        )
    ]


def create_deterministic_source_archive(source_root: Path, destination: Path) -> str:
    """Create a byte-for-byte reproducible gzip-compressed source archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in _source_files(source_root):
            relative = path.relative_to(source_root)
            info = archive.gettarinfo(str(path), arcname=str(relative))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    payload = buffer.getvalue()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ManagedResumeProofConfig:
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
    job_a_stop_step: int = 40
    job_b_stop_step: int = 80
    heartbeat_seconds: float = 30.0
    max_runtime_seconds: int = 1_800

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
            ("job_a_stop_step", self.job_a_stop_step),
            ("job_b_stop_step", self.job_b_stop_step),
            ("max_runtime_seconds", self.max_runtime_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if self.job_a_stop_step % self.checkpoint_steps != 0:
            raise ValueError("job_a_stop_step must align to checkpoint_steps")
        if self.job_b_stop_step % self.checkpoint_steps != 0:
            raise ValueError("job_b_stop_step must align to checkpoint_steps")
        if self.job_b_stop_step <= self.job_a_stop_step:
            raise ValueError("job_b_stop_step must be greater than job_a_stop_step")


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


def source_s3_uri(bucket: str, commit: str, source_sha256: str) -> str:
    return (
        f"s3://{bucket}/retrieval/two-tower/source/"
        f"{commit}/{source_sha256}/source.tar.gz"
    )


def run_s3_prefix(bucket: str, run_id: str) -> str:
    return f"s3://{bucket}/retrieval/two-tower/runs/resume-proof/{run_id}/"


def pipeline_name(run_id: str) -> str:
    candidate = f"otto-two-tower-resume-proof-{run_id[:24]}"
    if not _PIPELINE_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid SageMaker pipeline name: {candidate}")
    return candidate


def channel_uris(bucket: str) -> dict[str, str]:
    return {
        "ranking": f"s3://{bucket}/candidates/ranking-training-cache/",
        "hard-negatives": f"s3://{bucket}/candidates/hard-negatives/",
        "items": f"s3://{bucket}/retrieval/two-tower/items/",
    }


def script_mode_hyperparameters(
    *,
    source_uri: str,
    commit: str,
    config: ManagedResumeProofConfig,
    resume: bool,
    resume_if_available: bool,
    stop_after_step: int,
) -> dict[str, str]:
    config.validate()
    payload = {
        "sagemaker_program": "sagemaker_entrypoint.py",
        "sagemaker_submit_directory": source_uri,
        "code-commit": commit,
        "validation-fold": str(config.validation_fold),
        "epochs": str(config.epochs),
        "batch-size": str(config.batch_size),
        "max-seq-len": str(config.max_seq_len),
        "train-rows": str(config.train_rows),
        "valid-rows": str(config.valid_rows),
        "checkpoint-steps": str(config.checkpoint_steps),
        "heartbeat-seconds": str(config.heartbeat_seconds),
        "stop-after-step": str(stop_after_step),
    }
    if resume:
        payload["resume"] = "true"
    if resume_if_available:
        payload["resume-if-available"] = "true"
    return payload


def _input_data_config(bucket: str) -> list[dict[str, Any]]:
    return [
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
        for channel_name, s3_uri in channel_uris(bucket).items()
    ]


def training_arguments(
    *,
    role_arn: str,
    image_uri: str,
    source_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    commit: str,
    run_id: str,
    config: ManagedResumeProofConfig,
    resume: bool,
    resume_if_available: bool,
    stop_after_step: int,
) -> dict[str, Any]:
    config.validate()
    if not role_arn.startswith("arn:aws"):
        raise ValueError("role_arn must be an AWS ARN")
    if not image_uri:
        raise ValueError("image_uri is required")
    return {
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
        },
        "RoleArn": role_arn,
        "InputDataConfig": _input_data_config(config.bucket),
        "OutputDataConfig": {"S3OutputPath": output_uri},
        "ResourceConfig": {
            "InstanceType": config.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": config.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": config.max_runtime_seconds},
        "CheckpointConfig": {
            "S3Uri": checkpoint_uri,
            "LocalPath": CHECKPOINT_LOCAL_PATH,
        },
        "HyperParameters": script_mode_hyperparameters(
            source_uri=source_uri,
            commit=commit,
            config=config,
            resume=resume,
            resume_if_available=resume_if_available,
            stop_after_step=stop_after_step,
        ),
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


def retry_policies() -> list[dict[str, Any]]:
    return [
        {
            "ExceptionType": "SageMaker.CAPACITY_ERROR",
            "IntervalSeconds": 60,
            "BackoffRate": 2.0,
            "MaxAttempts": 3,
        },
        {
            "ExceptionType": "Step.SERVICE_FAULT",
            "IntervalSeconds": 30,
            "BackoffRate": 2.0,
            "MaxAttempts": 3,
        },
        {
            "ExceptionType": "SageMaker.JOB_INTERNAL_ERROR",
            "IntervalSeconds": 30,
            "BackoffRate": 2.0,
            "MaxAttempts": 3,
        },
    ]


def build_pipeline_definition(
    *,
    role_arn: str,
    image_uri: str,
    source_uri: str,
    commit: str,
    run_id: str,
    config: ManagedResumeProofConfig,
) -> dict[str, Any]:
    config.validate()
    run_prefix = run_s3_prefix(config.bucket, run_id)
    checkpoint_uri = f"{run_prefix}checkpoints/"
    job_a = training_arguments(
        role_arn=role_arn,
        image_uri=image_uri,
        source_uri=source_uri,
        checkpoint_uri=checkpoint_uri,
        output_uri=f"{run_prefix}output/job-a/",
        commit=commit,
        run_id=run_id,
        config=config,
        resume=False,
        resume_if_available=True,
        stop_after_step=config.job_a_stop_step,
    )
    job_b = training_arguments(
        role_arn=role_arn,
        image_uri=image_uri,
        source_uri=source_uri,
        checkpoint_uri=checkpoint_uri,
        output_uri=f"{run_prefix}output/job-b/",
        commit=commit,
        run_id=run_id,
        config=config,
        resume=True,
        resume_if_available=False,
        stop_after_step=config.job_b_stop_step,
    )
    return {
        "Version": PIPELINE_VERSION,
        "Metadata": {
            "Project": "otto-recommender-system",
            "Purpose": "two-tower-resume-proof",
            "RunId": run_id,
            "CodeCommit": commit,
        },
        "Parameters": [],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [
            {
                "Name": "CreateDurableCheckpoint",
                "Type": "Training",
                "Arguments": job_a,
                "RetryPolicies": retry_policies(),
            },
            {
                "Name": "ResumeAndAdvance",
                "Type": "Training",
                "DependsOn": ["CreateDurableCheckpoint"],
                "Arguments": job_b,
                "RetryPolicies": retry_policies(),
            },
        ],
    }


def run_contract_payload(
    *,
    commit: str,
    source_sha256: str,
    source_uri: str,
    image_uri: str,
    role_arn: str,
    config: ManagedResumeProofConfig,
    input_manifests: dict[str, str],
) -> dict[str, Any]:
    return {
        "code_commit": commit,
        "source_sha256": source_sha256,
        "source_s3_uri": source_uri,
        "image_uri": image_uri,
        "role_arn": role_arn,
        "config": asdict(config),
        "input_manifests": input_manifests,
        "channels": channel_uris(config.bucket),
    }
