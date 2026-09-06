from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from otto_recsys.cloud.sagemaker_pipeline import (
    CHECKPOINT_LOCAL_PATH,
    PIPELINE_VERSION,
    canonical_sha256,
    channel_uris,
    retry_policies,
    validate_pipeline_metadata,
)

_PIPELINE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,255}$")


@dataclass(frozen=True)
class FoldTrainingConfig:
    bucket: str
    region: str = "us-west-2"
    instance_type: str = "ml.g6.xlarge"
    volume_size_gb: int = 100
    validation_fold: int = 0
    epochs: int = 8
    batch_size: int = 256
    max_seq_len: int = 50
    checkpoint_steps: int = 4000
    heartbeat_seconds: float = 30.0
    max_runtime_seconds: int = 21_600
    seed: int = 20260905

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
            ("checkpoint_steps", self.checkpoint_steps),
            ("max_runtime_seconds", self.max_runtime_seconds),
            ("seed", self.seed),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")


def load_fold_config(path: Path, *, profile: str, bucket: str) -> FoldTrainingConfig:
    import tomllib

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = payload.get(profile)
    if not isinstance(raw, dict):
        raise ValueError(f"missing [{profile}] configuration in {path}")
    config = FoldTrainingConfig(
        bucket=bucket,
        region=str(raw.get("region", "us-west-2")),
        instance_type=str(raw.get("instance_type", "ml.g6.xlarge")),
        volume_size_gb=int(raw.get("volume_size_gb", 100)),
        validation_fold=int(raw.get("validation_fold", 0)),
        epochs=int(raw.get("epochs", 8)),
        batch_size=int(raw.get("batch_size", 256)),
        max_seq_len=int(raw.get("max_seq_len", 50)),
        checkpoint_steps=int(raw.get("checkpoint_steps", 4000)),
        heartbeat_seconds=float(raw.get("heartbeat_seconds", 30.0)),
        max_runtime_seconds=int(raw.get("max_runtime_seconds", 21_600)),
        seed=int(raw.get("seed", 20260905)),
    )
    config.validate()
    return config


def fold_run_prefix(bucket: str, validation_fold: int, run_id: str) -> str:
    return (
        f"s3://{bucket}/retrieval/two-tower/runs/folds/"
        f"fold-{validation_fold}/{run_id}/"
    )


def fold_pipeline_name(validation_fold: int, run_id: str) -> str:
    candidate = f"otto-two-tower-fold-{validation_fold}-{run_id[:24]}"
    if not _PIPELINE_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid SageMaker pipeline name: {candidate}")
    return candidate


def fold_hyperparameters(
    *, source_uri: str, commit: str, config: FoldTrainingConfig
) -> dict[str, str]:
    config.validate()
    return {
        "sagemaker_program": "sagemaker_entrypoint.py",
        "sagemaker_submit_directory": source_uri,
        "code-commit": commit,
        "validation-fold": str(config.validation_fold),
        "epochs": str(config.epochs),
        "batch-size": str(config.batch_size),
        "max-seq-len": str(config.max_seq_len),
        "checkpoint-steps": str(config.checkpoint_steps),
        "heartbeat-seconds": str(config.heartbeat_seconds),
        "seed": str(config.seed),
        "resume-if-available": "true",
    }


def build_fold_pipeline_definition(
    *,
    role_arn: str,
    image_uri: str,
    source_uri: str,
    commit: str,
    run_id: str,
    config: FoldTrainingConfig,
) -> dict[str, Any]:
    config.validate()
    run_prefix = fold_run_prefix(config.bucket, config.validation_fold, run_id)
    checkpoint_uri = f"{run_prefix}checkpoints/"
    output_uri = f"{run_prefix}output/"

    input_data_config = [
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
        for channel_name, s3_uri in channel_uris(config.bucket).items()
    ]

    training_arguments = {
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
        },
        "RoleArn": role_arn,
        "InputDataConfig": input_data_config,
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
        "HyperParameters": fold_hyperparameters(
            source_uri=source_uri,
            commit=commit,
            config=config,
        ),
        "Environment": {
            "OTTO_CODE_COMMIT": commit,
            "OTTO_RUN_ID": run_id,
            "OTTO_MODE": "fold-training",
            "PYTHONUNBUFFERED": "1",
        },
        "Tags": [
            {"Key": "Project", "Value": "otto-recommender-system"},
            {"Key": "Model", "Value": "two-tower"},
            {"Key": "Purpose", "Value": "fold-training"},
            {"Key": "ValidationFold", "Value": str(config.validation_fold)},
            {"Key": "CodeCommit", "Value": commit[:40]},
            {"Key": "RunId", "Value": run_id[:40]},
        ],
    }

    definition = {
        "Version": PIPELINE_VERSION,
        "Metadata": {
            "Project": "otto-recommender-system",
            "Purpose": "two-tower-fold-training",
            "RunId": run_id,
            "CodeCommit": commit,
            "ValidationFold": str(config.validation_fold),
        },
        "Parameters": [],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [
            {
                "Name": "TrainFold",
                "Type": "Training",
                "Arguments": training_arguments,
                "RetryPolicies": retry_policies(),
            }
        ],
    }
    validate_pipeline_metadata(definition)
    return definition


def fold_run_contract(
    *,
    commit: str,
    source_sha256: str,
    source_manifest_sha256: str,
    source_uri: str,
    image_uri: str,
    role_arn: str,
    config: FoldTrainingConfig,
    input_manifests: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "two-tower-fold-training",
        "code_commit": commit,
        "source_sha256": source_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_s3_uri": source_uri,
        "image_uri": image_uri,
        "role_arn": role_arn,
        "config": asdict(config),
        "input_manifests": dict(sorted(input_manifests.items())),
        "channels": channel_uris(config.bucket),
    }


def fold_run_id(contract: dict[str, Any]) -> str:
    return canonical_sha256(contract)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
