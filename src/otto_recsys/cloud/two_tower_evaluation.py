"""Managed inference using the completed fold's image and saved weights."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from otto_recsys.cloud.sagemaker_pipeline import PIPELINE_VERSION, validate_pipeline_metadata


def evaluation_definition(
    *,
    training_definition: dict[str, Any],
    bucket: str,
    training_run_id: str,
    evaluation_id: str,
    source_uri: str,
    commit: str,
    training_manifest: dict[str, Any],
    input_manifests: dict[str, str],
    max_runtime_seconds: int = 7200,
    batch_size: int = 128,
) -> dict[str, Any]:
    if max_runtime_seconds <= 0 or batch_size <= 0:
        raise ValueError("runtime and batch size must be positive")
    fold = int(training_manifest["validation_fold"])
    if not 0 <= fold < 5:
        raise ValueError("invalid validation fold")
    arguments = deepcopy(training_definition["Steps"][0]["Arguments"])
    prefix = f"s3://{bucket}/retrieval/two-tower/evaluations/fold-{fold}/{evaluation_id}/"
    training_prefix = f"s3://{bucket}/retrieval/two-tower/runs/folds/fold-{fold}/{training_run_id}/"
    arguments["InputDataConfig"] = [
        row for row in arguments["InputDataConfig"] if row["ChannelName"] in {"ranking", "items"}
    ]
    if {row["ChannelName"] for row in arguments["InputDataConfig"]} != {"ranking", "items"}:
        raise ValueError("original training definition lacks required channels")
    arguments["InputDataConfig"].append(
        {
            "ChannelName": "trained",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": training_prefix + "checkpoints/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    )
    arguments["HyperParameters"] = {
        "sagemaker_program": "evaluate.py",
        "sagemaker_submit_directory": source_uri,
        "expected-ranking-id": input_manifests["ranking"],
        "expected-items-id": input_manifests["items"],
        "training-input-id": training_manifest["input_id"],
        "code-commit": commit,
        "candidate-depth": "800",
        "batch-size": str(batch_size),
        "heartbeat-seconds": "30",
    }
    arguments["CheckpointConfig"] = {
        "LocalPath": "/opt/ml/checkpoints",
        "S3Uri": prefix + "checkpoints/",
    }
    arguments["OutputDataConfig"] = {"S3OutputPath": prefix + "output/"}
    arguments["StoppingCondition"] = {"MaxRuntimeInSeconds": max_runtime_seconds}
    arguments["ResourceConfig"]["InstanceCount"] = 1
    arguments["Environment"] = {
        "PYTHONUNBUFFERED": "1",
        "OTTO_CODE_COMMIT": commit,
        "OTTO_RUN_ID": evaluation_id,
        "OTTO_MODE": "fold-evaluation",
    }
    arguments["Tags"] = [
        {"Key": "Project", "Value": "otto-recommender-system"},
        {"Key": "Purpose", "Value": "fold-evaluation"},
    ]
    definition = {
        "Version": PIPELINE_VERSION,
        "Metadata": {
            "Purpose": "fold-evaluation",
            "ValidationFold": str(fold),
            "CodeCommit": commit,
            "RunId": evaluation_id,
        },
        "Parameters": [],
        "Steps": [{"Name": "EvaluateFold", "Type": "Training", "Arguments": arguments}],
    }
    # Deliberate retries use the same durable namespace. No automatic paid retry loop.
    validate_pipeline_metadata(definition)
    return definition
