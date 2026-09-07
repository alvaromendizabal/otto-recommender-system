"""Source and managed execution contracts for the ANN experiment."""

from __future__ import annotations

import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

from otto_recsys.cloud.sagemaker_pipeline import source_tree_manifest, validate_pipeline_metadata
from otto_recsys.cloud.two_tower_evaluation import evaluation_definition


def same_ann_experiment(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow a reporting commit to reuse identical worker bytes and inputs.

    The original run identity and commit stay intact. Every contract field other
    than the enclosing repository commit must match, including the full source
    archive hash, trained model, exact reference, runtime, and search settings.
    """
    required = {
        "code_commit",
        "source_sha256",
        "training_run_id",
        "reference_run_id",
        "reference_input_id",
        "reference_manifest_sha256",
        "training_definition",
        "max_runtime_seconds",
        "sample_sessions",
        "parameters",
    }
    for contract in (previous, current):
        if not required <= contract.keys():
            raise ValueError("incomplete ANN experiment contract")
        if not re.fullmatch(r"[a-f0-9]{64}", str(contract["source_sha256"])):
            raise ValueError("invalid ANN source archive checksum")
    return {k: v for k, v in previous.items() if k != "code_commit"} == {
        k: v for k, v in current.items() if k != "code_commit"
    }


def load_ann_parameters(path: Path) -> dict[str, str]:
    allowed = {
        "sample_sessions",
        "nlist",
        "train_items",
        "train_iterations",
        "probes",
        "candidate_depth",
        "target_overlap",
        "threads",
        "batch_size",
        "index_shard_rows",
        "latency_queries",
        "latency_repeats",
        "warmup_queries",
        "export_fold_predictions",
        "seed",
    }
    data = tomllib.loads(path.read_text())["benchmark"]
    if set(data) != allowed:
        raise ValueError("ANN configuration must contain exactly the benchmark parameters")
    return {
        key.replace("_", "-"): ",".join(map(str, value))
        if key == "probes"
        else str(value).lower()
        if isinstance(value, bool)
        else str(value)
        for key, value in data.items()
    }


def stage_ann_source(source: Path, destination: Path) -> None:
    """Select the committed ANN dependency profile in an isolated source tree."""
    for relative in source_tree_manifest(source):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    shutil.copyfile(source / "requirements-ann.txt", destination / "requirements.txt")


def ann_definition(
    *,
    training_definition: dict[str, Any],
    bucket: str,
    training_run_id: str,
    run_id: str,
    reference_run_id: str,
    reference_input_id: str,
    reference_manifest_sha256: str,
    source_uri: str,
    commit: str,
    training_manifest: dict[str, Any],
    input_manifests: dict[str, str],
    max_runtime_seconds: int = 7200,
    sample_sessions: int = 4096,
    parameters: dict[str, str] | None = None,
    region: str = "us-west-2",
) -> dict[str, Any]:
    definition = evaluation_definition(
        training_definition=training_definition,
        bucket=bucket,
        training_run_id=training_run_id,
        evaluation_id=run_id,
        source_uri=source_uri,
        commit=commit,
        training_manifest=training_manifest,
        input_manifests=input_manifests,
        max_runtime_seconds=max_runtime_seconds,
    )
    fold = int(training_manifest["validation_fold"])
    prefix = f"s3://{bucket}/retrieval/two-tower/ann/fold-{fold}/{run_id}/"
    arguments = definition["Steps"][0]["Arguments"]
    arguments["InputDataConfig"].append(
        {
            "ChannelName": "reference",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3Uri": f"s3://{bucket}/retrieval/two-tower/evaluations/fold-{fold}/{reference_run_id}/checkpoints/",
                }
            },
        }
    )
    arguments["HyperParameters"] = {
        "sagemaker_program": "benchmark.py",
        "sagemaker_submit_directory": source_uri,
        "run-id": run_id,
        "code-commit": commit,
        "reference-input-id": reference_input_id,
        "reference-manifest-sha256": reference_manifest_sha256,
        "checkpoint-uri": prefix + "checkpoints/",
        "sample-sessions": str(sample_sessions),
        "nlist": "1024",
        "train-items": "65536",
        "train-iterations": "20",
        "probes": "32,64,128,256",
        "candidate-depth": "800",
        "target-overlap": "0.98",
        "threads": "4",
        "heartbeat-seconds": "30",
        "region": region,
    }
    if parameters is not None:
        arguments["HyperParameters"].update(parameters)
        arguments["HyperParameters"]["sample-sessions"] = str(sample_sessions)
    # The worker synchronously publishes data, then receipts. A background checkpoint
    # uploader must not race that commit order. Recovery is explicit through the SDK.
    arguments.pop("CheckpointConfig", None)
    arguments["OutputDataConfig"] = {"S3OutputPath": prefix + "output/"}
    arguments["Environment"].update(
        {
            "OTTO_MODE": "ann-benchmark",
            "OTTO_RUN_ID": run_id,
            "OTTO_INSTANCE_TYPE": arguments["ResourceConfig"]["InstanceType"],
            "OMP_NUM_THREADS": arguments["HyperParameters"]["threads"],
            "MKL_NUM_THREADS": arguments["HyperParameters"]["threads"],
            "OPENBLAS_NUM_THREADS": arguments["HyperParameters"]["threads"],
        }
    )
    arguments["Tags"] = [
        {"Key": "Project", "Value": "otto-recommender-system"},
        {"Key": "Purpose", "Value": "ann-benchmark"},
    ]
    definition["Steps"][0]["Name"] = "BenchmarkANN"
    definition["Metadata"].update({"Purpose": "ann-benchmark", "RunId": run_id})
    validate_pipeline_metadata(definition)
    return definition
