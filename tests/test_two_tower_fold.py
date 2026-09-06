from __future__ import annotations

from pathlib import Path

from otto_recsys.cloud.two_tower_fold import (
    FoldTrainingConfig,
    build_fold_pipeline_definition,
    fold_pipeline_name,
    fold_run_contract,
    fold_run_id,
    fold_run_prefix,
    load_fold_config,
)


def _config() -> FoldTrainingConfig:
    return FoldTrainingConfig(bucket="otto-test-bucket")


def test_fold_config_loads_canonical_profile() -> None:
    config = load_fold_config(
        Path("configs/two_tower.toml"), profile="fold0", bucket="otto-test-bucket"
    )
    assert config.validation_fold == 0
    assert config.instance_type == "ml.g6.xlarge"
    assert config.batch_size == 256
    assert config.checkpoint_steps == 4000


def test_fold_config_rejects_invalid_fold() -> None:
    config = FoldTrainingConfig(bucket="otto-test-bucket", validation_fold=5)
    try:
        config.validate()
    except ValueError as exc:
        assert "validation_fold" in str(exc)
    else:
        raise AssertionError("invalid fold must fail")


def test_fold_pipeline_name_and_prefix_are_deterministic() -> None:
    run_id = "a" * 64
    assert fold_pipeline_name(0, run_id) == "otto-two-tower-fold-0-" + "a" * 24
    assert (
        fold_run_prefix("otto-test-bucket", 0, run_id)
        == f"s3://otto-test-bucket/retrieval/two-tower/runs/folds/fold-0/{run_id}/"
    )


def test_fold_definition_is_single_resumable_training_step() -> None:
    definition = build_fold_pipeline_definition(
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch:test",
        source_uri="s3://otto-test-bucket/source/source.tar.gz",
        commit="b" * 40,
        run_id="c" * 64,
        config=_config(),
    )
    steps = definition["Steps"]
    assert len(steps) == 1
    assert steps[0]["Name"] == "TrainFold"
    arguments = steps[0]["Arguments"]
    hyperparameters = arguments["HyperParameters"]
    assert hyperparameters["resume-if-available"] == "true"
    assert "stop-after-step" not in hyperparameters
    assert "train-rows" not in hyperparameters
    assert "valid-rows" not in hyperparameters
    assert arguments["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints"


def test_fold_definition_uses_frozen_input_channels() -> None:
    definition = build_fold_pipeline_definition(
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="image",
        source_uri="s3://otto-test-bucket/source/source.tar.gz",
        commit="b" * 40,
        run_id="c" * 64,
        config=_config(),
    )
    channels = definition["Steps"][0]["Arguments"]["InputDataConfig"]
    by_name = {row["ChannelName"]: row for row in channels}
    assert set(by_name) == {"ranking", "hard-negatives", "items"}
    assert by_name["ranking"]["DataSource"]["S3DataSource"]["S3Uri"].endswith(
        "/candidates/ranking-training-cache/"
    )


def test_fold_run_id_changes_with_source_or_config() -> None:
    base = fold_run_contract(
        commit="a" * 40,
        source_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        source_uri="s3://bucket/source.tar.gz",
        image_uri="image",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        config=_config(),
        input_manifests={"ranking": "d" * 64},
    )
    changed = dict(base)
    changed["source_sha256"] = "e" * 64
    assert fold_run_id(base) != fold_run_id(changed)
    assert fold_run_id(base) == fold_run_id(dict(base))
