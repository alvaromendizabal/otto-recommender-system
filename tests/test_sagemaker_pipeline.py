from __future__ import annotations

from otto_recsys.cloud.sagemaker_pipeline import (
    ManagedResumeProofConfig,
    build_pipeline_definition,
    pipeline_name,
)


def _config() -> ManagedResumeProofConfig:
    return ManagedResumeProofConfig(bucket="otto-test-bucket")


def test_managed_resume_config_rejects_unaligned_boundaries() -> None:
    config = ManagedResumeProofConfig(
        bucket="otto-test-bucket",
        checkpoint_steps=20,
        job_a_stop_step=30,
        job_b_stop_step=80,
    )
    try:
        config.validate()
    except ValueError as exc:
        assert "job_a_stop_step" in str(exc)
    else:
        raise AssertionError("expected invalid stop boundary to fail")


def test_pipeline_name_is_deterministic_and_valid() -> None:
    name = pipeline_name("a" * 64)
    assert name == "otto-two-tower-resume-proof-" + "a" * 24
    assert len(name) <= 256


def test_pipeline_definition_is_serial_and_uses_one_checkpoint_prefix() -> None:
    definition = build_pipeline_definition(
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch:test",
        source_uri="s3://otto-test-bucket/source/source.tar.gz",
        commit="b" * 40,
        run_id="c" * 64,
        config=_config(),
    )
    steps = definition["Steps"]
    assert [step["Name"] for step in steps] == [
        "CreateDurableCheckpoint",
        "ResumeAndAdvance",
    ]
    assert steps[1]["DependsOn"] == ["CreateDurableCheckpoint"]

    args_a = steps[0]["Arguments"]
    args_b = steps[1]["Arguments"]
    assert args_a["CheckpointConfig"] == args_b["CheckpointConfig"]
    assert args_a["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints"
    assert args_a["HyperParameters"]["stop-after-step"] == "40"
    assert "resume" not in args_a["HyperParameters"]
    assert args_a["HyperParameters"]["resume-if-available"] == "true"
    assert args_b["HyperParameters"]["stop-after-step"] == "80"
    assert args_b["HyperParameters"]["resume"] == "true"


def test_pipeline_uses_frozen_s3_channels() -> None:
    definition = build_pipeline_definition(
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch:test",
        source_uri="s3://otto-test-bucket/source/source.tar.gz",
        commit="b" * 40,
        run_id="c" * 64,
        config=_config(),
    )
    channels = definition["Steps"][0]["Arguments"]["InputDataConfig"]
    by_name = {channel["ChannelName"]: channel for channel in channels}
    assert set(by_name) == {"ranking", "hard-negatives", "items"}
    assert (
        by_name["ranking"]["DataSource"]["S3DataSource"]["S3Uri"]
        == "s3://otto-test-bucket/candidates/ranking-training-cache/"
    )
    assert (
        by_name["hard-negatives"]["DataSource"]["S3DataSource"]["S3Uri"]
        == "s3://otto-test-bucket/candidates/hard-negatives/"
    )
    assert (
        by_name["items"]["DataSource"]["S3DataSource"]["S3Uri"]
        == "s3://otto-test-bucket/retrieval/two-tower/items/"
    )

def test_retry_policies_use_current_service_exception_type_shape() -> None:
    definition = build_pipeline_definition(
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch:test",
        source_uri="s3://otto-test-bucket/source/source.tar.gz",
        commit="b" * 40,
        run_id="c" * 64,
        config=_config(),
    )

    for step in definition["Steps"]:
        policies = step["RetryPolicies"]
        assert policies
        for policy in policies:
            exception_types = policy["ExceptionType"]
            assert isinstance(exception_types, list)
            assert exception_types
            assert all(isinstance(value, str) for value in exception_types)
