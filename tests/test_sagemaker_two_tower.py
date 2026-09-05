from __future__ import annotations

from otto_recsys.cloud.sagemaker_two_tower import (
    ResumeProofConfig,
    build_training_request,
    canonical_sha256,
    derive_role_name_from_sts_arn,
    official_pytorch_image,
)


def test_official_pytorch_image_is_region_specific() -> None:
    assert official_pytorch_image("us-west-2") == (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
        "pytorch:2.13.0-cu133-amzn2023-sagemaker"
    )


def test_build_training_request_uses_durable_checkpoint_and_channels() -> None:
    config = ResumeProofConfig(bucket="example-bucket")
    request = build_training_request(
        job_name_value="otto-tt-resume-abc-a",
        role_arn="arn:aws:iam::123456789012:role/example",
        image_uri=official_pytorch_image("us-west-2"),
        source_prefix="s3://example-bucket/source/",
        checkpoint_prefix="s3://example-bucket/checkpoints/",
        output_prefix="s3://example-bucket/output/",
        commit="abc123",
        run_id="1234567890abcdef",
        config=config,
        resume=True,
    )
    assert request["CheckpointConfig"] == {
        "S3Uri": "s3://example-bucket/checkpoints/",
        "LocalPath": "/opt/ml/checkpoints",
    }
    channels = {item["ChannelName"] for item in request["InputDataConfig"]}
    assert channels == {"source", "ranking", "hard-negatives", "items"}
    command = request["AlgorithmSpecification"]["ContainerArguments"][0]
    assert "--resume" in command
    assert "--code-commit abc123" in command


def test_role_name_is_derived_from_assumed_role_arn() -> None:
    arn = "arn:aws:sts::123456789012:assumed-role/SageMakerExecutionRole/session"
    assert derive_role_name_from_sts_arn(arn) == "SageMakerExecutionRole"
    assert derive_role_name_from_sts_arn("arn:aws:iam::123456789012:user/example") is None


def test_canonical_sha256_is_order_independent() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
