from __future__ import annotations

from otto_recsys.cloud.two_tower_evaluation import evaluation_definition
from otto_recsys.cloud.two_tower_fold import FoldTrainingConfig, build_fold_pipeline_definition


def test_evaluation_reuses_image_and_weights_without_training_hyperparameters() -> None:
    original = build_fold_pipeline_definition(
        role_arn="role",
        image_uri="proven-image",
        source_uri="old",
        commit="a" * 40,
        run_id="b" * 64,
        config=FoldTrainingConfig(bucket="bucket"),
    )
    result = evaluation_definition(
        training_definition=original,
        bucket="bucket",
        training_run_id="b" * 64,
        evaluation_id="c" * 64,
        source_uri="new-source",
        commit="d" * 40,
        training_manifest={"input_id": "trained", "validation_fold": 0},
        input_manifests={"ranking": "ranking-id", "items": "items-id"},
    )
    args = result["Steps"][0]["Arguments"]
    assert args["AlgorithmSpecification"]["TrainingImage"] == "proven-image"
    assert args["HyperParameters"]["sagemaker_program"] == "evaluate.py"
    assert "epochs" not in args["HyperParameters"]
    assert args["StoppingCondition"]["MaxRuntimeInSeconds"] == 7200
    assert "evaluations/" in args["CheckpointConfig"]["S3Uri"]
    assert "runs/folds/" in args["InputDataConfig"][-1]["DataSource"]["S3DataSource"]["S3Uri"]
    assert result["Metadata"]["ValidationFold"] == "0"
    assert "RetryPolicies" not in result["Steps"][0]
    assert (
        original["Steps"][0]["Arguments"]["HyperParameters"]["sagemaker_program"]
        == "sagemaker_entrypoint.py"
    )
