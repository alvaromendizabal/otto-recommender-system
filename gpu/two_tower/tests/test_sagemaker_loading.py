"""Compare preflight against the pinned AWS loader, framework split and serializer."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from otto_two_tower.ann_cli import hyperparameters_to_argv, parse_args


def test_captured_managed_launch_matches_real_toolkit_loading(
    toolkit_argv: Callable[[dict[str, str]], list[str]],
) -> None:
    observed = json.loads((Path(__file__).parent / "fixtures/ann_launch.json").read_text())
    actual = toolkit_argv(observed["hyperparameters"])
    assert actual == observed["observed_argv"]
    assert hyperparameters_to_argv(observed["hyperparameters"]) == actual
    assert parse_args(actual).export_fold_predictions == "true"


@pytest.mark.parametrize("export_predictions", ["true", "false"])
def test_production_ann_definition_survives_toolkit_loading(
    monkeypatch: pytest.MonkeyPatch,
    toolkit_argv: Callable[[dict[str, str]], list[str]],
    export_predictions: str,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repository / "src"))
    from otto_recsys.cloud.two_tower_ann import ann_definition, load_ann_parameters
    from otto_recsys.cloud.two_tower_fold import FoldTrainingConfig, build_fold_pipeline_definition

    training = build_fold_pipeline_definition(
        role_arn="role",
        image_uri="image",
        source_uri="source",
        commit="previous",
        run_id="training",
        config=FoldTrainingConfig(bucket="bucket"),
    )
    parameters = load_ann_parameters(repository / "configs/two_tower_ann.toml")
    parameters["export-fold-predictions"] = export_predictions
    definition = ann_definition(
        training_definition=training,
        bucket="bucket",
        training_run_id="trained",
        run_id="run",
        reference_run_id="export",
        reference_input_id="reference",
        reference_manifest_sha256="hash",
        source_uri="source",
        commit="commit",
        training_manifest={"input_id": "trained", "validation_fold": 0},
        input_manifests={"ranking": "ranking", "items": "items"},
        parameters=parameters,
    )
    hyperparameters = definition["Steps"][0]["Arguments"]["HyperParameters"]
    actual = toolkit_argv(hyperparameters)
    assert actual[actual.index("--export-fold-predictions") + 1] == export_predictions.title()
    assert hyperparameters_to_argv(hyperparameters) == actual
    assert parse_args(actual).export_fold_predictions == export_predictions
