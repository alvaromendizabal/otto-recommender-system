from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def toolkit_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[dict[str, str]], list[str]]:
    # Import lazily: dependency-free launch tests also load this conftest.
    from sagemaker_training import environment, mapping, params

    path = tmp_path / "hyperparameters.json"
    monkeypatch.setattr(environment, "hyperparameters_file_dir", str(path))

    def load(parameters: dict[str, str]) -> list[str]:
        path.write_text(json.dumps(parameters))
        decoded = environment.read_hyperparameters()
        user = mapping.split_by_criteria(
            decoded, keys=params.SAGEMAKER_HYPERPARAMETERS, prefix=params.SAGEMAKER_PREFIX
        ).excluded
        return list(mapping.to_cmd_args(user))

    return load
