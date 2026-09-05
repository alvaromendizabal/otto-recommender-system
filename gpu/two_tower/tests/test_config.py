from __future__ import annotations

import pytest

from otto_two_tower.config import DataConfig, ModelConfig, TrainConfig


def test_configs_validate() -> None:
    DataConfig().validate()
    ModelConfig().validate()
    TrainConfig().validate()


def test_invalid_validation_fold_rejected() -> None:
    with pytest.raises(ValueError):
        DataConfig(validation_fold=5, folds=5).validate()
