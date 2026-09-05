from __future__ import annotations

import argparse

import pytest
from sagemaker_entrypoint import _parse_bool


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", True])
def test_parse_bool_true(value: str | bool) -> None:
    assert _parse_bool(value) is True


@pytest.mark.parametrize("value", ["0", "false", "NO", "off", False])
def test_parse_bool_false(value: str | bool) -> None:
    assert _parse_bool(value) is False


def test_parse_bool_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="invalid boolean value"):
        _parse_bool("maybe")
