"""Configuration values must not be silently ignored by a fixed feature schema."""

from pathlib import Path

import pytest

from otto_recsys.research.configuration import read_config


@pytest.mark.parametrize(
    "old,new",
    [
        ("graph_prefix_lengths = [1, 3, 5, 10, 20, 50]", "graph_prefix_lengths = [1, 10]"),
        ("candidate_budgets = [100, 200, 400]", "candidate_budgets = [100, 200]"),
        ("memory_gib = 12", "memory_gib = 0"),
    ],
)
def test_unsupported_schema_and_resource_settings_fail_early(tmp_path, old, new):
    original = Path("configs/research.toml").read_text()
    assert old in original
    path = tmp_path / "research.toml"
    path.write_text(original.replace(old, new))
    with pytest.raises(ValueError):
        read_config(path)
    assert read_config(Path("configs/research.toml"))["features"]["max_retained"] == 128
