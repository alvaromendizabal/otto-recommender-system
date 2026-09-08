"""Reject silent changes to the fixed, fingerprinted feature-study schema."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from otto_recsys.research.features import LENGTHS, WEIGHTS
from otto_recsys.research.retrievers import HISTORY_HOURS


def read_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text())
    fixed = {
        "graph_prefix_lengths": list(LENGTHS),
        "graph_weights": list(WEIGHTS),
        "history_hours": list(HISTORY_HOURS),
    }
    for name, expected in fixed.items():
        if config["features"].get(name) != expected:
            raise ValueError(
                f"{name} differs from the fixed feature schema; change the "
                "implementation and create fresh artifacts for a new schema"
            )
    if config["retrieval"].get("candidate_budgets") != [100, 200, 400]:
        raise ValueError("this study requires the predeclared 100/200/400 candidate budgets")
    if config["resources"]["memory_gib"] < 1 or config["training"]["threads"] < 1:
        raise ValueError("research memory and thread limits must be positive")
    return config
