"""Whole-query sampling and quality screening contracts."""

from __future__ import annotations

import numpy as np
import pytest

from otto_recsys.research.dataset import sampled_rows, session_hash
from otto_recsys.research.screening import quality_screen


def test_fit_sampling_keeps_every_positive_and_is_deterministic():
    targets = np.zeros((100, 3), dtype=np.int8)
    targets[99, 2] = 1
    targets[70, 1] = 1
    targets[2, 0] = 1
    items = np.arange(100)
    chosen = sampled_rows(targets, items, 42, 10, 123)
    assert {2, 70, 99}.issubset(chosen)
    assert chosen.size == 13 and len(set(chosen)) == 13
    assert np.array_equal(chosen, sampled_rows(targets, items, 42, 10, 123))
    folds = session_hash(np.array([42, 42]), 123) % 3
    assert folds[0] == folds[1]


def test_quality_screen_identifies_constant_duplicate_and_target_equivalent_features():
    rng = np.random.default_rng(8)
    signal = rng.normal(size=100)
    truth = rng.integers(0, 2, size=(100, 3))
    x = np.column_stack([np.ones(100), signal, signal, truth[:, 0]])
    names = ("source_time_score", "source_cart_score", "source_order_score", "source_revisit_score")
    records = quality_screen(x, truth, names)
    assert [r["status"] for r in records] == [
        "constant",
        "eligible",
        "duplicate",
        "target_equivalent",
    ]
    assert records[2]["duplicate_of"] == names[1]
    with pytest.raises(ValueError, match="catalog"):
        quality_screen(x, truth, ("aid", *names[1:]))
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        quality_screen(x, truth, names)
