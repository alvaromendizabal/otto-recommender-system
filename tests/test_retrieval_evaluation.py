from __future__ import annotations

import numpy as np
import pytest

from otto_recsys.evaluation.retrieval import (
    candidate_recall_ceiling_at_k,
    incremental_candidate_recall_at_k,
    paired_poisson_bootstrap_delta,
    weighted_objective_score,
)


def test_candidate_recall_ceiling_matches_top20_contract() -> None:
    labels = {1: [10, 20], 2: [30]}
    candidates = {1: [99, 10, 20], 2: [30, 40]}
    result = candidate_recall_ceiling_at_k(
        objective="orders", labels=labels, candidates=candidates, k=20
    )
    assert result.hits == 3
    assert result.denominator == 3
    assert result.recall == 1.0


def test_candidate_recall_ceiling_deduplicates_predictions() -> None:
    result = candidate_recall_ceiling_at_k(
        objective="clicks",
        labels={1: [10]},
        candidates={1: [10, 10, 10]},
        k=20,
    )
    assert result.hits == 1
    assert result.denominator == 1


def test_candidate_recall_ceiling_caps_hits_at_twenty() -> None:
    labels = {1: list(range(30))}
    candidates = {1: list(range(30))}
    result = candidate_recall_ceiling_at_k(
        objective="orders", labels=labels, candidates=candidates, k=30
    )
    assert result.hits == 20
    assert result.denominator == 20
    assert result.recall == 1.0


def test_incremental_recall_tracks_neural_only_positive_hits() -> None:
    result = incremental_candidate_recall_at_k(
        objective="orders",
        labels={1: [10, 20], 2: [30]},
        base_candidates={1: [10], 2: [99]},
        neural_candidates={1: [20], 2: [30]},
        k=20,
    )
    assert result.base_hits == 1
    assert result.neural_hits == 2
    assert result.union_hits == 3
    assert result.neural_unique_hits == 2
    assert result.incremental_recall == pytest.approx(2 / 3)


def test_weighted_objective_score_uses_official_weights() -> None:
    score = weighted_objective_score(
        {"clicks": 0.5, "carts": 0.6, "orders": 0.7}
    )
    assert score == pytest.approx(0.65)


def test_weighted_objective_score_requires_all_objectives() -> None:
    with pytest.raises(ValueError, match="missing objective recalls"):
        weighted_objective_score({"clicks": 0.5})


def test_paired_poisson_bootstrap_is_deterministic_and_contains_point() -> None:
    base = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
    union = np.array([1, 1, 0, 1, 1, 1], dtype=np.float64)
    denom = np.ones(6, dtype=np.float64)
    first = paired_poisson_bootstrap_delta(
        base_numerators=base,
        union_numerators=union,
        denominators=denom,
        iterations=200,
        seed=7,
        chunk_size=8,
    )
    second = paired_poisson_bootstrap_delta(
        base_numerators=base,
        union_numerators=union,
        denominators=denom,
        iterations=200,
        seed=7,
        chunk_size=8,
    )
    assert first == second
    point, lower, upper = first
    assert lower <= point <= upper
    assert point == pytest.approx(2 / 6)
