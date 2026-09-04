import pytest

from otto_recsys.evaluation.metrics import recall_at_k, weighted_recall_at_k


def test_recall_at_k_micro_definition() -> None:
    predictions = {1: [1, 2, 3], 2: [8, 9]}
    targets = {1: [2, 4], 2: [8]}
    assert recall_at_k(predictions, targets, k=20) == pytest.approx(2 / 3)


def test_recall_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        recall_at_k({}, {}, k=0)


def test_weighted_recall_matches_competition_weights() -> None:
    predictions = {
        "clicks": {1: [10]},
        "carts": {1: [11]},
        "orders": {1: [12]},
    }
    targets = {
        "clicks": {1: [10]},
        "carts": {1: [99]},
        "orders": {1: [12]},
    }

    score, detail = weighted_recall_at_k(predictions, targets)

    assert detail == {"clicks": 1.0, "carts": 0.0, "orders": 1.0}
    assert score == pytest.approx(0.7)
