from __future__ import annotations

import numpy as np
import pytest

from otto_two_tower.ranking_metrics import paired_recall_interval, ranking_counts, summarize_ranking


def test_official_micro_recall_differs_from_mean_session_recall() -> None:
    a = ranking_counts({1}, [1])
    b = ranking_counts(set(range(2, 32)), [2, 3])
    counts = np.array([[a, a, a], [b, b, b]])
    result = summarize_ranking(counts)
    assert result["weighted_recall_at_20"] == pytest.approx(3 / 21)
    assert result["objectives"]["clicks"]["mrr_at_20"] == 1
    assert result["objectives"]["clicks"]["precision_at_20"] == pytest.approx(3 / 40)
    assert result["weighted_recall_at_20"] != pytest.approx((1 + 2 / 20) / 2)


def test_rank_position_diagnostics_and_unknown_positives() -> None:
    values = ranking_counts({999, 4}, [2, 4, 3])
    assert values[0] == 2
    assert values[1] == 1
    assert values[3] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert values[4] == 0.5
    assert ranking_counts(set(), [1, 2]).sum() == 0
    with pytest.raises(ValueError):
        ranking_counts({1}, [1, 1])


def test_identical_paired_predictions_have_zero_interval() -> None:
    counts = np.array([[ranking_counts({1}, [1])] * 3] * 10)
    result = paired_recall_interval(counts, counts, iterations=20)
    assert result["weighted_recall_at_20_delta_ci95"] == [0, 0]
    changed = counts.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="denominators"):
        paired_recall_interval(counts, changed)
