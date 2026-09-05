from __future__ import annotations

import duckdb
import pytest

from otto_recsys.retrieval.candidate_budget import (
    coverage_histogram_rows,
    normalize_quotas,
    quota_summary,
    recommend_quotas,
)


def test_normalize_quotas_sorts_deduplicates_and_adds_zero() -> None:
    assert normalize_quotas([50, 10, 50, 200]) == (0, 10, 50, 200)
    with pytest.raises(ValueError):
        normalize_quotas([])
    with pytest.raises(ValueError):
        normalize_quotas([-1, 20])
    with pytest.raises(ValueError):
        normalize_quotas([0])


def test_quota_summary_tracks_exact_union_recall_and_candidate_cost() -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            """
            CREATE TEMP TABLE vlabels (
                session BIGINT,
                objective VARCHAR,
                aid BIGINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO vlabels VALUES
                (1, 'clicks', 10),
                (1, 'clicks', 20),
                (1, 'clicks', 30),
                (1, 'clicks', 40),
                (1, 'carts', 50),
                (1, 'orders', 60)
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE source_candidates (
                source VARCHAR,
                objective VARCHAR,
                session BIGINT,
                aid BIGINT,
                score DOUBLE,
                source_rank BIGINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO source_candidates VALUES
                ('time', 'clicks', 1, 10, 1.0, 1),
                ('item2vec', 'clicks', 1, 10, 0.9, 1),
                ('item2vec', 'clicks', 1, 20, 0.8, 2),
                ('type', 'clicks', 1, 30, 0.7, 1),
                ('item2vec', 'clicks', 1, 70, 0.6, 3),
                ('type', 'carts', 1, 50, 1.0, 1),
                ('item2vec', 'carts', 1, 80, 0.5, 1),
                ('revisit', 'orders', 1, 60, 1.0, 1)
            """
        )

        label_histogram, candidate_histogram = coverage_histogram_rows(connection)
        metrics, candidates = quota_summary(
            label_histogram,
            candidate_histogram,
            sessions=1,
            quotas=(0, 1, 2, 3),
        )

        assert metrics["clicks.covisit_recall_ceiling"] == pytest.approx(0.5)
        assert metrics["clicks.item2vec_marginal_recall_1"] == pytest.approx(0.0)
        assert metrics["clicks.item2vec_marginal_recall_2"] == pytest.approx(0.25)
        assert metrics["clicks.union_recall_2"] == pytest.approx(0.75)
        assert candidates["clicks.covisit_average_candidates"] == pytest.approx(2.0)
        assert candidates["clicks.item2vec_average_candidates_2"] == pytest.approx(2.0)
        assert candidates["clicks.added_average_candidates_2"] == pytest.approx(1.0)
        assert candidates["clicks.union_average_candidates_2"] == pytest.approx(3.0)
    finally:
        connection.close()


def test_recommend_quotas_uses_smallest_quota_meeting_capture_target() -> None:
    metrics = {
        "clicks.item2vec_marginal_recall_0": 0.0,
        "clicks.item2vec_marginal_recall_50": 0.018,
        "clicks.item2vec_marginal_recall_100": 0.021,
        "clicks.item2vec_marginal_recall_200": 0.022,
        "carts.item2vec_marginal_recall_0": 0.0,
        "carts.item2vec_marginal_recall_50": 0.0145,
        "carts.item2vec_marginal_recall_100": 0.0150,
        "carts.item2vec_marginal_recall_200": 0.0155,
        "orders.item2vec_marginal_recall_0": 0.0,
        "orders.item2vec_marginal_recall_50": 0.0065,
        "orders.item2vec_marginal_recall_100": 0.0067,
        "orders.item2vec_marginal_recall_200": 0.0068,
    }
    recommended = recommend_quotas(
        metrics,
        quotas=(0, 50, 100, 200),
        capture_fraction=0.95,
    )
    assert recommended == {
        "clicks": 100,
        "carts": 100,
        "orders": 50,
    }
