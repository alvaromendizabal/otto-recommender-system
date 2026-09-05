from __future__ import annotations

import duckdb

from otto_recsys.ranking.hard_negatives import (
    create_hard_negative_training_rows,
    hard_negative_contract_rows,
)


def test_hard_negatives_exclude_all_future_positive_labels() -> None:
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
                (1, 'carts', 20),
                (1, 'carts', 30),
                (1, 'orders', 40)
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
                ('time', 'clicks', 1, 11, 0.9, 2),
                ('item2vec', 'clicks', 1, 12, 0.8, 1),
                ('type', 'carts', 1, 20, 1.0, 1),
                ('item2vec', 'carts', 1, 30, 0.95, 1),
                ('type', 'carts', 1, 31, 0.9, 2),
                ('item2vec', 'carts', 1, 32, 0.8, 2),
                ('buy', 'orders', 1, 40, 1.0, 1),
                ('buy', 'orders', 1, 41, 0.9, 2),
                ('item2vec', 'orders', 1, 42, 0.8, 1)
            """
        )

        create_hard_negative_training_rows(connection, hard_negatives=2)
        rows, sessions, empty_negatives, false_negative_rows = (
            hard_negative_contract_rows(connection)
        )
        assert rows == 4
        assert sessions == 1
        assert empty_negatives == 0
        assert false_negative_rows == 0

        cart = connection.execute(
            """
            SELECT hard_negative_aids
            FROM hard_negative_rows
            WHERE objective = 'carts' AND positive_aid = 20
            """
        ).fetchone()
        assert cart is not None
        assert 20 not in cart[0]
        assert 30 not in cart[0]
    finally:
        connection.close()
