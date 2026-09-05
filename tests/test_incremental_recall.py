from __future__ import annotations

import duckdb

from otto_recsys.retrieval.incremental_recall import label_hit_rows


def test_label_hit_rows_tracks_shared_and_exclusive_recovery() -> None:
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
                (2, 'clicks', 20),
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
                ('item2vec', 'clicks', 1, 10, 0.9, 1),
                ('item2vec', 'clicks', 2, 20, 0.8, 1),
                ('type', 'carts', 1, 30, 1.0, 1)
            """
        )

        rows = {row[0]: row[1:] for row in label_hit_rows(connection)}

        clicks = rows["clicks"]
        assert clicks[0] == 2  # labels
        assert clicks[2] == 1  # time hits
        assert clicks[5] == 2  # item2vec hits
        assert clicks[6] == 1  # covisit hits
        assert clicks[7] == 2  # union hits
        assert clicks[8] == 1  # shared hits
        assert clicks[9] == 0  # covisit-only hits
        assert clicks[10] == 1  # item2vec-only hits
        assert clicks[11] == 0  # missed
        assert clicks[16] == 1  # item2vec globally unique

        carts = rows["carts"]
        assert carts[0] == 1
        assert carts[3] == 1  # type hits
        assert carts[6] == 1  # covisit hits
        assert carts[7] == 1  # union hits
        assert carts[9] == 1  # covisit-only hits
        assert carts[14] == 1  # type globally unique

        orders = rows["orders"]
        assert orders[0] == 1
        assert orders[7] == 0  # union hits
        assert orders[11] == 1  # missed
    finally:
        connection.close()
