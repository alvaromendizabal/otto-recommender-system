from __future__ import annotations

import duckdb
import numpy as np

from otto_recsys.retrieval.candidate_union import (
    candidate_count_rows,
    embedding_candidate_table,
)


def test_embedding_candidate_table_filters_missing_neighbors() -> None:
    sessions = np.asarray([10, 20], dtype=np.int64)
    neighbors = np.asarray(
        [
            [101, 102, -1],
            [201, 202, 203],
        ],
        dtype=np.int64,
    )
    distances = np.asarray(
        [
            [0.9, 0.8, -1.0],
            [0.7, 0.6, 0.5],
        ],
        dtype=np.float32,
    )

    table = embedding_candidate_table(
        sessions=sessions,
        neighbors=neighbors,
        distances=distances,
        objective="clicks",
    )

    assert table.num_rows == 5
    assert table.column("source").to_pylist() == ["item2vec"] * 5
    assert table.column("objective").to_pylist() == ["clicks"] * 5
    assert table.column("session").to_pylist() == [10, 10, 20, 20, 20]
    assert table.column("aid").to_pylist() == [101, 102, 201, 202, 203]
    assert table.column("source_rank").to_pylist() == [1, 2, 1, 2, 3]


def test_embedding_candidate_table_rejects_shape_mismatch() -> None:
    sessions = np.asarray([10], dtype=np.int64)
    neighbors = np.asarray([[101, 102]], dtype=np.int64)
    distances = np.asarray([[0.9]], dtype=np.float32)

    try:
        embedding_candidate_table(
            sessions=sessions,
            neighbors=neighbors,
            distances=distances,
            objective="clicks",
        )
    except ValueError as exc:
        assert "matching shapes" in str(exc)
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def test_candidate_count_rows_deduplicates_across_sources() -> None:
    connection = duckdb.connect(database=":memory:")
    try:
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
                ('type', 'clicks', 1, 10, 0.9, 1),
                ('time', 'clicks', 1, 11, 0.8, 2),
                ('item2vec', 'clicks', 1, 10, 0.7, 1),
                ('item2vec', 'clicks', 1, 12, 0.6, 2)
            """
        )

        rows = candidate_count_rows(connection)
        assert rows == [("clicks", 2, 2, 3)]
    finally:
        connection.close()
