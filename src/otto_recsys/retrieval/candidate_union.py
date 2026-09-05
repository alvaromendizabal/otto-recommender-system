from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models import KeyedVectors  # type: ignore[import-untyped]

_OBJECTIVES = ("clicks", "carts", "orders")
_MEMORY_RE = re.compile(r"^[1-9][0-9]*(MB|GB)$")
_EMBEDDING_WEIGHTS = {
    "clicks": np.asarray([1.0, 2.0, 2.5], dtype=np.float32),
    "carts": np.asarray([1.0, 4.0, 3.0], dtype=np.float32),
    "orders": np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
}


def sql_literal(value: str | Path) -> str:
    """Escape a filesystem path for a DuckDB SQL string literal."""
    return str(value).replace("'", "''")


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> None:
    """Apply bounded-memory settings used by candidate-union workloads."""
    if threads <= 0:
        raise ValueError("threads must be positive")
    if not _MEMORY_RE.fullmatch(memory_limit):
        raise ValueError("invalid DuckDB memory limit")

    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads = {threads}")
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(
        "SET temp_directory = "
        f"'{sql_literal(temp_directory)}'"
    )


def create_validation_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    items_path: Path,
    labels_path: Path,
    bucket: int,
) -> int:
    """Materialize one validation bucket as bounded temporary tables."""
    if bucket < 0:
        raise ValueError("bucket must be non-negative")

    items_sql = sql_literal(items_path)
    labels_sql = sql_literal(labels_path)

    connection.execute(
        f"""
        CREATE TEMP TABLE vitems AS
        SELECT session, aid, ts, event_type, event_index, recency_rank
        FROM read_parquet('{items_sql}')
        WHERE bucket = {bucket}
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE vlabels AS
        SELECT session, objective, aid
        FROM read_parquet('{labels_sql}')
        WHERE bucket = {bucket}
        """
    )

    row = connection.execute(
        "SELECT count(DISTINCT session) FROM vitems"
    ).fetchone()
    if row is None:
        raise RuntimeError("DuckDB did not return validation-session count")
    return int(row[0])


def _create_revisit_candidates(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_k: int,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE revisit_candidates AS
        WITH scored AS (
            SELECT
                session,
                aid,
                CASE event_type
                    WHEN 0 THEN 1.0
                    WHEN 1 THEN 2.0
                    ELSE 2.5
                END / (1.0 + 0.10 * (recency_rank - 1)) AS clicks_score,
                CASE event_type
                    WHEN 0 THEN 1.0
                    WHEN 1 THEN 4.0
                    ELSE 3.0
                END / (1.0 + 0.10 * (recency_rank - 1)) AS carts_score,
                CASE event_type
                    WHEN 0 THEN 1.0
                    WHEN 1 THEN 3.0
                    ELSE 5.0
                END / (1.0 + 0.10 * (recency_rank - 1)) AS orders_score
            FROM vitems
        ),
        long_form AS (
            SELECT 'clicks' AS objective, session, aid, clicks_score AS score
            FROM scored
            UNION ALL
            SELECT 'carts', session, aid, carts_score
            FROM scored
            UNION ALL
            SELECT 'orders', session, aid, orders_score
            FROM scored
        ),
        ranked AS (
            SELECT
                objective,
                session,
                aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY score DESC, aid
                ) AS source_rank
            FROM long_form
        )
        SELECT
            'revisit' AS source,
            objective,
            session,
            aid,
            score,
            source_rank
        FROM ranked
        WHERE source_rank <= {source_k}
        """
    )


def _create_time_candidates(
    connection: duckdb.DuckDBPyConnection,
    matrix_path: Path,
    *,
    source_k: int,
) -> None:
    matrix_sql = sql_literal(matrix_path)

    connection.execute(
        f"""
        CREATE TEMP TABLE time_candidates AS
        WITH scored AS (
            SELECT
                v.session,
                m.target_aid AS aid,
                sum(
                    m.score
                    * CASE v.event_type
                        WHEN 0 THEN 1.0
                        WHEN 1 THEN 2.0
                        ELSE 2.5
                    END
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS clicks_score,
                sum(
                    m.score
                    * CASE v.event_type
                        WHEN 0 THEN 1.0
                        WHEN 1 THEN 4.0
                        ELSE 3.0
                    END
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS carts_score,
                sum(
                    m.score
                    * CASE v.event_type
                        WHEN 0 THEN 1.0
                        WHEN 1 THEN 3.0
                        ELSE 5.0
                    END
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS orders_score
            FROM vitems AS v
            INNER JOIN read_parquet('{matrix_sql}') AS m
                ON v.aid = m.source_aid
            WHERE m.objective = 'all'
            GROUP BY v.session, m.target_aid
        ),
        long_form AS (
            SELECT 'clicks' AS objective, session, aid, clicks_score AS score
            FROM scored
            UNION ALL
            SELECT 'carts', session, aid, carts_score
            FROM scored
            UNION ALL
            SELECT 'orders', session, aid, orders_score
            FROM scored
        ),
        ranked AS (
            SELECT
                objective,
                session,
                aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY score DESC, aid
                ) AS source_rank
            FROM long_form
        )
        SELECT
            'time' AS source,
            objective,
            session,
            aid,
            score,
            source_rank
        FROM ranked
        WHERE source_rank <= {source_k}
        """
    )


def _create_objective_matrix_candidates(
    connection: duckdb.DuckDBPyConnection,
    matrix_path: Path,
    *,
    source_name: str,
    objectives: Sequence[str],
    source_k: int,
) -> None:
    if source_name not in {"type", "buy"}:
        raise ValueError("unsupported source name")
    if not objectives or any(value not in _OBJECTIVES for value in objectives):
        raise ValueError("unsupported objectives")

    matrix_sql = sql_literal(matrix_path)
    objective_sql = ", ".join(f"'{value}'" for value in objectives)

    connection.execute(
        f"""
        CREATE TEMP TABLE {source_name}_candidates AS
        WITH scored AS (
            SELECT
                m.objective,
                v.session,
                m.target_aid AS aid,
                sum(
                    m.score
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS score
            FROM vitems AS v
            INNER JOIN read_parquet('{matrix_sql}') AS m
                ON v.aid = m.source_aid
            WHERE m.objective IN ({objective_sql})
            GROUP BY m.objective, v.session, m.target_aid
        ),
        ranked AS (
            SELECT
                objective,
                session,
                aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY score DESC, aid
                ) AS source_rank
            FROM scored
        )
        SELECT
            '{source_name}' AS source,
            objective,
            session,
            aid,
            score,
            source_rank
        FROM ranked
        WHERE source_rank <= {source_k}
        """
    )


def create_covisit_source_candidates(
    connection: duckdb.DuckDBPyConnection,
    *,
    covisit_dir: Path,
    source_k: int,
) -> None:
    """Build revisit/time/type/buy candidates for the current bucket."""
    if source_k <= 0:
        raise ValueError("source_k must be positive")

    _create_revisit_candidates(connection, source_k=source_k)
    _create_time_candidates(
        connection,
        covisit_dir / "time.parquet",
        source_k=source_k,
    )
    _create_objective_matrix_candidates(
        connection,
        covisit_dir / "type.parquet",
        source_name="type",
        objectives=_OBJECTIVES,
        source_k=source_k,
    )
    _create_objective_matrix_candidates(
        connection,
        covisit_dir / "buy.parquet",
        source_name="buy",
        objectives=("carts", "orders"),
        source_k=source_k,
    )

    connection.execute(
        """
        CREATE TEMP TABLE source_candidates AS
        SELECT * FROM revisit_candidates
        UNION ALL
        SELECT * FROM time_candidates
        UNION ALL
        SELECT * FROM type_candidates
        UNION ALL
        SELECT * FROM buy_candidates
        """
    )


def _queries_for_objective(
    *,
    items_path: Path,
    vectors: KeyedVectors,
    bucket: int,
    objective: str,
) -> tuple[np.ndarray, np.ndarray]:
    if objective not in _OBJECTIVES:
        raise ValueError("unsupported objective")

    table = pq.read_table(
        items_path,
        filters=[("bucket", "=", bucket)],
        columns=["session", "aid", "event_type", "recency_rank"],
    )

    sessions = table.column("session").to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)
    aids = table.column("aid").to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)
    event_types = table.column("event_type").to_numpy(
        zero_copy_only=False,
    ).astype(np.int64, copy=False)
    recency_rank = table.column("recency_rank").to_numpy(
        zero_copy_only=False,
    ).astype(np.float32, copy=False)

    if sessions.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(
            (0, vectors.vector_size),
            dtype=np.float32,
        )

    unique_sessions, inverse = np.unique(sessions, return_inverse=True)
    vector_rows: list[int] = []
    keep_mask = np.zeros(aids.shape[0], dtype=bool)

    for index, aid in enumerate(aids.tolist()):
        vector_index = vectors.key_to_index.get(int(aid))
        if vector_index is not None:
            vector_rows.append(int(vector_index))
            keep_mask[index] = True

    if not vector_rows:
        return np.empty(0, dtype=np.int64), np.empty(
            (0, vectors.vector_size),
            dtype=np.float32,
        )

    kept_inverse = inverse[keep_mask]
    kept_types = event_types[keep_mask]
    kept_recency = recency_rank[keep_mask]
    item_vectors = np.asarray(
        vectors.vectors[np.asarray(vector_rows, dtype=np.int64)],
        dtype=np.float32,
    )

    recency_weight = 1.0 / (1.0 + 0.10 * (kept_recency - 1.0))
    action_weight = _EMBEDDING_WEIGHTS[objective][kept_types]
    row_weight = recency_weight * action_weight

    queries = np.zeros(
        (unique_sessions.size, vectors.vector_size),
        dtype=np.float32,
    )
    weight_sum = np.zeros(unique_sessions.size, dtype=np.float32)

    np.add.at(queries, kept_inverse, item_vectors * row_weight[:, None])
    np.add.at(weight_sum, kept_inverse, row_weight)

    valid = weight_sum > 0
    queries[valid] /= weight_sum[valid, None]
    valid_queries = np.ascontiguousarray(queries[valid], dtype=np.float32)
    faiss.normalize_L2(valid_queries)
    return unique_sessions[valid], valid_queries


def embedding_candidate_table(
    *,
    sessions: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    objective: str,
) -> pa.Table:
    if neighbors.shape != distances.shape:
        raise ValueError("neighbors and distances must have matching shapes")
    if neighbors.ndim != 2:
        raise ValueError("neighbors must be a 2D array")
    if sessions.shape[0] != neighbors.shape[0]:
        raise ValueError("session count must match neighbor rows")

    rows, ann_k = neighbors.shape
    if rows == 0 or ann_k == 0:
        return pa.table(
            {
                "source": pa.array([], type=pa.string()),
                "objective": pa.array([], type=pa.string()),
                "session": pa.array([], type=pa.int64()),
                "aid": pa.array([], type=pa.int64()),
                "score": pa.array([], type=pa.float32()),
                "source_rank": pa.array([], type=pa.int32()),
            }
        )

    repeated_sessions = np.repeat(sessions.astype(np.int64, copy=False), ann_k)
    flattened_neighbors = neighbors.reshape(-1).astype(np.int64, copy=False)
    flattened_distances = distances.reshape(-1).astype(np.float32, copy=False)
    ranks = np.tile(
        np.arange(1, ann_k + 1, dtype=np.int32),
        rows,
    )
    keep = flattened_neighbors >= 0

    kept_count = int(np.count_nonzero(keep))
    return pa.table(
        {
            "source": pa.array(["item2vec"] * kept_count),
            "objective": pa.array([objective] * kept_count),
            "session": pa.array(repeated_sessions[keep]),
            "aid": pa.array(flattened_neighbors[keep]),
            "score": pa.array(flattened_distances[keep]),
            "source_rank": pa.array(ranks[keep]),
        }
    )


def append_item2vec_candidates(
    connection: duckdb.DuckDBPyConnection,
    *,
    items_path: Path,
    vectors: KeyedVectors,
    index: faiss.Index,
    bucket: int,
    ann_k: int,
    ef_search: int,
) -> int:
    """Append target-conditioned Item2Vec ANN candidates to source_candidates."""
    if ann_k <= 0:
        raise ValueError("ann_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    faiss.ParameterSpace().set_index_parameter(index, "efSearch", ef_search)
    appended = 0

    for objective in _OBJECTIVES:
        sessions, queries = _queries_for_objective(
            items_path=items_path,
            vectors=vectors,
            bucket=bucket,
            objective=objective,
        )
        if queries.shape[0] == 0:
            continue

        distances, neighbors = index.search(queries, ann_k)
        table = embedding_candidate_table(
            sessions=sessions,
            neighbors=neighbors,
            distances=distances,
            objective=objective,
        )
        if table.num_rows == 0:
            continue

        connection.register("embedding_batch", table)
        try:
            connection.execute(
                """
                INSERT INTO source_candidates
                SELECT
                    source,
                    objective,
                    session,
                    aid,
                    score,
                    source_rank
                FROM embedding_batch
                """
            )
        finally:
            connection.unregister("embedding_batch")
        appended += table.num_rows

    return appended


def candidate_count_rows(
    connection: duckdb.DuckDBPyConnection,
) -> list[tuple[str, int, int, int]]:
    """Return per-objective candidate counts for covisit, Item2Vec, and union."""
    rows = connection.execute(
        """
        WITH per_session AS (
            SELECT
                objective,
                session,
                count(DISTINCT aid) FILTER (
                    WHERE source IN ('revisit', 'time', 'type', 'buy')
                ) AS covisit_candidates,
                count(DISTINCT aid) FILTER (
                    WHERE source = 'item2vec'
                ) AS item2vec_candidates,
                count(DISTINCT aid) AS union_candidates
            FROM source_candidates
            GROUP BY objective, session
        )
        SELECT
            objective,
            sum(covisit_candidates),
            sum(item2vec_candidates),
            sum(union_candidates)
        FROM per_session
        GROUP BY objective
        ORDER BY objective
        """
    ).fetchall()

    return [
        (str(row[0]), int(row[1]), int(row[2]), int(row[3]))
        for row in rows
    ]
