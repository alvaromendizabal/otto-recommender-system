from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.runtime import Heartbeat

_MATRIX_NAMES = frozenset({"time", "type", "buy"})
_MEMORY_RE = re.compile(r"^[1-9][0-9]*(MB|GB)$")


@dataclass(frozen=True)
class CovisitManifest:
    name: str
    source_sha256: str
    top_k: int
    position_window: int
    rows: int
    output_sha256: str


def _literal(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _configure(
    connection: duckdb.DuckDBPyConnection,
    *,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")

    if not _MEMORY_RE.fullmatch(memory_limit):
        raise ValueError("invalid memory limit")

    temp_directory.mkdir(parents=True, exist_ok=True)

    connection.execute(f"SET threads = {threads}")
    connection.execute(
        f"SET memory_limit = '{memory_limit}'"
    )
    connection.execute(
        "SET preserve_insertion_order = false"
    )
    connection.execute(
        "SET temp_directory = "
        f"'{_literal(temp_directory)}'"
    )


def _matrix_query(
    name: str,
    source: Path,
    destination: Path,
    *,
    top_k: int,
    position_window: int,
) -> str:
    source_sql = _literal(source)
    destination_sql = _literal(destination)

    pair_cte = f"""
        WITH pairs AS (
            SELECT
                a.aid AS aid_a,
                b.aid AS aid_b,
                a.event_type AS type_a,
                b.event_type AS type_b,
                b.event_index - a.event_index AS position_gap,
                b.ts - a.ts AS time_gap
            FROM read_parquet('{source_sql}') AS a
            INNER JOIN read_parquet('{source_sql}') AS b
                ON a.session = b.session
               AND b.event_index > a.event_index
            WHERE
                a.aid <> b.aid
                AND b.event_index - a.event_index
                    <= {position_window}
                AND b.ts >= a.ts
                AND b.ts - a.ts <= 86400000
        ),
        directed AS (
            SELECT
                aid_a AS source_aid,
                aid_b AS target_aid,
                type_a AS source_type,
                type_b AS target_type,
                position_gap,
                time_gap
            FROM pairs

            UNION ALL

            SELECT
                aid_b AS source_aid,
                aid_a AS target_aid,
                type_b AS source_type,
                type_a AS target_type,
                position_gap,
                time_gap
            FROM pairs
        )
    """

    if name == "time":
        body = f"""
        {pair_cte},
        aggregated AS (
            SELECT
                source_aid,
                target_aid,
                sum(
                    exp(
                        -cast(time_gap AS DOUBLE)
                        / 21600000.0
                    )
                    /
                    sqrt(
                        cast(position_gap AS DOUBLE)
                    )
                ) AS score
            FROM directed
            GROUP BY source_aid, target_aid
        ),
        ranked AS (
            SELECT
                'all' AS objective,
                source_aid,
                target_aid,
                score,
                row_number() OVER (
                    PARTITION BY source_aid
                    ORDER BY score DESC, target_aid
                ) AS rank
            FROM aggregated
        )
        SELECT *
        FROM ranked
        WHERE rank <= {top_k}
        """

    elif name == "type":
        body = f"""
        {pair_cte},
        aggregated AS (
            SELECT
                source_aid,
                target_aid,

                sum(
                    (
                        CASE source_type
                            WHEN 0 THEN 1.0
                            WHEN 1 THEN 1.5
                            ELSE 2.0
                        END
                    )
                    *
                    (
                        CASE target_type
                            WHEN 0 THEN 3.0
                            WHEN 1 THEN 2.0
                            ELSE 2.5
                        END
                    )
                    /
                    sqrt(cast(position_gap AS DOUBLE))
                ) AS clicks_score,

                sum(
                    (
                        CASE source_type
                            WHEN 0 THEN 1.0
                            WHEN 1 THEN 2.0
                            ELSE 2.5
                        END
                    )
                    *
                    (
                        CASE target_type
                            WHEN 0 THEN 1.0
                            WHEN 1 THEN 4.0
                            ELSE 3.0
                        END
                    )
                    /
                    sqrt(cast(position_gap AS DOUBLE))
                ) AS carts_score,

                sum(
                    (
                        CASE source_type
                            WHEN 0 THEN 1.0
                            WHEN 1 THEN 2.5
                            ELSE 3.0
                        END
                    )
                    *
                    (
                        CASE target_type
                            WHEN 0 THEN 1.0
                            WHEN 1 THEN 3.0
                            ELSE 5.0
                        END
                    )
                    /
                    sqrt(cast(position_gap AS DOUBLE))
                ) AS orders_score

            FROM directed
            GROUP BY source_aid, target_aid
        ),
        long_form AS (
            SELECT
                'clicks' AS objective,
                source_aid,
                target_aid,
                clicks_score AS score
            FROM aggregated

            UNION ALL

            SELECT
                'carts',
                source_aid,
                target_aid,
                carts_score
            FROM aggregated

            UNION ALL

            SELECT
                'orders',
                source_aid,
                target_aid,
                orders_score
            FROM aggregated
        ),
        ranked AS (
            SELECT
                objective,
                source_aid,
                target_aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, source_aid
                    ORDER BY score DESC, target_aid
                ) AS rank
            FROM long_form
        )
        SELECT *
        FROM ranked
        WHERE rank <= {top_k}
        """

    elif name == "buy":
        body = f"""
        {pair_cte},
        buy_pairs AS (
            SELECT *
            FROM directed
            WHERE
                source_type IN (1, 2)
                AND target_type IN (1, 2)
        ),
        aggregated AS (
            SELECT
                source_aid,
                target_aid,

                sum(
                    exp(
                        -cast(time_gap AS DOUBLE)
                        / 43200000.0
                    )
                    *
                    CASE target_type
                        WHEN 1 THEN 4.0
                        ELSE 2.5
                    END
                ) AS carts_score,

                sum(
                    exp(
                        -cast(time_gap AS DOUBLE)
                        / 43200000.0
                    )
                    *
                    CASE target_type
                        WHEN 2 THEN 6.0
                        ELSE 3.0
                    END
                ) AS orders_score

            FROM buy_pairs
            GROUP BY source_aid, target_aid
        ),
        long_form AS (
            SELECT
                'carts' AS objective,
                source_aid,
                target_aid,
                carts_score AS score
            FROM aggregated

            UNION ALL

            SELECT
                'orders',
                source_aid,
                target_aid,
                orders_score
            FROM aggregated
        ),
        ranked AS (
            SELECT
                objective,
                source_aid,
                target_aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, source_aid
                    ORDER BY score DESC, target_aid
                ) AS rank
            FROM long_form
        )
        SELECT *
        FROM ranked
        WHERE rank <= {top_k}
        """

    else:
        raise ValueError(
            f"unsupported matrix {name!r}"
        )

    return f"""
    COPY (
        {body}
    )
    TO '{destination_sql}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """


def build_covisit_matrix(
    name: str,
    session_items_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    logger: logging.Logger,
    top_k: int,
    position_window: int = 10,
    threads: int = 2,
    memory_limit: str = "2GB",
    temp_directory: str | Path = "data/interim/duckdb",
    heartbeat_seconds: float = 30.0,
) -> CovisitManifest:
    """Build one external-memory co-visitation matrix."""
    if name not in _MATRIX_NAMES:
        raise ValueError(
            f"matrix must be one of {sorted(_MATRIX_NAMES)}"
        )

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if position_window <= 0:
        raise ValueError(
            "position_window must be positive"
        )

    source = Path(session_items_path).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    destination = Path(output_path).resolve()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_destination = Path(
        manifest_path
    ).resolve()
    manifest_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_sha256 = sha256_file(source)

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temporary.unlink(missing_ok=True)

    connection = duckdb.connect(
        database=":memory:"
    )

    _configure(
        connection,
        threads=threads,
        memory_limit=memory_limit,
        temp_directory=Path(
            temp_directory
        ).resolve(),
    )

    query = _matrix_query(
        name,
        source,
        temporary,
        top_k=top_k,
        position_window=position_window,
    )

    started = time.perf_counter()

    logger.info(
        "covisit_build_start",
        extra={
            "event": "covisit_build_start",
            "stage": f"covisit_{name}",
            "matrix": name,
            "top_k": top_k,
        },
    )

    try:
        with Heartbeat(
            logger,
            stage=f"covisit_{name}",
            interval_seconds=heartbeat_seconds,
        ):
            connection.execute(query)

        os.replace(
            temporary,
            destination,
        )

        row = connection.execute(
            "SELECT count(*) "
            f"FROM read_parquet('{_literal(destination)}')"
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "matrix row count was not returned"
            )

        rows = int(row[0])

    finally:
        connection.close()
        temporary.unlink(missing_ok=True)

    if rows <= 0:
        raise RuntimeError(
            f"{name} matrix contains no rows"
        )

    manifest = CovisitManifest(
        name=name,
        source_sha256=source_sha256,
        top_k=top_k,
        position_window=position_window,
        rows=rows,
        output_sha256=sha256_file(destination),
    )

    manifest_temp = (
        manifest_destination.with_suffix(
            manifest_destination.suffix + ".tmp"
        )
    )

    manifest_temp.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        manifest_temp,
        manifest_destination,
    )

    logger.info(
        "covisit_build_complete",
        extra={
            "event": "covisit_build_complete",
            "stage": f"covisit_{name}",
            "status": "passed",
            "matrix": name,
            "rows": rows,
            "elapsed_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        },
    )

    return manifest
