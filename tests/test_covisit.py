from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.retrieval.covisit import (
    build_covisit_matrix,
)


def make_session_items(path: Path) -> None:
    table = pa.table(
        {
            "session": pa.array(
                [1, 1, 1, 2, 2],
                type=pa.int32(),
            ),
            "aid": pa.array(
                [10, 20, 30, 10, 20],
                type=pa.int32(),
            ),
            "ts": pa.array(
                [1000, 2000, 3000, 4000, 5000],
                type=pa.int64(),
            ),
            "event_type": pa.array(
                [0, 1, 2, 1, 2],
                type=pa.int8(),
            ),
            "event_index": pa.array(
                [0, 1, 2, 0, 1],
                type=pa.uint16(),
            ),
            "recency_rank": pa.array(
                [3, 2, 1, 2, 1],
                type=pa.uint16(),
            ),
        }
    )

    pq.write_table(
        table,
        path,
        compression="zstd",
    )


def test_time_covisit_builds_nonempty_matrix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session_items.parquet"
    output = tmp_path / "time.parquet"
    manifest_path = tmp_path / "time.json"

    make_session_items(source)

    manifest = build_covisit_matrix(
        "time",
        source,
        output,
        manifest_path,
        logger=logging.getLogger("test"),
        top_k=10,
        position_window=10,
        threads=1,
        memory_limit="512MB",
        temp_directory=tmp_path / "duckdb",
        heartbeat_seconds=10.0,
    )

    assert output.is_file()
    assert manifest.rows > 0

    result = pq.read_table(output)

    assert result.num_rows == manifest.rows
    assert {
        "objective",
        "source_aid",
        "target_aid",
        "score",
        "rank",
    }.issubset(result.column_names)


def test_buy_covisit_uses_purchase_events(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session_items.parquet"
    output = tmp_path / "buy.parquet"
    manifest_path = tmp_path / "buy.json"

    make_session_items(source)

    manifest = build_covisit_matrix(
        "buy",
        source,
        output,
        manifest_path,
        logger=logging.getLogger("test"),
        top_k=10,
        position_window=10,
        threads=1,
        memory_limit="512MB",
        temp_directory=tmp_path / "duckdb",
        heartbeat_seconds=10.0,
    )

    assert manifest.rows > 0
