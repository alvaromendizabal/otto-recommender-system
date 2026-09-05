import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.data.schema import EVENT_SCHEMA
from otto_recsys.retrieval.session_items import (
    build_session_item_cache,
)


def test_session_item_cache_deduplicates_and_caps(
    tmp_path: Path,
) -> None:
    table = pa.Table.from_pydict(
        {
            "session": [1, 1, 1, 2, 2],
            "aid": [10, 11, 10, 20, 20],
            "ts": [100, 200, 300, 100, 200],
            "event_type": [0, 1, 0, 0, 1],
            "event_index": [0, 1, 2, 0, 1],
        },
        schema=EVENT_SCHEMA,
    )

    source = tmp_path / "part-000000.parquet"
    pq.write_table(table, source)

    validation_manifest = tmp_path / "validation.json"
    validation_manifest.write_text(
        json.dumps({"split_ts": 1000}),
        encoding="utf-8",
    )

    output = tmp_path / "session_items.parquet"
    output_manifest = tmp_path / "manifest.json"

    manifest = build_session_item_cache(
        source,
        validation_manifest,
        output,
        output_manifest,
        logger=logging.getLogger("test"),
        max_items_per_session=20,
        threads=1,
        memory_limit="512MB",
        temp_directory=tmp_path / "duckdb",
        heartbeat_seconds=10.0,
    )

    result = pq.read_table(output)

    assert manifest.rows == 2
    assert result.num_rows == 2
    assert set(result.column("aid").to_pylist()) == {10, 11}
