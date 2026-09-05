from __future__ import annotations

import json
import logging
from pathlib import Path

import orjson
import pyarrow.parquet as pq

from otto_recsys.retrieval.validation_cache import (
    build_retrieval_validation_cache,
)


def test_validation_cache_is_deduplicated_and_aligned(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()

    (validation / "manifest.json").write_text(
        json.dumps({"manifest_id": "abc123"}),
        encoding="utf-8",
    )

    session_record = {
        "session": 5,
        "events": [
            {"aid": 10, "ts": 100, "type": "clicks"},
            {"aid": 11, "ts": 200, "type": "carts"},
            {"aid": 10, "ts": 300, "type": "orders"},
        ],
    }
    label_record = {
        "session": 5,
        "labels": {
            "clicks": 20,
            "carts": [21, 22],
            "orders": [23],
        },
    }

    (validation / "test_sessions.jsonl").write_bytes(
        orjson.dumps(session_record) + b"\n"
    )
    (validation / "test_labels.jsonl").write_bytes(
        orjson.dumps(label_record) + b"\n"
    )

    output = tmp_path / "cache"
    manifest = build_retrieval_validation_cache(
        validation,
        output,
        logger=logging.getLogger("test"),
        buckets=8,
        flush_sessions=1,
        heartbeat_seconds=10.0,
    )

    items = pq.read_table(output / "items.parquet")
    labels = pq.read_table(output / "labels.parquet")

    assert manifest.sessions == 1
    assert manifest.item_rows == 2
    assert manifest.label_rows == 4
    assert set(items.column("aid").to_pylist()) == {10, 11}

    item_by_aid = {
        aid: (
            ts,
            event_type,
            event_index,
            recency_rank,
        )
        for aid, ts, event_type, event_index, recency_rank in zip(
            items.column("aid").to_pylist(),
            items.column("ts").to_pylist(),
            items.column("event_type").to_pylist(),
            items.column("event_index").to_pylist(),
            items.column("recency_rank").to_pylist(),
            strict=True,
        )
    }

    assert item_by_aid[10] == (300, 2, 2, 1)
    assert item_by_aid[11] == (200, 1, 1, 2)
    assert set(labels.column("objective").to_pylist()) == {
        "clicks",
        "carts",
        "orders",
    }
