import logging
from pathlib import Path

import orjson

from otto_recsys.data.manifest import build_manifest, write_manifest
from otto_recsys.validation.kaggle_split import (
    DAY_MILLIS,
    _future_labels,
    build_validation,
)


def test_future_labels_use_next_click_and_future_sets() -> None:
    future = [
        {"aid": 10, "ts": 1, "type": "carts"},
        {"aid": 11, "ts": 2, "type": "clicks"},
        {"aid": 10, "ts": 3, "type": "orders"},
        {"aid": 12, "ts": 4, "type": "clicks"},
        {"aid": 13, "ts": 5, "type": "carts"},
    ]

    assert _future_labels(future) == {
        "clicks": 11,
        "carts": [10, 13],
        "orders": [10],
    }


def test_validation_is_deterministic_and_filters_unknown_items(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    day = DAY_MILLIS

    source_sessions = [
        {
            "session": 1,
            "events": [
                {"aid": 10, "ts": day, "type": "clicks"},
                {"aid": 11, "ts": day + 100, "type": "carts"},
            ],
        },
        {
            "session": 2,
            "events": [
                {"aid": 10, "ts": 2 * day + 100, "type": "clicks"},
                {"aid": 11, "ts": 2 * day + 200, "type": "carts"},
                {"aid": 999, "ts": 2 * day + 300, "type": "orders"},
            ],
        },
    ]

    with source.open("wb") as handle:
        for session_record in source_sessions:
            handle.write(orjson.dumps(session_record) + b"\n")

    raw_manifest = build_manifest(
        [source],
        source="unit-test",
        logger=logging.getLogger("test"),
    )

    raw_manifest_path = tmp_path / "raw-manifest.json"
    write_manifest(raw_manifest, raw_manifest_path)

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = build_validation(
        source,
        raw_manifest_path,
        first_output,
        max_ts=4 * day,
        days=2,
        seed=42,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )

    second = build_validation(
        source,
        raw_manifest_path,
        second_output,
        max_ts=4 * day,
        days=2,
        seed=42,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )

    assert first.manifest_id == second.manifest_id
    assert first.train_sessions == 1
    assert first.test_sessions == 1

    assert (
        first_output / "train_sessions.jsonl"
    ).read_bytes() == (
        second_output / "train_sessions.jsonl"
    ).read_bytes()

    assert (
        first_output / "test_sessions.jsonl"
    ).read_bytes() == (
        second_output / "test_sessions.jsonl"
    ).read_bytes()

    assert b"999" not in (
        first_output / "test_sessions.jsonl"
    ).read_bytes()
