from __future__ import annotations

import json
import logging
from pathlib import Path

import orjson
import pyarrow.parquet as pq
import pytest

from otto_recsys.ranking.training_cache import (
    build_ranking_training_cache,
    fold_for_session,
)


def _write_validation_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"manifest_id": "a" * 64}),
        encoding="utf-8",
    )
    sessions = [
        {
            "session": 10,
            "events": [
                {"aid": 1, "ts": 100, "type": "clicks"},
                {"aid": 2, "ts": 200, "type": "carts"},
                {"aid": 1, "ts": 300, "type": "clicks"},
            ],
        },
        {
            "session": 11,
            "events": [
                {"aid": 3, "ts": 110, "type": "clicks"},
                {"aid": 4, "ts": 210, "type": "orders"},
            ],
        },
    ]
    labels = [
        {
            "session": 10,
            "labels": {"clicks": 99, "carts": [98, 98], "orders": [97]},
        },
        {"session": 11, "labels": {"carts": [96]}},
    ]
    with (root / "test_sessions.jsonl").open("wb") as handle:
        for record in sessions:
            handle.write(orjson.dumps(record) + b"\n")
    with (root / "test_labels.jsonl").open("wb") as handle:
        for record in labels:
            handle.write(orjson.dumps(record) + b"\n")


def test_fold_assignment_is_deterministic() -> None:
    values = [fold_for_session(session, seed=7, folds=5) for session in range(100)]
    assert values == [
        fold_for_session(session, seed=7, folds=5)
        for session in range(100)
    ]
    assert all(0 <= value < 5 for value in values)
    with pytest.raises(ValueError):
        fold_for_session(1, seed=7, folds=1)


def test_training_cache_uses_only_frozen_observed_prefixes(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    output = tmp_path / "cache"
    _write_validation_fixture(validation)
    manifest = build_ranking_training_cache(
        validation,
        output,
        logger=logging.getLogger("test"),
        buckets=4,
        folds=2,
        fold_seed=17,
        flush_sessions=1,
        heartbeat_seconds=3600,
    )

    assert manifest.validation_manifest_id == "a" * 64
    assert manifest.sessions == 2
    assert manifest.event_rows == 5
    assert manifest.item_rows == 4
    assert manifest.label_rows == 4
    assert manifest.click_labels == 1
    assert manifest.cart_labels == 2
    assert manifest.order_labels == 1
    assert sum(manifest.fold_session_counts) == 2

    events = pq.read_table(output / "events.parquet").to_pydict()
    labels = pq.read_table(output / "labels.parquet").to_pydict()
    assert 99 not in events["aid"]
    assert 98 not in events["aid"]
    assert 97 not in events["aid"]
    assert 96 not in events["aid"]
    assert set(labels["aid"]) == {96, 97, 98, 99}

    examples = pq.read_table(output / "examples.parquet").to_pydict()
    assert len(examples["session"]) == 2
    assert all(0 <= fold < 2 for fold in examples["fold"])


def test_cache_rejects_misaligned_validation_streams(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    output = tmp_path / "cache"
    _write_validation_fixture(validation)
    labels_path = validation / "test_labels.jsonl"
    records = [orjson.loads(line) for line in labels_path.read_bytes().splitlines()]
    records[1]["session"] = 12
    labels_path.write_bytes(
        b"\n".join(orjson.dumps(record) for record in records) + b"\n"
    )

    with pytest.raises(RuntimeError, match="misaligned"):
        build_ranking_training_cache(
            validation,
            output,
            logger=logging.getLogger("test"),
            buckets=4,
            folds=2,
            heartbeat_seconds=3600,
        )
