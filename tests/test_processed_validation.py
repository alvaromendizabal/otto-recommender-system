import logging
from pathlib import Path

from otto_recsys.data.convert import convert_jsonl_to_parquet
from otto_recsys.data.manifest import build_manifest, write_manifest
from otto_recsys.data.processed_validation import (
    validate_processed_dataset,
)


def test_processed_validation_matches_conversion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"

    source.write_text(
        '{"session":1,"events":['
        '{"aid":10,"ts":1000,"type":"clicks"},'
        '{"aid":11,"ts":2000,"type":"carts"}'
        ']}\n'
        '{"session":2,"events":['
        '{"aid":20,"ts":3000,"type":"clicks"},'
        '{"aid":21,"ts":4000,"type":"orders"}'
        ']}\n',
        encoding="utf-8",
    )

    raw_manifest = build_manifest(
        [source],
        source="unit-test",
        logger=logging.getLogger("test"),
    )

    raw_manifest_path = tmp_path / "raw-manifest.json"
    write_manifest(raw_manifest, raw_manifest_path)

    output = tmp_path / "processed"

    convert_jsonl_to_parquet(
        source,
        output,
        raw_manifest_path,
        logger=logging.getLogger("test"),
        events_per_part=2,
        heartbeat_seconds=10.0,
    )

    summary = validate_processed_dataset(
        output,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )

    assert summary.parts == 2
    assert summary.rows == 4
    assert summary.sessions == 2
    assert summary.min_ts == 1000
    assert summary.max_ts == 4000
