import logging
from pathlib import Path

import pyarrow.parquet as pq

from otto_recsys.data.convert import convert_jsonl_to_parquet
from otto_recsys.data.manifest import build_manifest, write_manifest


def make_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "train.jsonl"

    source.write_text(
        '{"session":1,"events":['
        '{"aid":10,"ts":1000,"type":"clicks"},'
        '{"aid":11,"ts":2000,"type":"carts"},'
        '{"aid":12,"ts":3000,"type":"orders"}'
        ']}\n'
        '{"session":2,"events":['
        '{"aid":20,"ts":4000,"type":"clicks"},'
        '{"aid":21,"ts":5000,"type":"orders"}'
        ']}\n',
        encoding="utf-8",
    )

    raw_manifest = build_manifest(
        [source],
        source="unit-test",
        logger=logging.getLogger("test"),
    )

    manifest_path = tmp_path / "raw-manifest.json"
    write_manifest(raw_manifest, manifest_path)

    return source, manifest_path


def test_conversion_preserves_all_events(tmp_path: Path) -> None:
    source, raw_manifest = make_source(tmp_path)
    output = tmp_path / "processed"

    manifest = convert_jsonl_to_parquet(
        source,
        output,
        raw_manifest,
        logger=logging.getLogger("test"),
        events_per_part=3,
        heartbeat_seconds=10.0,
    )

    assert manifest.status == "complete"
    assert manifest.sessions_processed == 2
    assert manifest.events_processed == 5
    assert manifest.parts_written == 2

    rows = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in sorted(output.glob("part-*.parquet"))
    )

    assert rows == 5


def test_conversion_resumes_from_partial_manifest(
    tmp_path: Path,
) -> None:
    source, raw_manifest = make_source(tmp_path)
    output = tmp_path / "processed"

    partial = convert_jsonl_to_parquet(
        source,
        output,
        raw_manifest,
        logger=logging.getLogger("test"),
        events_per_part=100,
        heartbeat_seconds=10.0,
        max_sessions=1,
    )

    assert partial.status == "partial"
    assert partial.sessions_processed == 1
    assert partial.events_processed == 3

    complete = convert_jsonl_to_parquet(
        source,
        output,
        raw_manifest,
        logger=logging.getLogger("test"),
        events_per_part=100,
        heartbeat_seconds=10.0,
    )

    assert complete.status == "complete"
    assert complete.sessions_processed == 2
    assert complete.events_processed == 5

    rows = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in sorted(output.glob("part-*.parquet"))
    )

    assert rows == 5
