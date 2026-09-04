import json
import logging
from pathlib import Path

from otto_recsys.data.manifest import (
    build_manifest,
    hash_file,
    write_manifest,
)


def test_hash_file(tmp_path: Path) -> None:
    source = tmp_path / "example.txt"
    source.write_text("otto\n", encoding="utf-8")

    record = hash_file(
        source,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )

    assert record.name == "example.txt"
    assert record.size_bytes == 5
    assert len(record.sha256) == 64


def test_manifest_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "example.txt"
    source.write_text("otto\n", encoding="utf-8")

    manifest = build_manifest(
        [source],
        source="unit-test",
        logger=logging.getLogger("test"),
    )

    destination = tmp_path / "manifest.json"
    write_manifest(manifest, destination)

    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded["source"] == "unit-test"
    assert len(loaded["files"]) == 1
