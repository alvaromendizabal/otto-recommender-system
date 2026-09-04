import json
import logging
from pathlib import Path

from otto_recsys.data.manifest import (
    build_manifest,
    hash_file,
    write_manifest,
)


def test_hash_file_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "example.txt"
    source.write_text("otto\n", encoding="utf-8")

    first = hash_file(
        source,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )
    second = hash_file(
        source,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )

    assert first == second
    assert first.name == "example.txt"
    assert first.size_bytes == 5
    assert len(first.sha256) == 64


def test_build_manifest_accepts_covariant_path_sequence(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    paths: tuple[Path, ...] = (first, second)

    manifest = build_manifest(
        paths,
        source="unit-test",
        logger=logging.getLogger("test"),
    )

    assert len(manifest.files) == 2
    assert len(manifest.manifest_id) == 64


def test_manifest_id_depends_on_content_not_input_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    logger = logging.getLogger("test")

    manifest_a = build_manifest(
        [first, second],
        source="unit-test",
        logger=logger,
    )
    manifest_b = build_manifest(
        [second, first],
        source="unit-test",
        logger=logger,
    )

    assert manifest_a.manifest_id == manifest_b.manifest_id


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

    assert loaded["manifest_id"] == manifest.manifest_id
    assert loaded["source"] == "unit-test"
    assert len(loaded["files"]) == 1
