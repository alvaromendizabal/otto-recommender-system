from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

import pytest

from otto_recsys.cloud.sagemaker_pipeline import (
    create_deterministic_source_archive,
    verify_source_archive,
)


def test_source_archive_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("print('a')\n", encoding="utf-8")
    nested = source / "package"
    nested.mkdir()
    (nested / "b.py").write_text("VALUE = 1\n", encoding="utf-8")

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_sha = create_deterministic_source_archive(source, first)
    second_sha = create_deterministic_source_archive(source, second)

    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()


def test_source_archive_excludes_generated_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "train.cpython-313.pyc").write_bytes(b"generated")

    archive = tmp_path / "source.tar.gz"
    create_deterministic_source_archive(source, archive)

    with gzip.open(archive, "rb") as compressed, tarfile.open(
        fileobj=compressed, mode="r:"
    ) as payload:
        names = payload.getnames()

    assert names == ["train.py"]


def test_source_archive_verification_proves_exact_byte_parity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "requirements.txt").write_text("numpy>=2\n", encoding="utf-8")

    archive = tmp_path / "source.tar.gz"
    create_deterministic_source_archive(source, archive)
    verification = verify_source_archive(source, archive)

    assert verification["status"] == "passed"
    assert verification["files"] == 2
    assert len(verification["manifest_sha256"]) == 64


def test_source_archive_verification_detects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = source / "train.py"
    train.write_text("VALUE = 1\n", encoding="utf-8")

    archive = tmp_path / "source.tar.gz"
    create_deterministic_source_archive(source, archive)
    train.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source archive differs from source tree"):
        verify_source_archive(source, archive)


def test_source_archive_rejects_symlinked_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    (source / "linked.py").symlink_to(outside)

    archive = tmp_path / "source.tar.gz"
    create_deterministic_source_archive(source, archive)

    with pytest.raises(ValueError, match="may not contain symlinks"):
        verify_source_archive(source, archive)
