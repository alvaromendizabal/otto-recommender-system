from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

from otto_recsys.cloud.sagemaker_pipeline import create_deterministic_source_archive


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
