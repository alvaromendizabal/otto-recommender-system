from __future__ import annotations

import gzip
import tarfile
import tomllib
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


def test_gpu_quality_toolchain_is_exactly_pinned() -> None:
    requirements = Path("gpu/two_tower/requirements-dev.txt").read_text(encoding="utf-8")

    assert "ruff==0.16.6" in requirements
    assert "mypy==2.3.1" in requirements
    assert "pytest==9.0.3" in requirements
    assert "ruff>=" not in requirements
    assert "mypy>=" not in requirements
    assert "pytest>=" not in requirements


def test_gpu_ruff_classifies_entrypoint_as_first_party() -> None:
    payload = tomllib.loads(
        Path("gpu/two_tower/pyproject.toml").read_text(encoding="utf-8")
    )
    known = payload["tool"]["ruff"]["lint"]["isort"]["known-first-party"]

    assert "otto_two_tower" in known
    assert "sagemaker_entrypoint" in known


def test_pinned_pytest_preflight_uses_python_module_invocation() -> None:
    launcher = Path("scripts/launch_two_tower_pipeline.py").read_text(encoding="utf-8")

    assert 'module="pytest"' in launcher
    assert '"python",\n        "-m",\n        module' in launcher
    assert 'executable="pytest"' not in launcher
    assert 'env={"PYTHONPATH": source_pythonpath}' in launcher


def test_changed_python_sources_have_single_terminal_newline() -> None:
    paths = [
        Path("gpu/two_tower/tests/test_sagemaker_entrypoint.py"),
        Path("tests/test_sagemaker_pipeline_launcher.py"),
    ]

    for path in paths:
        payload = path.read_bytes()
        assert payload.endswith(b"\n")
        assert not payload.endswith(b"\n\n")
