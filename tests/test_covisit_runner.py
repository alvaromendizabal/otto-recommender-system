from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.build_covisit as runner
from otto_recsys.experiments.manifest import sha256_file


def test_safe_memory_budget_for_32_gib_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Memory:
        total = 32 * 1024**3
        available = 28 * 1024**3

    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: Memory(),
    )

    assert runner.safe_duckdb_memory_limit() == "8GB"


def test_small_host_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Memory:
        total = 4 * 1024**3
        available = 3 * 1024**3

    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: Memory(),
    )

    with pytest.raises(
        RuntimeError,
        match="larger instance",
    ):
        runner.safe_duckdb_memory_limit()


def test_low_available_memory_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Memory:
        total = 32 * 1024**3
        available = 4 * 1024**3

    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: Memory(),
    )

    with pytest.raises(
        RuntimeError,
        match="Close other workloads",
    ):
        runner.safe_duckdb_memory_limit()


def test_completed_matrix_requires_matching_hash(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "time.parquet"
    manifest = tmp_path / "time.json"

    matrix.write_bytes(b"valid-matrix")

    manifest.write_text(
        json.dumps(
            {
                "rows": 10,
                "output_sha256": sha256_file(matrix),
            }
        ),
        encoding="utf-8",
    )

    assert runner.completed_matrix_is_valid(
        matrix,
        manifest,
    )


def test_hash_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "time.parquet"
    manifest = tmp_path / "time.json"

    matrix.write_bytes(b"matrix")

    manifest.write_text(
        json.dumps(
            {
                "rows": 10,
                "output_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert not runner.completed_matrix_is_valid(
        matrix,
        manifest,
    )
