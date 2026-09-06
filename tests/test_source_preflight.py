from __future__ import annotations

from pathlib import Path

import pytest

from otto_recsys.cloud.source_preflight import load_pinned_quality_toolchain


def test_source_preflight_requires_all_exact_quality_pins(tmp_path: Path) -> None:
    (tmp_path / "requirements-dev.txt").write_text(
        "ruff==0.16.6\nmypy==2.3.1\npytest==9.0.3\n",
        encoding="utf-8",
    )
    assert load_pinned_quality_toolchain(tmp_path) == {
        "ruff": "0.16.6",
        "mypy": "2.3.1",
        "pytest": "9.0.3",
    }


def test_source_preflight_fails_closed_on_missing_pin(tmp_path: Path) -> None:
    (tmp_path / "requirements-dev.txt").write_text(
        "ruff==0.16.6\npytest==9.0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing exact quality-tool pins"):
        load_pinned_quality_toolchain(tmp_path)


def test_fold_launcher_requires_exact_source_preflight() -> None:
    launcher = Path("scripts/launch_two_tower_fold.py").read_text(encoding="utf-8")
    assert "run_exact_source_preflight(source_root)" in launcher
    assert "verify_uploaded_source_roundtrip" in launcher
    assert "ensure_committed_remote_state()" in launcher
