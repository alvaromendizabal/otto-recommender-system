from __future__ import annotations

from pathlib import Path

import pytest

from otto_recsys.cloud.source_preflight import (
    CPU_SAFE_TESTS,
    load_pinned_quality_toolchain,
)


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


def test_cpu_safe_source_preflight_excludes_torch_checkpoint_test() -> None:
    assert CPU_SAFE_TESTS == (
        "tests/test_resume_contract.py",
        "tests/test_sagemaker_entrypoint.py",
    )
    assert "tests/test_checkpoint.py" not in CPU_SAFE_TESTS


def test_gpu_runtime_validation_owns_checkpoint_roundtrip() -> None:
    runtime_validation = Path("gpu/two_tower/runtime_validation.py").read_text(
        encoding="utf-8"
    )
    assert "load_checkpoint(" in runtime_validation
    assert "CHECKPOINT_RNG_ROUNDTRIP_PASSED" in runtime_validation


def test_source_preflight_import_boundary_matches_ruff_isort_contract() -> None:
    source = Path("src/otto_recsys/cloud/source_preflight.py").read_text(encoding="utf-8")
    marker = "from otto_recsys.cloud.sagemaker_pipeline import verify_source_archive"
    assert f"{marker}\n\nCPU_SAFE_TESTS = (" in source
    assert f"{marker}\n\n\nCPU_SAFE_TESTS = (" not in source
