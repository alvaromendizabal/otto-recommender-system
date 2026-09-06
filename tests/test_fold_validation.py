from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from otto_recsys.cloud.fold_validation import (
    QUALITY_TOOLS,
    exact_source_preflight_command,
    gpu_mypy_command,
    gpu_pytest_command,
    gpu_ruff_command,
    load_quality_pins,
    root_sync_command,
    safe_archive_members,
    validate_archive_text_hygiene,
)


def test_quality_pins_are_exact_and_complete(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements-dev.txt"
    requirements.write_text(
        "-r requirements.txt\nruff==0.16.6\nmypy==2.3.1\npytest==9.0.3\n",
        encoding="utf-8",
    )
    pins = load_quality_pins(requirements)
    assert pins == {"ruff": "0.16.6", "mypy": "2.3.1", "pytest": "9.0.3"}


def test_quality_pin_contract_fails_closed(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements-dev.txt"
    requirements.write_text("ruff==0.16.6\npytest==9.0.3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing exact quality-tool pins"):
        load_quality_pins(requirements)


def test_root_sync_is_frozen_and_installs_dev_plus_ml_extras() -> None:
    assert root_sync_command() == [
        "uv",
        "sync",
        "--frozen",
        "--extra",
        "dev",
        "--extra",
        "ml",
    ]


def test_gpu_static_commands_use_stage_python(tmp_path: Path) -> None:
    python_path = tmp_path / ".venv" / "bin" / "python"
    assert gpu_ruff_command(python_path)[:3] == [
        str(python_path),
        "-m",
        "ruff",
    ]
    assert gpu_mypy_command(python_path)[:3] == [
        str(python_path),
        "-m",
        "mypy",
    ]


def test_gpu_pytest_isolated_at_exact_package_version() -> None:
    command = gpu_pytest_command("9.0.3")
    assert command[:7] == [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        "pytest==9.0.3",
        "python",
    ]
    assert command[7:11] == ["-m", "pytest", "-q", "tests/test_resume_contract.py"]


def test_archive_hygiene_rejects_cache(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("src/pkg/__pycache__/x.pyc", b"bad")
    with pytest.raises(RuntimeError, match="generated cache"):
        safe_archive_members(archive_path)


def test_quality_tool_order_is_stable() -> None:
    assert QUALITY_TOOLS == ("ruff", "mypy", "pytest")


def test_validator_uses_locked_complete_environment() -> None:
    validator = Path("scripts/validate_two_tower_fold.py").read_text(encoding="utf-8")
    assert '"--frozen", "--extra", "dev", "--extra", "ml"' in validator
    assert "real_install_pinned_quality_toolchain" not in validator
    assert '"uv", "pip", "install"' not in validator


def test_archive_text_hygiene_rejects_extra_eof_blank_line(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("tests/test_example.py", "def test_ok():\n    assert True\n\n")
    with pytest.raises(RuntimeError, match="exactly one newline"):
        validate_archive_text_hygiene(archive_path)


def test_archive_text_hygiene_rejects_trailing_whitespace(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("src/example.py", "value = 1 \n")
    with pytest.raises(RuntimeError, match="trailing whitespace"):
        validate_archive_text_hygiene(archive_path)


def test_archive_text_hygiene_accepts_canonical_text(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("src/example.py", "value = 1\n")
    validate_archive_text_hygiene(archive_path)


def test_validator_checks_whitespace_before_environment_bootstrap() -> None:
    source = Path("src/otto_recsys/cloud/fold_validation.py").read_text(
        encoding="utf-8"
    )
    whitespace = source.index('"worktree_whitespace_preflight"')
    bootstrap = source.index("bootstrap_stage(stage_root)")
    assert whitespace < bootstrap

def test_exact_source_preflight_command_calls_launcher_preflight(tmp_path: Path) -> None:
    python_path = tmp_path / ".venv" / "bin" / "python"
    command = exact_source_preflight_command(python_path)
    assert command[:2] == [str(python_path), "-c"]
    assert "run_exact_source_preflight" in command[2]
    assert "gpu/two_tower" in command[2]


def test_validator_executes_exact_launcher_source_preflight() -> None:
    source = Path("src/otto_recsys/cloud/fold_validation.py").read_text(
        encoding="utf-8"
    )
    assert '"exact_launcher_source_preflight"' in source
    assert "exact_source_preflight_command(python_path)" in source
