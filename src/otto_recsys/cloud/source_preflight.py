from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from otto_recsys.cloud.sagemaker_pipeline import verify_source_archive

CPU_SAFE_TESTS = (
    "tests/test_resume_contract.py",
    "tests/test_sagemaker_entrypoint.py",
    "tests/test_evaluation_cli.py",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_command(
    command: list[str],
    *,
    check: bool = False,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=process_env,
        input=input_text,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


def validate_evaluation_launch(source_root: Path, definition: dict[str, Any]) -> None:
    """Check the actual pipeline parameters against its packaged worker parser."""
    started = time.perf_counter()
    print(f"[{utc_now()}] evaluation_launch_contract_start", flush=True)
    parameters = definition["Steps"][0]["Arguments"]["HyperParameters"]
    completed = run_command(
        [sys.executable, "-m", "otto_two_tower.evaluation_cli"],
        cwd=source_root.resolve(),
        env={"PYTHONPATH": str(source_root.resolve())},
        input_text=json.dumps(parameters),
    )
    elapsed = time.perf_counter() - started
    status = "passed" if completed.returncode == 0 else "failed"
    print(
        f"[{utc_now()}] evaluation_launch_contract_complete "
        f"status={status} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    if completed.returncode:
        raise RuntimeError(f"evaluation launch contract rejected: {completed.stderr.strip()}")
    print(completed.stdout.rstrip(), flush=True)


def load_pinned_quality_toolchain(source_root: Path) -> dict[str, str]:
    requirements_path = source_root / "requirements-dev.txt"
    if not requirements_path.is_file():
        raise RuntimeError(f"missing GPU quality-tool requirements: {requirements_path}")

    required_tools = {"ruff", "mypy", "pytest"}
    versions: dict[str, str] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        if "==" not in line:
            continue
        name, version = (part.strip() for part in line.split("==", maxsplit=1))
        if name in required_tools:
            versions[name] = version

    missing = sorted(required_tools - versions.keys())
    if missing:
        raise RuntimeError(f"missing exact quality-tool pins in {requirements_path}: {missing}")
    return versions


def _tool_command(
    *, distribution: str, version: str, executable: str, arguments: list[str]
) -> list[str]:
    if shutil.which("uvx"):
        runner = ["uvx"]
    elif shutil.which("uv"):
        runner = ["uv", "tool", "run"]
    else:
        raise RuntimeError("uv/uvx is required for pinned source checks")
    return [
        *runner,
        "--from",
        f"{distribution}=={version}",
        executable,
        *arguments,
    ]


def _python_module_command(
    *, distribution: str, version: str, module: str, arguments: list[str]
) -> list[str]:
    if not shutil.which("uv"):
        raise RuntimeError("uv is required for pinned Python-module source checks")
    return [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        f"{distribution}=={version}",
        "python",
        "-m",
        module,
        *arguments,
    ]


def _run_stage(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    started = time.perf_counter()
    print(f"[{utc_now()}] source_preflight_stage_start name={name}", flush=True)
    completed = run_command(command, cwd=cwd, env=env)
    elapsed = time.perf_counter() - started
    if completed.stdout.strip():
        print(completed.stdout.rstrip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"source preflight stage failed: {name} rc={completed.returncode}")
    print(
        f"[{utc_now()}] source_preflight_stage_complete "
        f"name={name} status=passed elapsed_seconds={elapsed:.3f}",
        flush=True,
    )


def run_exact_source_preflight(source_root: Path) -> None:
    source_root = source_root.resolve()
    toolchain = load_pinned_quality_toolchain(source_root)
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    source_pythonpath = str(source_root)
    if inherited_pythonpath:
        source_pythonpath = f"{source_pythonpath}{os.pathsep}{inherited_pythonpath}"

    print(
        f"[{utc_now()}] source_preflight_toolchain "
        f"ruff={toolchain['ruff']} mypy={toolchain['mypy']} "
        f"pytest={toolchain['pytest']}",
        flush=True,
    )
    _run_stage(
        "compile",
        [sys.executable, "-m", "compileall", "-q", "."],
        cwd=source_root,
    )
    _run_stage(
        "ruff",
        _tool_command(
            distribution="ruff",
            version=toolchain["ruff"],
            executable="ruff",
            arguments=["check", "--config", "pyproject.toml", "."],
        ),
        cwd=source_root,
    )
    _run_stage(
        "mypy",
        _tool_command(
            distribution="mypy",
            version=toolchain["mypy"],
            executable="mypy",
            arguments=[
                "--config-file",
                "pyproject.toml",
                "--python-version",
                "3.13",
                "otto_two_tower",
                "train.py",
                "evaluate.py",
                "prepare.py",
                "runtime_validation.py",
                "sagemaker_entrypoint.py",
            ],
        ),
        cwd=source_root,
    )
    _run_stage(
        "cpu_safe_contract_tests",
        _python_module_command(
            distribution="pytest",
            version=toolchain["pytest"],
            module="pytest",
            arguments=["-q", *CPU_SAFE_TESTS],
        ),
        cwd=source_root,
        env={"PYTHONPATH": source_pythonpath},
    )
    print(f"[{utc_now()}] source_preflight_complete status=passed", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_uploaded_source_roundtrip(
    *, source_root: Path, source_uri: str, expected_sha256: str
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="otto-source-roundtrip-") as tmpdir:
        downloaded = Path(tmpdir) / "source.tar.gz"
        run_command(
            ["aws", "s3", "cp", source_uri, str(downloaded), "--only-show-errors"],
            check=True,
        )
        observed_sha256 = sha256_file(downloaded)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                "S3 source round-trip SHA-256 mismatch: "
                f"expected={expected_sha256} observed={observed_sha256}"
            )
        verification = verify_source_archive(source_root, downloaded)
    return {
        **verification,
        "archive_sha256": observed_sha256,
        "s3_roundtrip": "passed",
    }
