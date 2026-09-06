from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

QUALITY_TOOLS = ("ruff", "mypy", "pytest")
ROOT_EXTRAS = ("dev", "ml")
PHASE_TESTS = (
    "tests/test_two_tower_fold.py",
    "tests/test_retrieval_evaluation.py",
    "tests/test_source_preflight.py",
    "tests/test_two_tower_portfolio.py",
    "tests/test_fold_validation.py",
)
GPU_TESTS = (
    "tests/test_resume_contract.py",
    "tests/test_sagemaker_entrypoint.py",
)


@dataclass(frozen=True)
class ValidationResult:
    stage: str
    return_code: int
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.return_code == 0


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_quality_pins(requirements_path: Path) -> dict[str, str]:
    if not requirements_path.is_file():
        raise RuntimeError(f"missing quality-tool requirements: {requirements_path}")

    pins: dict[str, str] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        if "==" not in line:
            continue
        name, version = (part.strip() for part in line.split("==", maxsplit=1))
        if name in QUALITY_TOOLS:
            pins[name] = version

    missing = sorted(set(QUALITY_TOOLS) - pins.keys())
    if missing:
        raise RuntimeError(
            f"missing exact quality-tool pins in {requirements_path}: {missing}"
        )
    return pins


def safe_archive_members(archive_path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(archive_path) as archive:
        members = tuple(info.filename for info in archive.infolist())

    if not members:
        raise RuntimeError("bundle is empty")

    forbidden_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for member in members:
        path = Path(member)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive path: {member}")
        if path.suffix == ".pyc" or forbidden_parts.intersection(path.parts):
            raise RuntimeError(f"generated cache is forbidden in bundle: {member}")
    return members


def run_stage(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> ValidationResult:
    started = time.perf_counter()
    print(
        f"[{utc_now()}] FOLD_VALIDATION_STAGE_START name={name} "
        f"cwd={cwd} command={json.dumps(list(command))}",
        flush=True,
    )
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=process_env,
        check=False,
        text=True,
    )
    elapsed = time.perf_counter() - started
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(
        f"[{utc_now()}] FOLD_VALIDATION_STAGE_COMPLETE name={name} "
        f"status={status} return_code={completed.returncode} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return ValidationResult(name, completed.returncode, elapsed)


def assert_success(result: ValidationResult) -> None:
    if not result.passed:
        raise RuntimeError(
            f"validation stage failed: {result.stage} rc={result.return_code}"
        )


def stage_python(stage_root: Path) -> Path:
    candidate = stage_root / ".venv" / "bin" / "python"
    if not candidate.is_file():
        candidate = stage_root / ".venv" / "bin" / "python3"
    if not candidate.is_file():
        raise RuntimeError(
            "clean-worktree interpreter not found under " f"{stage_root}/.venv"
        )
    return candidate


def root_sync_command() -> list[str]:
    """Install the complete lockfile-governed control-plane environment."""
    command = ["uv", "sync", "--frozen"]
    for extra in ROOT_EXTRAS:
        command.extend(["--extra", extra])
    return command


def package_origin_probe(stage_root: Path) -> str:
    return (
        "from pathlib import Path; import otto_recsys; "
        "repo=Path.cwd().resolve(); p=Path(otto_recsys.__file__).resolve(); "
        "print(f'otto_recsys_file={p}'); "
        "assert repo in p.parents, f'package outside stage: {p}'"
    )


def root_tool_version_commands(python_path: Path) -> tuple[list[str], ...]:
    return tuple(
        [str(python_path), "-m", tool, "--version"] for tool in QUALITY_TOOLS
    )


def exact_static_tool_probe(python_path: Path, pins: dict[str, str]) -> list[str]:
    program = (
        "from importlib.metadata import version; "
        f"expected={{'ruff': {pins['ruff']!r}, 'mypy': {pins['mypy']!r}}}; "
        "observed={name: version(name) for name in expected}; "
        "print('gpu_static_tool_versions=' + str(observed)); "
        "assert observed == expected, (expected, observed)"
    )
    return [str(python_path), "-c", program]


def phase_test_command(python_path: Path) -> list[str]:
    return [str(python_path), "-m", "pytest", "-q", *PHASE_TESTS]


def gpu_ruff_command(python_path: Path) -> list[str]:
    return [
        str(python_path),
        "-m",
        "ruff",
        "check",
        "--config",
        "pyproject.toml",
        ".",
    ]


def gpu_mypy_command(python_path: Path) -> list[str]:
    return [
        str(python_path),
        "-m",
        "mypy",
        "--config-file",
        "pyproject.toml",
        "--python-version",
        "3.13",
        "otto_two_tower",
        "train.py",
        "prepare.py",
        "runtime_validation.py",
        "sagemaker_entrypoint.py",
    ]


def gpu_pytest_command(pytest_version: str) -> list[str]:
    return [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        f"pytest=={pytest_version}",
        "python",
        "-m",
        "pytest",
        "-q",
        *GPU_TESTS,
    ]


def canonical_bundle_paths(members: Iterable[str]) -> list[str]:
    return sorted(
        member
        for member in members
        if member and not member.endswith("/") and not member.startswith(".")
    )


def extract_bundle(archive_path: Path, destination: Path) -> None:
    members = safe_archive_members(archive_path)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    if not canonical_bundle_paths(members):
        raise RuntimeError("bundle contains no source files")


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def assert_repo_baseline(repo: Path, expected_head: str) -> None:
    actual = git_output(repo, "rev-parse", "--short", "HEAD")
    if actual != expected_head:
        raise RuntimeError(f"HEAD mismatch: expected={expected_head} actual={actual}")
    if git_output(repo, "status", "--porcelain"):
        raise RuntimeError("working tree must be clean before Fold 0 integration")


def validate_archive_sha(archive_path: Path, expected_sha256: str) -> str:
    observed = sha256_file(archive_path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"bundle SHA-256 mismatch: expected={expected_sha256} observed={observed}"
        )
    return observed


def make_stage_path(repo: Path) -> Path:
    root = Path(tempfile.mkdtemp(prefix="otto-fold0-worktree-", dir=repo.parent))
    root.rmdir()
    return root


def remove_worktree(repo: Path, stage_root: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(stage_root)],
        cwd=repo,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(stage_root, ignore_errors=True)


def bootstrap_stage(stage_root: Path) -> Path:
    assert_success(
        run_stage(
            "uv_sync_frozen_dev_ml",
            root_sync_command(),
            cwd=stage_root,
        )
    )
    return stage_python(stage_root)


def validate_stage(stage_root: Path, pins: dict[str, str]) -> None:
    python_path = stage_python(stage_root)

    for tool, command in zip(
        QUALITY_TOOLS, root_tool_version_commands(python_path), strict=True
    ):
        assert_success(run_stage(f"root_tool_version_{tool}", command, cwd=stage_root))

    assert_success(
        run_stage(
            "package_origin",
            [str(python_path), "-c", package_origin_probe(stage_root)],
            cwd=stage_root,
        )
    )
    assert_success(
        run_stage(
            "root_quality_gate",
            [str(python_path), "scripts/run_quality_gate.py"],
            cwd=stage_root,
        )
    )
    assert_success(
        run_stage("phase6a_tests", phase_test_command(python_path), cwd=stage_root)
    )

    # Root validation is governed by the project's frozen dev+ml extras.
    # GPU package validation is deliberately isolated: Ruff/mypy must match
    # the exact GPU-package pins, while pytest runs in a throwaway exact-version
    # environment so we never mutate the root lockfile-governed environment.
    assert_success(
        run_stage(
            "gpu_static_tool_pin_contract",
            exact_static_tool_probe(python_path, pins),
            cwd=stage_root,
        )
    )

    gpu_root = stage_root / "gpu" / "two_tower"
    assert_success(
        run_stage(
            "gpu_ruff",
            gpu_ruff_command(python_path),
            cwd=gpu_root,
        )
    )
    assert_success(
        run_stage(
            "gpu_mypy",
            gpu_mypy_command(python_path),
            cwd=gpu_root,
        )
    )

    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    gpu_pythonpath = str(gpu_root)
    if inherited_pythonpath:
        gpu_pythonpath = f"{gpu_pythonpath}{os.pathsep}{inherited_pythonpath}"
    assert_success(
        run_stage(
            "gpu_contract_tests_exact_pytest",
            gpu_pytest_command(pins["pytest"]),
            cwd=gpu_root,
            env={"PYTHONPATH": gpu_pythonpath},
        )
    )
    assert_success(
        run_stage(
            "dependency_integrity",
            ["uv", "pip", "check", "--python", str(python_path)],
            cwd=stage_root,
        )
    )
    assert_success(run_stage("whitespace", ["git", "diff", "--check"], cwd=stage_root))


def validate_clean_worktree(
    *, repo: Path, archive_path: Path, expected_head: str, expected_sha256: str
) -> tuple[str, tuple[str, ...]]:
    repo = repo.resolve()
    archive_path = archive_path.resolve()
    assert_repo_baseline(repo, expected_head)
    observed_sha = validate_archive_sha(archive_path, expected_sha256)
    members = safe_archive_members(archive_path)
    stage_root = make_stage_path(repo)
    print(
        f"[{utc_now()}] FOLD_VALIDATION_WORKTREE_CREATE path={stage_root}",
        flush=True,
    )
    try:
        assert_success(
            run_stage(
                "git_worktree_add",
                ["git", "worktree", "add", "--detach", str(stage_root), expected_head],
                cwd=repo,
            )
        )
        extract_bundle(archive_path, stage_root)
        pins = load_quality_pins(
            stage_root / "gpu" / "two_tower" / "requirements-dev.txt"
        )
        print(
            f"[{utc_now()}] FOLD_VALIDATION_TOOLCHAIN "
            + " ".join(f"{name}={pins[name]}" for name in QUALITY_TOOLS),
            flush=True,
        )
        bootstrap_stage(stage_root)
        validate_stage(stage_root, pins)
        print(f"[{utc_now()}] OTTO_FOLD0_CLEAN_WORKTREE_VALIDATION_PASSED", flush=True)
    finally:
        remove_worktree(repo, stage_root)
    return observed_sha, members
