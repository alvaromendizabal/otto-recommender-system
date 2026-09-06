from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]


STAGES = (
    Stage(
        "compile",
        (
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "otto_two_tower",
            "tests",
            "train.py",
            "prepare.py",
            "runtime_validation.py",
            "sagemaker_entrypoint.py",
        ),
    ),
    Stage(
        "ruff",
        (
            sys.executable,
            "-m",
            "ruff",
            "check",
            "otto_two_tower",
            "tests",
            "train.py",
            "prepare.py",
            "runtime_validation.py",
            "sagemaker_entrypoint.py",
        ),
    ),
    Stage(
        "mypy",
        (
            sys.executable,
            "-m",
            "mypy",
            "otto_two_tower",
            "train.py",
            "prepare.py",
            "runtime_validation.py",
            "sagemaker_entrypoint.py",
        ),
    ),
    Stage("pytest", (sys.executable, "-m", "pytest", "-q")),
)


def main() -> int:
    started = time.perf_counter()
    print(f"TWO_TOWER_QUALITY_GATE_START unix_seconds={time.time():.3f}")
    for stage in STAGES:
        stage_started = time.perf_counter()
        print(f"STAGE_START name={stage.name}", flush=True)
        completed = subprocess.run(stage.command, check=False)
        elapsed = time.perf_counter() - stage_started
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"STAGE_COMPLETE name={stage.name} status={status} elapsed_seconds={elapsed:.3f}")
        if completed.returncode != 0:
            print(f"OTTO_TWO_TOWER_QUALITY_GATE_FAILED stage={stage.name}")
            return completed.returncode
    total = time.perf_counter() - started
    print(f"OTTO_TWO_TOWER_QUALITY_GATE_PASSED total_seconds={total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
