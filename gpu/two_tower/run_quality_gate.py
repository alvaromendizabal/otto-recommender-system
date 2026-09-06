from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from otto_two_tower.logging_utils import configure_logging
from otto_two_tower.telemetry import TrainingHeartbeat


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
            "evaluate.py",
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
            "evaluate.py",
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
            "evaluate.py",
            "prepare.py",
            "runtime_validation.py",
            "sagemaker_entrypoint.py",
        ),
    ),
    Stage("pytest", (sys.executable, "-m", "pytest", "-q")),
)


def main() -> int:
    started = time.perf_counter()
    logger = configure_logging("neural_quality_gate", Path("/tmp/otto-neural-quality-logs"))
    print(f"[{datetime.now(UTC).isoformat()}] TWO_TOWER_QUALITY_GATE_START", flush=True)
    for stage in STAGES:
        stage_started = time.perf_counter()
        logger.info("stage_start", extra={"stage": stage.name})
        with TrainingHeartbeat(
            logger, stage=stage.name, interval_seconds=15, progress_provider=lambda: {}
        ):
            completed = subprocess.run(stage.command, check=False)
        elapsed = time.perf_counter() - stage_started
        status = "PASS" if completed.returncode == 0 else "FAIL"
        logger.info(
            "stage_complete",
            extra={"stage": stage.name, "status": status, "elapsed_seconds": round(elapsed, 3)},
        )
        if completed.returncode != 0:
            logger.error(
                "OTTO_TWO_TOWER_QUALITY_GATE_FAILED",
                extra={"stage": stage.name, "elapsed_seconds": time.perf_counter() - started},
            )
            return completed.returncode
    total = time.perf_counter() - started
    print(f"OTTO_TWO_TOWER_QUALITY_GATE_PASSED total_seconds={total:.3f}", flush=True)
    logger.info("quality_gate_complete", extra={"elapsed_seconds": round(total, 3)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
