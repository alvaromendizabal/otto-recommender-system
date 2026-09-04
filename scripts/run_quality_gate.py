from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

from otto_recsys.logging_utils import configure_logging
from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]


STAGES: tuple[Stage, ...] = (
    Stage(
        "compile",
        (sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"),
    ),
    Stage(
        "ruff",
        (sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"),
    ),
    Stage(
        "mypy",
        (sys.executable, "-m", "mypy", "src", "scripts"),
    ),
    Stage(
        "pytest",
        (sys.executable, "-m", "pytest", "-q"),
    ),
    Stage(
        "smoke",
        (sys.executable, "scripts/run_smoke.py"),
    ),
)


def run_stage(stage: Stage) -> tuple[int, float]:
    logger = configure_logging("quality_gate")
    started = time.perf_counter()

    logger.info(
        "stage_start",
        extra={
            "event": "stage_start",
            "stage": stage.name,
            "command": list(stage.command),
        },
    )

    process = subprocess.Popen(
        stage.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        with Heartbeat(
            logger,
            stage=stage.name,
            interval_seconds=15.0,
            pid_provider=lambda: process.pid,
        ):
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()

    elapsed = round(time.perf_counter() - started, 3)
    logger.info(
        "stage_complete" if return_code == 0 else "stage_failed",
        extra={
            "event": "stage_complete" if return_code == 0 else "stage_failed",
            "stage": stage.name,
            "status": "passed" if return_code == 0 else "failed",
            "elapsed_seconds": elapsed,
            "exit_code": return_code,
        },
    )
    return return_code, elapsed


def main() -> int:
    logger = configure_logging("quality_gate")
    gate_started = time.perf_counter()
    results: list[tuple[str, int, float]] = []

    logger.info("quality_gate_start", extra={"event": "quality_gate_start"})

    for stage in STAGES:
        return_code, elapsed = run_stage(stage)
        results.append((stage.name, return_code, elapsed))
        if return_code != 0:
            break

    total_elapsed = round(time.perf_counter() - gate_started, 3)
    failed = [name for name, code, _ in results if code != 0]

    print("\n=== QUALITY GATE SUMMARY ===")
    for name, code, elapsed in results:
        status = "PASS" if code == 0 else "FAIL"
        print(f"{name:<12} {status:<4} {elapsed:>8.3f}s")
    print(f"total               {total_elapsed:>8.3f}s")

    if failed:
        print(f"\nOTTO_QUALITY_GATE_FAILED stage={failed[0]}")
        logger.error(
            "quality_gate_failed",
            extra={
                "event": "quality_gate_failed",
                "stage": failed[0],
                "elapsed_seconds": total_elapsed,
            },
        )
        return 1

    print("\nOTTO_QUALITY_GATE_PASSED")
    logger.info(
        "quality_gate_passed",
        extra={
            "event": "quality_gate_passed",
            "elapsed_seconds": total_elapsed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
