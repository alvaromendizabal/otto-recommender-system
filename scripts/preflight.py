from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from otto_recsys.logging_utils import configure_logging
from otto_recsys.runtime import Heartbeat

EXPECTED_PACKAGES: dict[str, str] = {
    "numpy": "2.5.2",
    "orjson": "3.12.0",
    "polars": "1.44.1",
    "psutil": "7.2.2",
    "pyarrow": "25.0.1",
    "catboost": "1.2.10",
    "faiss-cpu": "1.15.0",
    "lightgbm": "4.7.0",
    "scikit-learn": "1.9.0",
    "xgboost": "3.4.1",
}


def run(command: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode, output[:4000]


def main() -> int:
    logger = configure_logging("preflight")
    started = time.perf_counter()
    findings: list[str] = []

    with Heartbeat(logger, stage="preflight", interval_seconds=15.0):
        python_version = platform.python_version()
        logger.info(
            "runtime",
            extra={
                "event": "runtime",
                "python": python_version,
                "executable": sys.executable,
                "cpu_count": psutil.cpu_count(logical=True),
                "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            },
        )
        if sys.version_info[:2] != (3, 13):
            findings.append(f"Expected Python 3.13.x, found {python_version}")

        for package, expected in EXPECTED_PACKAGES.items():
            try:
                observed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                observed = "missing"
            logger.info(
                "package",
                extra={
                    "event": "package",
                    "package": package,
                    "expected": expected,
                    "observed": observed,
                },
            )
            if observed != expected:
                findings.append(
                    f"{package}: expected {expected}, observed {observed}"
                )

        for command in ("git", "uv", "aws", "kaggle"):
            path = shutil.which(command)
            logger.info(
                "tool",
                extra={"event": "tool", "command": command, "path": path},
            )
            if path is None:
                findings.append(f"Missing command: {command}")

        if shutil.which("aws"):
            code, output = run(["aws", "sts", "get-caller-identity"])
            logger.info(
                "aws_identity",
                extra={"event": "aws_identity", "exit_code": code, "output": output},
            )
            if code != 0:
                findings.append("AWS identity check failed")

        if shutil.which("kaggle"):
            code, output = run(
                ["kaggle", "datasets", "files", "-d", "otto/recsys-dataset"],
                timeout=60,
            )
            logger.info(
                "kaggle_access",
                extra={"event": "kaggle_access", "exit_code": code, "output": output},
            )
            if code != 0:
                findings.append("Kaggle OTTO dataset access check failed")

        disk = shutil.disk_usage(Path.home())
        logger.info(
            "storage",
            extra={
                "event": "storage",
                "free_gb": round(disk.free / (1024**3), 1),
                "total_gb": round(disk.total / (1024**3), 1),
            },
        )

        gpu_contract = json.loads(
            Path("config/runtime/gpu.json").read_text(encoding="utf-8")
        )
        logger.info(
            "gpu_contract",
            extra={"event": "gpu_contract", **gpu_contract},
        )

    elapsed = round(time.perf_counter() - started, 3)

    print("\n=== PREFLIGHT SUMMARY ===")
    if findings:
        for finding in findings:
            print(f"- {finding}")
        print(f"Total seconds: {elapsed}")
        print("OTTO_PREFLIGHT_FAILED")
        return 1

    print(f"Total seconds: {elapsed}")
    print("OTTO_PREFLIGHT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
