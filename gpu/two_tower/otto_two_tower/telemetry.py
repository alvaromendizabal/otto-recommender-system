from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

import psutil


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = completed.stdout.strip().splitlines()[0]
        fields = [value.strip() for value in first.split(",")]
        return {
            "gpu_util_percent": float(fields[0]),
            "gpu_memory_used_mb": float(fields[1]),
            "gpu_memory_total_mb": float(fields[2]),
            "gpu_temperature_c": float(fields[3]),
            "gpu_power_w": float(fields[4]),
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return {
            "gpu_util_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
        }


class TrainingHeartbeat:
    def __init__(
        self,
        logger: logging.Logger,
        *,
        stage: str,
        interval_seconds: float,
        progress_provider: Callable[[], dict[str, Any]],
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._logger = logger
        self._stage = stage
        self._interval_seconds = interval_seconds
        self._progress_provider = progress_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> TrainingHeartbeat:
        self._started = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{self._stage}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 6.0)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(self._interval_seconds):
            gpu = _gpu_snapshot()
            progress = self._progress_provider()
            used = gpu["gpu_memory_used_mb"]
            total = gpu["gpu_memory_total_mb"]
            util = gpu["gpu_util_percent"]
            if used is None or total is None or util is None:
                event = "heartbeat gpu=unavailable"
            else:
                event = (
                    f"heartbeat gpu_util={util:.0f}% "
                    f"vram={used / 1024:.1f}/{total / 1024:.1f}GiB"
                )
            extra: dict[str, Any] = {
                "event": event,
                "stage": self._stage,
                "elapsed_seconds": round(time.perf_counter() - self._started, 1),
                "rss_mb": round(process.memory_info().rss / (1024**2), 1),
                "cpu_percent": process.cpu_percent(interval=None),
                **gpu,
                **progress,
            }
            self._logger.info("heartbeat", extra=extra)
