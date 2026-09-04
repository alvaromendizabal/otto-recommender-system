from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import psutil


def process_rss_mb(pid: int | None = None) -> float:
    """Return process RSS in MiB."""
    process = psutil.Process(pid) if pid is not None else psutil.Process()
    return float(round(process.memory_info().rss / (1024**2), 1))


class Heartbeat:
    """Emit periodic liveness/resource telemetry."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        stage: str,
        interval_seconds: float = 15.0,
        pid_provider: Callable[[], int | None] | None = None,
        progress_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._logger = logger
        self._stage = stage
        self._interval_seconds = interval_seconds
        self._pid_provider = pid_provider
        self._progress_provider = progress_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> Heartbeat:
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{self._stage}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            pid = self._pid_provider() if self._pid_provider is not None else None
            extra: dict[str, Any] = {
                "event": "heartbeat",
                "stage": self._stage,
                "elapsed_seconds": round(time.perf_counter() - self._started_at, 1),
            }
            try:
                process = psutil.Process(pid) if pid is not None else psutil.Process()
                extra["rss_mb"] = float(round(process.memory_info().rss / (1024**2), 1))
                extra["cpu_percent"] = float(process.cpu_percent(interval=None))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                extra["rss_mb"] = None
                extra["cpu_percent"] = None

            if self._progress_provider is not None:
                extra.update(self._progress_provider())

            self._logger.info("heartbeat", extra=extra)
