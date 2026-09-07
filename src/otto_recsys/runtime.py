from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil


def process_rss_mb(pid: int | None = None) -> float:
    """Return process RSS in MiB."""
    # /proc/self follows the caller even when a container exposes host process
    # IDs in /proc. Looking up os.getpid() there can fail or identify another
    # process. Use this live RSS counter for self on Linux; retain psutil for
    # other platforms and explicitly requested processes.
    if pid is None:
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return float(round(resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024**2), 1))
        except (OSError, ValueError, IndexError, AttributeError):
            pass
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
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        self._logger = logger
        self._stage = stage
        self._interval_seconds = interval_seconds
        self._pid_provider = pid_provider
        self._progress_provider = progress_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._cpu_sample: tuple[tuple[int | None, float], float, float] | None = None

    def __enter__(self) -> Heartbeat:
        self._stop.clear()
        self._cpu_sample = None
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

    def _sample_resources(self) -> dict[str, float | None]:
        """Measure CPU-time deltas; 100% means one fully occupied CPU core."""
        pid = self._pid_provider() if self._pid_provider is not None else None
        try:
            identity: tuple[int | None, float]
            if pid is None:
                identity = (None, 0.0)
                cpu_seconds = time.process_time()
                rss_mb = process_rss_mb()
            else:
                process = psutil.Process(pid)
                identity = (pid, process.create_time())
                times = process.cpu_times()
                cpu_seconds = times.user + times.system
                rss_mb = float(round(process.memory_info().rss / (1024**2), 1))
            now = time.perf_counter()
            previous = self._cpu_sample
            self._cpu_sample = (identity, now, cpu_seconds)
            percent = None
            if previous is not None and previous[0] == identity and now > previous[1]:
                percent = round(100 * max(0, cpu_seconds - previous[2]) / (now - previous[1]), 1)
            return {"rss_mb": rss_mb, "cpu_percent": percent}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self._cpu_sample = None
            return {"rss_mb": None, "cpu_percent": None}

    def _run(self) -> None:
        # Seed cumulative CPU time before the first wait. A new psutil.Process
        # with cpu_percent(None) on every tick always returns a meaningless zero.
        self._sample_resources()
        while not self._stop.wait(self._interval_seconds):
            extra: dict[str, Any] = {
                "event": "heartbeat",
                "stage": self._stage,
                "elapsed_seconds": round(time.perf_counter() - self._started_at, 1),
            }
            extra.update(self._sample_resources())

            if self._progress_provider is not None:
                extra.update(self._progress_provider())

            self._logger.info("heartbeat", extra=extra)
