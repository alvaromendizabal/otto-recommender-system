import logging
import time

import pytest

from otto_recsys.runtime import Heartbeat, process_rss_mb


def test_process_rss_is_positive() -> None:
    assert process_rss_mb() > 0.0


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_heartbeat_rejects_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        Heartbeat(
            logging.getLogger("test"),
            stage="unit",
            interval_seconds=interval,
        )


def test_heartbeat_context_exits_cleanly() -> None:
    logger = logging.getLogger("heartbeat-test")
    with Heartbeat(logger, stage="unit", interval_seconds=0.01):
        time.sleep(0.03)


def test_self_rss_remains_available_when_numeric_pid_lookup_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    import psutil

    if not Path("/proc/self/statm").is_file():
        pytest.skip("Linux procfs contract")

    def unavailable(*args: object, **kwargs: object) -> None:
        raise psutil.NoSuchProcess(999999)

    monkeypatch.setattr(psutil, "Process", unavailable)
    assert process_rss_mb() > 0
    with pytest.raises(psutil.NoSuchProcess):
        process_rss_mb(999999)


def test_cpu_telemetry_measures_work_between_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    import otto_recsys.runtime as runtime

    clock = iter([10.0, 10.5, 11.0])
    cpu = iter([2.0, 2.2, 3.2])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runtime.time, "process_time", lambda: next(cpu))
    monkeypatch.setattr(runtime, "process_rss_mb", lambda: 64.0)
    beat = Heartbeat(logging.getLogger("cpu"), stage="work")
    assert beat._sample_resources() == {"rss_mb": 64.0, "cpu_percent": None}
    assert beat._sample_resources()["cpu_percent"] == pytest.approx(40.0)
    # Multithreaded native kernels can occupy more than one core.
    assert beat._sample_resources()["cpu_percent"] == pytest.approx(200.0)


def test_child_restart_and_missing_process_reset_cpu_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import psutil

    import otto_recsys.runtime as runtime

    state = {"created": 1.0, "cpu": 4.0, "missing": False}

    class Process:
        def create_time(self) -> float:
            return float(state["created"])

        def cpu_times(self) -> SimpleNamespace:
            return SimpleNamespace(user=state["cpu"], system=0)

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1024**2)

    def process(pid: int) -> Process:
        if state["missing"]:
            raise psutil.NoSuchProcess(pid)
        return Process()

    monkeypatch.setattr(runtime.psutil, "Process", process)
    clock = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(clock))
    beat = Heartbeat(logging.getLogger("child"), stage="work", pid_provider=lambda: 7)
    assert beat._sample_resources()["cpu_percent"] is None
    state["cpu"] = 5.0
    assert beat._sample_resources()["cpu_percent"] == 100.0
    state.update(created=2.0, cpu=0.1)
    assert beat._sample_resources()["cpu_percent"] is None
    state["missing"] = True
    assert beat._sample_resources() == {"rss_mb": None, "cpu_percent": None}
    state["missing"] = False
    assert beat._sample_resources()["cpu_percent"] is None
