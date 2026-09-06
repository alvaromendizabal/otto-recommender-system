import logging
import time

import pytest

from otto_recsys.runtime import Heartbeat, process_rss_mb


def test_process_rss_is_positive() -> None:
    assert process_rss_mb() > 0.0


def test_heartbeat_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError):
        Heartbeat(
            logging.getLogger("test"),
            stage="unit",
            interval_seconds=0,
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
