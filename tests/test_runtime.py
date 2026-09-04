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
