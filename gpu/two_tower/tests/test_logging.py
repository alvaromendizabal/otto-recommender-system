from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from otto_two_tower.logging_utils import configure_logging
from otto_two_tower.telemetry import TrainingHeartbeat


def test_heartbeat_persists_timestamp_elapsed_and_progress(tmp_path: Path) -> None:
    logger = configure_logging("heartbeat_contract", tmp_path)
    observed = threading.Event()

    def progress() -> dict[str, int]:
        observed.set()
        return {"bucket": 3, "examples": 128}

    with TrainingHeartbeat(
        logger, stage="retrieval", interval_seconds=0.01, progress_provider=progress
    ):
        assert observed.wait(timeout=10)
    rows = [
        json.loads(line)
        for line in (tmp_path / "heartbeat_contract.jsonl").read_text().splitlines()
    ]
    assert rows
    assert datetime.fromisoformat(rows[0]["timestamp"]).tzinfo is not None
    assert rows[0]["elapsed_seconds"] >= 0
    assert rows[0]["bucket"] == 3
    assert rows[0]["examples"] == 128
