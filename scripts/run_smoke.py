from __future__ import annotations

import tempfile
import time
from pathlib import Path

from otto_recsys.evaluation.metrics import weighted_recall_at_k
from otto_recsys.logging_utils import configure_logging
from otto_recsys.runtime import Heartbeat


def main() -> int:
    logger = configure_logging("smoke")
    started = time.perf_counter()

    with Heartbeat(logger, stage="smoke", interval_seconds=1.0):
        predictions = {
            "clicks": {1: [10, 11], 2: [20]},
            "carts": {1: [11], 2: [21]},
            "orders": {1: [12], 2: [22]},
        }
        targets = {
            "clicks": {1: [10], 2: [20]},
            "carts": {1: [11], 2: [21]},
            "orders": {1: [12], 2: [99]},
        }

        score, detail = weighted_recall_at_k(
            predictions,
            targets,
            k=20,
        )

        assert detail == {
            "clicks": 1.0,
            "carts": 1.0,
            "orders": 0.5,
        }
        assert abs(score - 0.7) < 1e-12

        with tempfile.TemporaryDirectory(prefix="otto-smoke-") as temp_dir:
            marker = Path(temp_dir) / "smoke.ok"
            marker.write_text("ok\n", encoding="utf-8")
            assert marker.read_text(encoding="utf-8") == "ok\n"

    elapsed = round(time.perf_counter() - started, 3)
    logger.info(
        "smoke_complete",
        extra={
            "event": "smoke_complete",
            "status": "passed",
            "elapsed_seconds": elapsed,
            "weighted_recall_20": score,
        },
    )
    print(f"OTTO_SMOKE_PASSED elapsed_seconds={elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
