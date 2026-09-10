"""Execute the preregistered CPU pilot, with a hard deadline and recoverable models."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.task_feature_pilot import run


def deadline(_signum: int, _frame: object) -> None:
    raise TimeoutError("bounded pilot reached its preregistered wall-clock limit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/task_feature_pilot.json"))
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/task_feature_inputs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/task_feature_pilot"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("task_feature_pilot", log_dir=args.output / "logs")
    started = time.perf_counter()
    status: dict[str, Any] = {"status": "running", "started_at_utc": utc_now_iso()}
    atomic_json(args.output / "status.json", status)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config["maximum_seconds"])
    try:
        result = run(args.inputs, args.output, config, logger=logger)
        status.update(status="passed", study_id=result["study_id"])
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        status.update(finished_at_utc=utc_now_iso(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(args.output / "status.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
