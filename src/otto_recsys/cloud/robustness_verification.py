"""Verify a completed replication using durable inputs and a separate audit receipt."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.robustness import verify_verification_launch
from otto_recsys.research.robustness_audit import verify_replication
from otto_recsys.runtime import Heartbeat


def run(launch: dict[str, Any], root: Path) -> dict[str, Any]:
    replication = verify_verification_launch(Path.cwd(), launch)
    logger = configure_logging("managed_robustness_verification", log_dir=root / "logs")
    storage = ResearchCheckpoints(
        root,
        launch["checkpoint_uri"],
        region=launch["region"],
        owner_account=launch["owner_account"],
        logger=logger,
    )
    start = time.perf_counter()
    status: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "status": "running",
        "stage": "restore",
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
        "training_source_commit": launch["training_launch"]["source_commit"],
        "robustness": replication,
    }
    destination = root / "robustness_audit/job_status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Heartbeat(logger, stage="verification_restore", interval_seconds=15):
            storage.restore()
        logger = configure_logging("managed_robustness_verification", log_dir=root / "logs")
        storage.logger = logger
        status["stage"] = "verification"
        atomic_json(destination, status)
        storage.publish(destination)
        report = verify_replication(Path.cwd(), root, launch["training_launch"], logger=logger)
        storage.publish(root / "robustness_audit/report.json")
        status.update(status="passed", stage="complete", audit_id=report["audit_id"])
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        logger.exception("managed_robustness_verification_failed")
        raise
    finally:
        status.update(finished_at=utc_now_iso(), elapsed_seconds=time.perf_counter() - start)
        atomic_json(destination, status)
        storage.publish(destination)
        for handler in logger.handlers:
            handler.flush()
        storage.publish(root / "logs/managed_robustness_verification.jsonl")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    parser.add_argument("--root", type=Path, default=Path("artifacts/research"))
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.launch.read_text()), args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
