"""Managed entry point for a bounded graph feature ablation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.graph_feature_study import run_study
from otto_recsys.research.protocol import atomic_json


def run(launch: dict[str, Any], inputs: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("graph_feature_study", log_dir=output / "logs")
    storage = ResearchCheckpoints(
        output,
        launch["checkpoint_uri"],
        region=launch["region"],
        owner_account=launch["owner_account"],
        logger=logger,
    )
    started = time.perf_counter()
    status = {
        "status": "running",
        "stage": "restore",
        "started_at": utc_now_iso(),
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
        "resources": launch["resources"],
    }
    try:
        storage.restore()
        logger = configure_logging("graph_feature_study", log_dir=output / "logs")
        storage.logger = logger
        status["stage"] = "graph_feature_ablation"
        atomic_json(output / "job_status.json", status)
        storage.publish(output / "job_status.json")
        result = run_study(inputs, output, launch["study"], logger=logger, publish=storage.publish)
        status.update(status="passed", stage="complete", study_id=result["study_id"])
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        logger.exception("graph_feature_study_failed")
        raise
    finally:
        status.update(finished_at=utc_now_iso(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "job_status.json", status)
        storage.publish(output / "job_status.json")
        for handler in logger.handlers:
            handler.flush()
        storage.publish(output / "logs/graph_feature_study.jsonl")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                json.loads(args.launch.read_text()),
                Path("artifacts/research"),
                Path("artifacts/graph_feature_study"),
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
