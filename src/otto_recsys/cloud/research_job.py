"""Run one bounded, resumable chronological ranking study on an AWS execution role."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints
from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.evaluation import run_evaluation
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.robustness import verify_seed_launch
from otto_recsys.research.study import run_ablations


def run(launch: dict[str, Any], root: Path) -> dict[str, Any]:
    replication = verify_seed_launch(Path.cwd(), launch) if "robustness" in launch else None
    root.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("managed_research", log_dir=root / "logs")
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
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
        "python": platform.python_version(),
        "lock_sha256": sha256_file(Path("uv.lock")),
        "training": launch["training"],
        "resources": launch["resources"],
        "stage": "restore",
        "status": "running",
    }
    if replication is not None:
        status["robustness"] = replication
        status["evaluation_seed"] = launch["evaluation_seed"]
    try:
        storage.restore()
        logger = configure_logging("managed_research", log_dir=root / "logs")
        storage.logger = logger
        status["stage"] = "ablations"
        atomic_json(root / "job_status.json", status)
        storage.publish(root / "job_status.json")
        ablations = run_ablations(root, launch["training"], logger=logger, publish=storage.publish)
        status.update(stage="reserved_evaluation", study_id=ablations["study_id"])
        atomic_json(root / "job_status.json", status)
        storage.publish(root / "job_status.json")
        evaluation = run_evaluation(
            root,
            seed=int(launch.get("evaluation_seed", launch["training"]["seed"])),
            workers=int(launch["resources"]["feature_workers"]),
            threads=int(launch["training"]["threads"]),
            logger=logger,
            publish=storage.publish,
            bootstrap_replicates=int(launch.get("bootstrap_replicates", 1000)),
        )
        status.update(status="passed", stage="complete", evaluation_id=evaluation["input_id"])
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        logger.exception("managed_research_failed")
        raise
    finally:
        status.update(finished_at=utc_now_iso(), elapsed_seconds=time.perf_counter() - start)
        atomic_json(root / "job_status.json", status)
        storage.publish(root / "job_status.json")
        # Publish a closed-stage log snapshot; active heartbeats never race its digest.
        for handler in logger.handlers:
            handler.flush()
        storage.publish(root / "logs/managed_research.jsonl")
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
