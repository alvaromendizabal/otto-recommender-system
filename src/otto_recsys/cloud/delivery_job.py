"""Explain frozen models and run full competition inference with durable parts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.competition_input import require_competition_input
from otto_recsys.research.deployment import prepare_history
from otto_recsys.research.inference import export_replay
from otto_recsys.research.interpretation import explain
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import build_retrievers


def refresh_history(launch: dict[str, Any], root: Path, storage: ResearchCheckpoints) -> None:
    """Compute deployment aggregates inside the account from existing S3 inputs."""
    require_competition_input(root.parent / "test")
    config = tomllib.loads(Path("configs/research.toml").read_text())
    deployment = root.parent / "inference"
    resources = launch["resources"]
    prepare_history(
        root.parent / "train",
        root.parent / "test",
        deployment / "corpus",
        logger=storage.logger,
        threads=resources["history_threads"],
        memory_gib=resources["history_memory_gib"],
    )
    build_retrievers(
        deployment / "corpus",
        deployment / "retrieval",
        config["retrieval"],
        logger=storage.logger,
        threads=resources["history_threads"],
        memory_gib=resources["history_memory_gib"],
    )
    for directory in ("corpus", "retrieval"):
        for path in sorted((deployment / directory).rglob("*")):
            if path.is_file() and path.suffix in {".parquet", ".json"}:
                storage.publish(path)


def notebook_prediction(launch: dict[str, Any], root: Path) -> dict[str, Any]:
    project = Path.cwd()
    environment = dict(os.environ)
    environment.update(
        OTTO_PREDICTION_CHECKPOINT_URI=launch["prediction_checkpoint_uri"],
        OTTO_OWNER_ACCOUNT=launch["owner_account"],
        OTTO_AWS_REGION=launch["region"],
        OTTO_PREDICTION_THREADS=str(launch["training"]["threads"]),
        OTTO_PREDICTION_WORKERS=str(launch["resources"]["feature_workers"]),
    )
    subprocess.run(
        [str(project.parent / "analysis/bin/python"), "scripts/execute_inference_notebook.py"],
        cwd=project,
        env=environment,
        check=True,
    )
    return dict(json.loads((root.parent / "inference/prediction/manifest.json").read_text()))


def run(launch: dict[str, Any], root: Path) -> dict[str, Any]:
    logger = configure_logging("research_delivery", log_dir=root / "logs")

    def storage(path: Path, uri: str) -> ResearchCheckpoints:
        return ResearchCheckpoints(
            path, uri, region=launch["region"], owner_account=launch["owner_account"], logger=logger
        )

    original = storage(root, launch["study_checkpoint_uri"])
    research = storage(root, launch["checkpoint_uri"])
    deployment = root.parent / "inference"
    predictions = storage(deployment, launch["prediction_checkpoint_uri"])
    start = time.perf_counter()
    status: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
        "status": "running",
        "stage": "restore",
        "resources": launch["resources"],
    }
    try:
        original.restore()
        research.restore()
        predictions.restore()
        logger = configure_logging("research_delivery", log_dir=root / "logs")
        for store in (original, research, predictions):
            store.logger = logger
        status["stage"] = "interpretation"
        atomic_json(root / "delivery_status.json", status)
        research.publish(root / "delivery_status.json")
        interpretation = explain(
            root,
            seed=launch["training"]["seed"],
            threads=launch["training"]["threads"],
            logger=logger,
            publish=research.publish,
        )
        status.update(stage="deployment_history", interpretation_id=interpretation["input_id"])
        atomic_json(root / "delivery_status.json", status)
        research.publish(root / "delivery_status.json")
        refresh_history(launch, root, predictions)
        status["stage"] = "competition_prediction"
        atomic_json(root / "delivery_status.json", status)
        research.publish(root / "delivery_status.json")
        prediction = notebook_prediction(launch, root)
        for path in sorted((deployment / "notebooks").glob("*")):
            if path.suffix in {".ipynb", ".json"}:
                predictions.publish(path)
        export_replay(
            root,
            root.parent / "test",
            deployment / "retrieval",
            deployment / "prediction",
            root / "inference_replay",
            publish=research.publish,
        )
        status.update(
            stage="complete",
            status="passed",
            prediction_id=prediction["input_id"],
            prediction_sha256=prediction["sha256"],
        )
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        logger.exception("research_delivery_failed")
        raise
    finally:
        status.update(finished_at=utc_now_iso(), elapsed_seconds=time.perf_counter() - start)
        atomic_json(root / "delivery_status.json", status)
        research.publish(root / "delivery_status.json")
        for handler in logger.handlers:
            handler.flush()
        research.publish(root / "logs/research_delivery.jsonl")
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
