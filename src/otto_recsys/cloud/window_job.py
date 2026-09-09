"""Prepare one temporal window, fit one seed, or independently audit that seed."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

from otto_recsys.cloud.research_checkpoints import ARTIFACT_SUFFIXES, ResearchCheckpoints
from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.dataset import build_cache
from otto_recsys.research.evaluation import run_evaluation
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.materialize import model_cache
from otto_recsys.research.protocol import (
    TemporalProtocol,
    atomic_json,
    build_temporal_corpus,
    source_inventory,
)
from otto_recsys.research.retrievers import build_retrievers
from otto_recsys.research.robustness import fingerprint
from otto_recsys.research.screening import screen_cache
from otto_recsys.research.study import run_ablations
from otto_recsys.research.temporal_robustness import (
    INPUT_REPORTS,
    original_source,
    protocol_values,
    verify_window_launch,
    verify_window_verification,
)
from otto_recsys.runtime import Heartbeat


def publish_directory(storage: ResearchCheckpoints, directory: Path) -> None:
    """Flush a closed stage; exclude temporary arrays, locks and DuckDB spill files."""
    paths = [
        p for p in directory.rglob("*") if p.is_file()
        and p.suffix in ARTIFACT_SUFFIXES and "working" not in p.parts
    ]
    for path in sorted(paths, key=lambda p: (p.suffix == ".json", p.as_posix())):
        storage.publish(path)


class CompletedParts(logging.Handler):
    """Synchronously persist immutable parts after the producer closes their receipts."""

    def __init__(self, root: Path, storage: ResearchCheckpoints) -> None:
        super().__init__()
        self.root, self.storage = root, storage

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message == "research_graph_part_complete":
            paths = [
                self.root / "retrieval" / p
                for p in ("history_statistics.parquet", "history_tail.parquet")
            ]
            bucket = record.__dict__["bucket"]
            paths.append(self.root / f"retrieval/parts/part-{bucket:03d}.parquet")
        elif message == "research_feature_part_complete":
            bucket = record.__dict__["bucket"]
            paths = [self.root / f"screening_cache/part-{bucket:04d}.parquet"]
        else:
            return
        for path in paths:
            self.storage.publish(path)
            self.storage.publish(path.with_suffix(".json"))


def prepare_window(
    repository: Path,
    root: Path,
    source: Path,
    launch: dict[str, Any],
    *,
    logger: logging.Logger,
    storage: ResearchCheckpoints,
) -> dict[str, Any]:
    from otto_recsys.research.window_audit import verify_window_inputs

    if source_inventory(source) != original_source(repository):
        raise ValueError("earlier window requires the exact original event source")
    if (root / "preparation.json").exists():
        verify_window_inputs(repository, root, launch)
        return dict(json.loads((root / "preparation.json").read_text()))
    protocol = TemporalProtocol(**protocol_values(launch))
    seed = protocol.seed
    threads = int(launch["training"]["threads"])
    memory = int(launch["resources"]["memory_gib"])
    handler = CompletedParts(root, storage)
    logger.addHandler(handler)

    def completed(name: str) -> None:
        publish_directory(storage, root / name)
        logger.info("window_stage_complete", extra={"stage": name})

    try:
        build_temporal_corpus(
            source, root / "corpus", protocol, logger=logger,
            threads=threads, memory_gib=memory,
        )
        completed("corpus")
        build_retrievers(
            root / "corpus", root / "retrieval", launch["retrieval"],
            logger=logger, threads=threads, memory_gib=memory,
        )
        completed("retrieval")
        build_cache(
            root / "corpus", root / "retrieval", root / "screening_cache", role="fit",
            names=tuple(f.name for f in feature_catalog()), candidate_budget=400,
            negative_budget=launch["training"]["negative_budget"], seed=seed,
            max_rows=launch["features"]["screening_rows"], logger=logger,
        )
        completed("screening_cache")
        selected = screen_cache(
            root / "screening_cache", root / "screening",
            max_retained=launch["features"]["max_retained"], seed=seed,
            logger=logger, threads=threads,
        )
        completed("screening")
        for role in ("fit", "selection"):
            model_cache(
                root / "corpus", root / "retrieval", root / f"{role}_cache", role=role,
                names=tuple(selected["retained"]), budget=400,
                negatives=launch["training"]["negative_budget"], seed=seed,
                workers=launch["resources"]["feature_workers"], logger=logger,
                publish=storage.publish,
            )
            completed(f"{role}_cache")
    finally:
        logger.removeHandler(handler)
        handler.close()
    result = {
        "status": "passed", "protocol_id": launch["robustness"]["protocol_id"],
        "window": launch["robustness"]["window"], "cohort_seed": seed,
        "source_commit": launch["source_commit"], "prepared_at": utc_now_iso(),
        "files": {p: sha256_file(root / p) for p in INPUT_REPORTS if p != "preparation.json"},
    }
    result["input_id"] = fingerprint(result)
    atomic_json(root / "preparation.json", result)
    verify_window_inputs(repository, root, launch)
    storage.publish(root / "preparation.json")
    return result


def run(launch: dict[str, Any], root: Path, source: Path) -> dict[str, Any]:
    repository = Path.cwd()
    auditing = launch["task"] == "window_verification"
    cell = (verify_window_verification if auditing else verify_window_launch)(repository, launch)
    root.mkdir(parents=True, exist_ok=True)
    log_name = "managed_window_verification" if auditing else "managed_window"
    logger = configure_logging(log_name, log_dir=root / "logs")
    stores = [
        ResearchCheckpoints(
            root, launch[key], region=launch["region"],
            owner_account=launch["owner_account"], logger=logger,
        )
        for key in ("input_checkpoint_uri", "checkpoint_uri")
    ]
    inputs, storage = stores
    destination = root / ("robustness_audit/job_status.json" if auditing else "job_status.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status: dict[str, Any] = {
        "started_at": utc_now_iso(), "status": "running", "stage": "restore",
        "source_commit": launch["source_commit"], "source_sha256": launch["source_sha256"],
        "python": platform.python_version(), "lock_sha256": sha256_file(repository / "uv.lock"),
        "robustness": cell, "training": launch["training"], "resources": launch["resources"],
        "evaluation_seed": launch["evaluation_seed"],
    }

    def stage(name: str) -> None:
        status["stage"] = name
        atomic_json(destination, status)
        storage.publish(destination)
        logger.info("window_stage_started", extra={"stage": name, "cell_id": cell["cell_id"]})

    try:
        with Heartbeat(logger, stage="window_restore", interval_seconds=15):
            for store in stores:
                store.restore()
        logger = configure_logging(log_name, log_dir=root / "logs")
        for store in stores:
            store.logger = logger
        if auditing:
            from otto_recsys.research.window_audit import verify_window

            status["training_source_commit"] = launch["training_launch"]["source_commit"]
            stage("verification")
            audit = verify_window(
                repository, root, source, launch["training_launch"], logger=logger
            )
            storage.publish(root / "robustness_audit/report.json")
            status["audit_id"] = audit["audit_id"]
        else:
            stage("preparation")
            prepared = prepare_window(
                repository, root, source, launch, logger=logger, storage=inputs
            )
            status["preparation_id"] = prepared["input_id"]
            stage("ablations")
            ablations = run_ablations(
                root, launch["training"], logger=logger, publish=storage.publish
            )
            status["study_id"] = ablations["study_id"]
            stage("reserved_evaluation")
            evaluation = run_evaluation(
                root, seed=launch["evaluation_seed"],
                workers=launch["resources"]["feature_workers"],
                threads=launch["training"]["threads"], logger=logger,
                publish=storage.publish, bootstrap_replicates=launch["bootstrap_replicates"],
            )
            status["evaluation_id"] = evaluation["input_id"]
        status.update(status="passed", stage="complete")
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        logger.exception("managed_window_failed")
        raise
    finally:
        status.update(finished_at=utc_now_iso(), elapsed_seconds=time.perf_counter() - start)
        atomic_json(destination, status)
        storage.publish(destination)
        for handler in logger.handlers:
            handler.flush()
        storage.publish(root / "logs" / f"{log_name}.jsonl")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    parser.add_argument("--root", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--source", type=Path, default=Path("artifacts/train"))
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.launch.read_text()), args.root, args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
