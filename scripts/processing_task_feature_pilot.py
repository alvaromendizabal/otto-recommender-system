"""Managed recovery of the bounded pilot using the previously proven CPU image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(project: Path, launch: dict[str, Any]) -> dict[str, Any]:
    from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints
    from otto_recsys.logging_utils import configure_logging, utc_now_iso
    from otto_recsys.research.protocol import atomic_json
    from otto_recsys.research.task_feature_pilot import run
    from otto_recsys.runtime import Heartbeat

    inputs, output = (
        project / "artifacts/task_feature_inputs",
        project / "artifacts/task_feature_pilot",
    )
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("task_feature_pilot", log_dir=output / "logs")
    storage = ResearchCheckpoints(
        output,
        launch["checkpoint_uri"],
        region=launch["region"],
        owner_account=launch["owner_account"],
        logger=logger,
    )
    storage.restore()
    status: dict[str, Any] = {
        "status": "running",
        "stage": "input_recovery",
        "started_at_utc": utc_now_iso(),
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
    }
    atomic_json(output / "status.json", status)
    storage.publish(output / "status.json")
    start = time.perf_counter()

    def download(entry: dict[str, Any]) -> None:
        path = inputs / entry["path"]
        if not path.resolve().is_relative_to(inputs.resolve()):
            raise ValueError("input path escapes workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and digest(path) == entry["sha256"]:
            return
        temporary = path.with_suffix(".download")
        storage.client.download_file(
            launch["bucket"],
            entry["key"],
            str(temporary),
            ExtraArgs={"ExpectedBucketOwner": launch["owner_account"]},
        )
        if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
            raise ValueError("input checksum or length differs")
        temporary.replace(path)
        logger.info("pilot_input_verified", extra={"artifact": entry["path"]})

    try:
        with (
            Heartbeat(logger, stage="download_pilot_inputs", interval_seconds=15),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            list(pool.map(download, launch["inputs"]))
        status.update(stage="smoke_tests")
        atomic_json(output / "status.json", status)
        storage.publish(output / "status.json")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_task_feature_pilot.py",
                "-W",
                "error",
            ],
            cwd=project,
            check=True,
            timeout=90,
        )
        status.update(stage="feature_pilot")
        atomic_json(output / "status.json", status)
        storage.publish(output / "status.json")
        config = json.loads((project / "configs/task_feature_pilot.json").read_text())
        result = run(inputs, output, config, logger=logger, publish=storage.publish)
        lgb: Any = importlib.import_module("lightgbm")
        training = lgb.train

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("complete restart attempted replacement training")

        lgb.train = forbidden
        try:
            replay = run(inputs, output, config, logger=logger, publish=storage.publish)
        finally:
            lgb.train = training
        if replay["arms"] != result["arms"]:
            raise ValueError("training-free replay changed models or scores")
        result["recovery"] = {
            "training_calls": 0,
            "all_models_and_metrics_identical": True,
            "replay_elapsed_seconds": replay["elapsed_seconds"],
        }
        atomic_json(output / "results.json", result)
        storage.publish(output / "results.json")
        pl = importlib.import_module("polars")
        rows = []
        for arm, summary in result["arms"].items():
            path = output / "models" / arm / "statistics.parquet"
            if digest(path) != summary["statistics_sha256"]:
                raise ValueError("metric statistics checksum differs")
            table = pl.read_parquet(path)
            metric = sum(
                weight * table[f"hits_{objective}"].sum() / table[f"denominator_{objective}"].sum()
                for weight, objective in zip(
                    (0.1, 0.3, 0.6), ("clicks", "carts", "orders"), strict=True
                )
            )
            if abs(metric - summary["weighted_recall_at_20"]) > 1e-12:
                raise ValueError("independent pooled metric differs")
            for objective, model in summary["models"].items():
                path = output / "models" / arm / objective / "model.txt"
                native = lgb.Booster(model_file=str(path))
                if (
                    digest(path) != model["model_sha256"]
                    or native.feature_name() != summary["features"][objective]
                ):
                    raise ValueError("native model identity or feature order differs")
            rows.append({"arm": arm, "weighted_recall_at_20": metric, "sessions": table.height})
        atomic_json(
            output / "audit.json",
            {
                "status": "passed",
                "arms": rows,
                "native_rankers_verified": 9,
                "training_free_replay": result["recovery"],
                "source_commit": launch["source_commit"],
                "scope": "Independent metric arithmetic and model hashes; raw features shared.",
            },
        )
        storage.publish(output / "audit.json")
        status.update(status="passed", stage="complete", study_id=result["study_id"])
        print(json.dumps({"pilot_results": rows}), flush=True)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status.update(finished_at_utc=utc_now_iso(), elapsed_seconds=time.perf_counter() - start)
        atomic_json(output / "status.json", status)
        storage.publish(output / "status.json")
        for handler in logger.handlers:
            handler.flush()
        storage.publish(output / "logs/task_feature_pilot.jsonl")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--project", type=Path, default=Path("/opt/ml/processing/work/project"))
    parser.add_argument(
        "--launch", type=Path, default=Path("/opt/ml/processing/input/launch/launch.json")
    )
    args = parser.parse_args()
    launch = json.loads(args.launch.read_text())
    if args.execute:
        print(json.dumps(execute(args.project, launch)), flush=True)
        return 0
    archive = Path("/opt/ml/processing/input/source/source.tar.gz")
    if digest(archive) != launch["source_sha256"]:
        raise ValueError("immutable source archive checksum differs")
    args.project.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not (args.project / member.name).resolve().is_relative_to(
                args.project.resolve()
            ):
                raise ValueError("unsafe source archive member")
        bundle.extractall(args.project, filter="data")
    environment = dict(os.environ)
    environment.update(
        UV_PYTHON_INSTALL_DIR="/opt/ml/processing/work/python",
        UV_CACHE_DIR="/opt/ml/processing/work/uv_cache",
        UV_PROJECT_ENVIRONMENT=str(args.project / ".venv"),
        PIP_ROOT_USER_ACTION="ignore",
        PYTHONUNBUFFERED="1",
        POLARS_MAX_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    environment.pop("VIRTUAL_ENV", None)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv==0.12.10"],
        env=environment,
        check=True,
        timeout=90,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uv",
            "sync",
            "--frozen",
            "--extra",
            "ml",
            "--extra",
            "cloud",
            "--extra",
            "dev",
        ],
        cwd=args.project,
        env=environment,
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            str(args.project / ".venv/bin/python"),
            str(args.project / "scripts/processing_task_feature_pilot.py"),
            "--execute",
            "--launch",
            str(args.launch),
            "--project",
            str(args.project),
        ],
        cwd=args.project,
        env=environment,
        check=True,
        timeout=720,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
