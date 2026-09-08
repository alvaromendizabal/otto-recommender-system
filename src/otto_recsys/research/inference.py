"""Refresh observable history and generate complete, resumable OTTO submissions."""

from __future__ import annotations

import gzip
import json
import logging
import multiprocessing
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import OBJECTIVES
from otto_recsys.research.deployment import source_identity
from otto_recsys.research.evaluation import verify_seal
from otto_recsys.research.features import FeatureEngine, Prefix
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import verified_file
from otto_recsys.runtime import Heartbeat


class ObservedQueries:
    """Competition inputs have observed events only; no target-file API exists."""

    def __init__(self, directory: Path) -> None:
        frame = pl.read_parquet(directory / "part-*.parquet").sort("session", "event_index")
        if frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
            raise ValueError("observed test events contain nulls")
        self.session, self.starts, self.counts = np.unique(
            frame["session"].to_numpy(), return_index=True, return_counts=True
        )
        if not self.session.size:
            raise ValueError("observed test queries are empty")
        self.aid = frame["aid"].to_numpy().astype(np.int64)
        self.ts = frame["ts"].to_numpy().astype(np.int64)
        self.kind = frame["event_type"].to_numpy().astype(np.int64)
        expected = np.arange(frame.height) - np.repeat(self.starts, self.counts)
        if not np.array_equal(frame["event_index"].to_numpy(), expected):
            raise ValueError("test session event indices are not complete contiguous sequences")
        boundary = np.zeros(frame.height, dtype=bool)
        boundary[self.starts] = True
        if (
            (self.aid < 0).any()
            or (self.session < 0).any()
            or not np.isin(self.kind, [0, 1, 2]).all()
            or ((np.diff(self.ts) < 0) & ~boundary[1:]).any()
        ):
            raise ValueError("observed test identifiers, types or timestamps are invalid")

    def prefix(self, index: int) -> Prefix:
        start = self.starts[index]
        stop = start + self.counts[index]
        return Prefix(
            int(self.session[index]),
            self.aid[start:stop],
            self.ts[start:stop],
            self.kind[start:stop],
        )


_ENGINE: FeatureEngine | None = None
_QUERIES: ObservedQueries | None = None
_NAMES: tuple[str, ...] = ()


def initialize_prediction(test: Path, retrieval: Path, names: tuple[str, ...]) -> None:
    global _ENGINE, _QUERIES, _NAMES
    _ENGINE, _QUERIES, _NAMES = FeatureEngine(retrieval), ObservedQueries(test), names
    if int(_QUERIES.ts.min()) < _ENGINE.cutoff:
        raise ValueError("refreshed history reaches beyond observed test availability")


def prediction_features(index: int) -> dict[str, Any]:
    if _ENGINE is None or _QUERIES is None:
        raise RuntimeError("prediction worker is not initialized")
    start = time.perf_counter()
    prefix = _QUERIES.prefix(index)
    candidates = _ENGINE.candidates(prefix, 400)
    matrix = _ENGINE.transform(prefix, candidates, _NAMES)
    return {
        "session": prefix.session,
        "aid": candidates.aid,
        "x": matrix,
        "feature_seconds": time.perf_counter() - start,
    }


def validate_submission(path: Path, sessions: np.ndarray) -> dict[str, Any]:
    """Validate every task row and ordered list against the exact input ledger."""
    frame = pl.read_csv(path)
    if frame.columns != ["session_type", "labels"] or frame.height != 3 * sessions.size:
        raise ValueError("submission requires the two official columns and three rows per session")
    if frame["session_type"].n_unique() != frame.height:
        raise ValueError("submission contains duplicated session/objective rows")
    expected = {f"{session}_{objective}" for session in sessions for objective in OBJECTIVES}
    if set(frame["session_type"].to_list()) != expected:
        raise ValueError("submission does not cover the exact observed session/objective ledger")
    for text in frame["labels"]:
        values = str(text).split()
        if len(values) != 20 or len(set(values)) != 20 or not all(v.isdecimal() for v in values):
            raise ValueError("submission needs 20 unique nonnegative integer item IDs per row")
    return {
        "status": "passed",
        "sessions": sessions.size,
        "rows": frame.height,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def run_prediction(
    model_root: Path,
    test: Path,
    retrieval: Path,
    output: Path,
    *,
    workers: int,
    threads: int,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    if not 1 <= workers <= 16 or threads < 1:
        raise ValueError("prediction requires 1-16 workers and positive model threads")
    seal = verify_seal(model_root)
    evaluated = json.loads((model_root / "evaluation/report.json").read_text())
    if evaluated["status"] != "passed" or evaluated["seal_id"] != seal["seal_id"]:
        raise ValueError(
            "competition prediction requires a completed evaluation of the sealed weights"
        )
    fitted = json.loads((retrieval / "manifest.json").read_text())
    contract = {
        "seal_id": seal["seal_id"],
        "test_files": source_identity(test),
        "retrieval_id": fitted["input_id"],
        "candidate_budget": 400,
        "code_sha256": sha256_file(Path(__file__)),
        "weights_refitted": False,
    }
    input_id = canonical_json_sha256(contract)
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        contract_path = output / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("submission workspace belongs to a different model or test input")
        atomic_json(contract_path, contract)
        if publish:
            publish(contract_path)
        queries = ObservedQueries(test)
        entries = [seal["models"][o] for o in OBJECTIVES]
        names = tuple(dict.fromkeys(n for m in entries for n in m["features"]))
        models = [lgb.Booster(model_file=str(model_root / m["path"])) for m in entries]
        if any(
            model.feature_name() != entry["features"]
            for model, entry in zip(models, entries, strict=True)
        ):
            raise ValueError("submission native model schema differs from the sealed weights")
        columns = [[names.index(n) for n in m["features"]] for m in entries]
        start = time.perf_counter()
        progress = {"sessions": 0, "total_sessions": queries.session.size}
        parts = []
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_prediction,
                initargs=(test, retrieval, names),
            ) as pool,
            Heartbeat(
                logger,
                stage="competition_prediction",
                interval_seconds=15,
                progress_provider=progress.copy,
            ),
        ):
            for bucket, begin in enumerate(range(0, queries.session.size, 1024)):
                stop = min(begin + 1024, queries.session.size)
                path = output / f"part-{bucket:04d}.csv"
                if not verified_file(path, input_id):
                    batch_start = time.perf_counter()
                    rows = list(pool.map(prediction_features, range(begin, stop), chunksize=8))
                    x = np.concatenate([row["x"] for row in rows])
                    prediction = [
                        np.asarray(m.predict(x[:, c], num_threads=threads))
                        for m, c in zip(models, columns, strict=True)
                    ]
                    records = []
                    offset = 0
                    for row in rows:
                        size = row["aid"].size
                        if size < 20:
                            raise ValueError(
                                "candidate catalog cannot fill 20 unique recommendations"
                            )
                        for j, objective in enumerate(OBJECTIVES):
                            scores = prediction[j][offset : offset + size]
                            if not np.isfinite(scores).all():
                                raise ValueError(
                                    "native ranker returned nonfinite prediction scores"
                                )
                            order = np.lexsort((row["aid"], -scores))[:20]
                            records.append(
                                (
                                    f"{row['session']}_{objective}",
                                    " ".join(map(str, row["aid"][order])),
                                )
                            )
                        offset += size
                    temporary = path.with_suffix(".csv.tmp")
                    pl.DataFrame(
                        records, schema=["session_type", "labels"], orient="row"
                    ).write_csv(temporary)
                    temporary.replace(path)
                    atomic_json(
                        path.with_suffix(".json"),
                        {
                            "input_id": input_id,
                            "sha256": sha256_file(path),
                            "sessions": len(rows),
                            "elapsed_seconds": time.perf_counter() - batch_start,
                            "worker_feature_seconds": sum(row["feature_seconds"] for row in rows),
                        },
                    )
                if publish:
                    publish(path)
                    publish(path.with_suffix(".json"))
                parts.append(path.name)
                progress["sessions"] += stop - begin
        destination = output / "submission.csv.gz"
        temporary = output / "submission.csv.gz.tmp"
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        ):
            compressed.write(b"session_type,labels\n")
            for name in parts:
                with (output / name).open("rb") as source:
                    if source.readline().strip() != b"session_type,labels":
                        raise ValueError("prediction part has a noncanonical CSV header")
                    shutil.copyfileobj(source, compressed)
        temporary.replace(destination)
        with Heartbeat(logger, stage="submission_validation", interval_seconds=15):
            report = validate_submission(destination, queries.session)
        report.update(
            input_id=input_id,
            seal_id=seal["seal_id"],
            files={n: sha256_file(output / n) for n in parts},
            elapsed_seconds=time.perf_counter() - start,
            feature_order=list(names),
            scope="Local format and coverage validation only; "
            "no Kaggle acceptance or hidden-test score claimed",
        )
        atomic_json(output / "manifest.json", report)
        if publish:
            publish(destination)
            publish(output / "manifest.json")
        return report


def export_replay(
    model_root: Path,
    test: Path,
    retrieval: Path,
    prediction: Path,
    output: Path,
    *,
    sessions: int = 8,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Export a compact real-data native-model replay, checked against the full run."""
    seal = verify_seal(model_root)
    complete = json.loads((prediction / "manifest.json").read_text())
    if complete["status"] != "passed" or complete["seal_id"] != seal["seal_id"]:
        raise ValueError("replay export requires completed prediction with the frozen weights")
    if not 1 <= sessions <= 1024:
        raise ValueError("compact replay requires 1-1024 initial test sessions")
    names = tuple(dict.fromkeys(n for o in OBJECTIVES for n in seal["models"][o]["features"]))
    initialize_prediction(test, retrieval, names)
    if _QUERIES is None:
        raise RuntimeError("replay queries are unavailable")
    count = min(sessions, _QUERIES.session.size)
    records = [prediction_features(i) for i in range(count)]
    frames = [
        pl.DataFrame(r["x"], schema=list(names)).with_columns(
            pl.lit(r["session"]).alias("session"),
            pl.Series("aid", r["aid"]),
            pl.Series("candidate_position", np.arange(r["aid"].size)),
        )
        for r in records
    ]
    output.mkdir(parents=True, exist_ok=True)
    features = output / "features.parquet"
    pl.concat(frames).write_parquet(features, compression="zstd")
    source = prediction / "part-0000.csv"
    if sha256_file(source) != complete["files"][source.name]:
        raise ValueError("full prediction part failed replay provenance validation")
    expected_keys = [f"{r['session']}_{o}" for r in records for o in OBJECTIVES]
    expected = pl.read_csv(source).filter(pl.col("session_type").is_in(expected_keys))
    if expected.height != 3 * count:
        raise ValueError("full prediction does not contain the requested replay sessions")
    expected.write_csv(output / "expected.csv")
    models = {}
    for objective in OBJECTIVES:
        entry = seal["models"][objective]
        path = output / f"{objective}.txt"
        shutil.copyfile(model_root / entry["path"], path)
        models[objective] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "features": entry["features"],
            "variant": entry["variant"],
        }
    files = [features, output / "expected.csv", *[output / m["path"] for m in models.values()]]
    report = {
        "status": "passed",
        "seal_id": seal["seal_id"],
        "sessions": count,
        "candidate_rows": sum(r["aid"].size for r in records),
        "models": models,
        "history_end": json.loads((retrieval / "manifest.json").read_text())["history_end"],
        "files": {p.name: sha256_file(p) for p in files},
        "full_prediction": {
            k: complete[k] for k in ["input_id", "sessions", "rows", "sha256", "elapsed_seconds"]
        },
        "scope": "First observed competition sessions; candidate features and frozen native models "
        "for exact replay against the full validated prediction. No test labels.",
    }
    atomic_json(output / "manifest.json", report)
    if publish:
        for path in files:
            publish(path)
        publish(output / "manifest.json")
    return report
