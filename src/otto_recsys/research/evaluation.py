"""Stream every reserved query after checking an immutable model-selection seal."""

from __future__ import annotations

import json
import logging
import multiprocessing
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
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.materialize import initialize_worker, transform_query
from otto_recsys.research.metrics import official_score, one_query, paired_bootstrap
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import verified_file
from otto_recsys.runtime import Heartbeat


def verify_seal(root: Path) -> dict[str, Any]:
    seal = json.loads((root / "evaluation_seal.json").read_text())
    if canonical_json_sha256({k: v for k, v in seal.items() if k != "seal_id"}) != seal["seal_id"]:
        raise ValueError("final evaluation seal was changed")
    if seal["evaluation_labels_consulted"] is not False:
        raise ValueError("model selection must finish before evaluation access")
    if sha256_file(Path(__file__).with_name("features.py")) != seal["feature_code_sha256"]:
        raise ValueError("feature implementation differs from the sealed models")
    if json.loads((root / "corpus/manifest.json").read_text())["input_id"] != seal["corpus_id"]:
        raise ValueError("evaluation corpus differs from the selected experiment")
    if (
        json.loads((root / "retrieval/manifest.json").read_text())["input_id"]
        != seal["retrieval_id"]
    ):
        raise ValueError("historical retrieval differs from the selected experiment")
    for group in ("models", "core_models"):
        if set(seal[group]) != set(OBJECTIVES):
            raise ValueError("seal requires all three task-specific models")
        for model in seal[group].values():
            path = root / model["path"]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or sha256_file(path) != model["sha256"]
            ):
                raise ValueError("sealed model path or checksum is invalid")
    return seal


def run_evaluation(
    root: Path,
    *,
    seed: int,
    workers: int,
    threads: int,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    if not 1 <= workers <= 16 or threads < 1:
        raise ValueError("evaluation requires 1-16 workers and positive prediction threads")
    seal = verify_seal(root)
    output = root / "evaluation"
    output.mkdir(exist_ok=True)
    contract = {
        "seal_id": seal["seal_id"],
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("evaluation.py", "materialize.py", "metrics.py")
        },
        "cohort": "all eligible reserved temporal sessions",
        "negative_sampling": False,
    }
    input_id = canonical_json_sha256(contract)
    with workspace_lock(output):
        contract_path = output / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("evaluation workspace is already tied to a different frozen model")
        atomic_json(contract_path, contract)
        if publish:
            publish(contract_path)
        report_path = output / "report.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if previous["input_id"] == input_id and all(
                (output / p).is_file() and sha256_file(output / p) == h
                for p, h in previous["files"].items()
            ):
                return previous
        # This is the first point at which the evaluation labels are opened.
        queries = Queries(root / "corpus", "evaluation")
        entries = [
            seal[group][objective]
            for group in ("models", "core_models")
            for objective in OBJECTIVES
        ]
        names = tuple(dict.fromkeys(name for model in entries for name in model["features"]))
        models = {m["sha256"]: lgb.Booster(model_file=str(root / m["path"])) for m in entries}
        if any(models[m["sha256"]].feature_name() != m["features"] for m in entries):
            raise ValueError("native model feature order differs from its evaluation seal")
        columns = {m["sha256"]: [names.index(n) for n in m["features"]] for m in entries}
        start = time.perf_counter()
        progress: dict[str, Any] = {"sessions": 0, "total_sessions": queries.session.size}
        parts = []
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(
                    root / "corpus",
                    root / "retrieval",
                    "evaluation",
                    names,
                    400,
                    None,
                    seed,
                ),
            ) as pool,
            Heartbeat(
                logger,
                stage="reserved_evaluation",
                interval_seconds=15,
                progress_provider=progress.copy,
            ),
        ):
            for bucket, begin in enumerate(range(0, queries.session.size, 1024)):
                path = output / f"part-{bucket:04d}.parquet"
                stop = min(begin + 1024, queries.session.size)
                if not verified_file(path, input_id):
                    batch_start = time.perf_counter()
                    rows = list(pool.map(transform_query, range(begin, stop), chunksize=8))
                    x = np.concatenate([r["features"] for r in rows])
                    predictions = {
                        key: model.predict(x[:, columns[key]], num_threads=threads)
                        for key, model in models.items()
                    }
                    statistics = []
                    position = 0
                    for offset, row in enumerate(rows):
                        prefix = queries.prefix(begin + offset)
                        size = row["aid"].size
                        record: dict[str, Any] = {
                            "session": prefix.session,
                            "observed_events": prefix.aid.size,
                            "query_ts": int(prefix.ts[-1]),
                        }
                        for j, objective in enumerate(OBJECTIVES):
                            truth = queries.labels.get(
                                (prefix.session, objective), np.array([], dtype=np.int64)
                            )
                            expected_denominator = queries.denominators[begin + offset, j]
                            record[f"denominator_{objective}"] = int(expected_denominator)
                            for label, score in (
                                (
                                    "selected",
                                    predictions[seal["models"][objective]["sha256"]][
                                        position : position + size
                                    ],
                                ),
                                (
                                    "core",
                                    predictions[seal["core_models"][objective]["sha256"]][
                                        position : position + size
                                    ],
                                ),
                                ("fusion", row["baseline"][:, j]),
                            ):
                                value = one_query(row["aid"], truth, np.asarray(score))
                                if value["denominator"] != expected_denominator:
                                    raise ValueError(
                                        "evaluation truth differs from the complete query ledger"
                                    )
                                for key in ("hits", "ndcg", "mrr", "hit_rate", "eligible"):
                                    record[f"{label}_{objective}_{key}"] = value[key]
                            for budget in (100, 200, 400):
                                record[f"pool_{budget}_{objective}"] = min(
                                    20, int(np.isin(row["aid"][:budget], truth).sum())
                                )
                        statistics.append(record)
                        position += size
                    temporary = path.with_suffix(".parquet.tmp")
                    pl.DataFrame(statistics).write_parquet(temporary, compression="zstd")
                    temporary.replace(path)
                    atomic_json(
                        path.with_suffix(".json"),
                        {
                            "input_id": input_id,
                            "sha256": sha256_file(path),
                            "sessions": stop - begin,
                            "elapsed_seconds": time.perf_counter() - batch_start,
                        },
                    )
                if publish:
                    publish(path)
                    publish(path.with_suffix(".json"))
                parts.append(path.name)
                progress["sessions"] += stop - begin
        frame = pl.concat([pl.read_parquet(output / name) for name in parts]).sort("session")
        if not np.array_equal(frame["session"].to_numpy(), queries.session):
            raise ValueError("evaluation dropped or duplicated reserved sessions")
        denominator = frame.select([f"denominator_{o}" for o in OBJECTIVES]).to_numpy()
        hits = {
            label: frame.select([f"{label}_{o}_hits" for o in OBJECTIVES]).to_numpy()
            for label in ("selected", "core", "fusion")
        }
        scores = {label: official_score(value, denominator) for label, value in hits.items()}
        for label in hits:
            for objective in OBJECTIVES:
                eligible = max(1, int(frame[f"{label}_{objective}_eligible"].sum()))
                scores[label]["objectives"][objective].update(
                    {
                        metric: float(frame[f"{label}_{objective}_{metric}"].sum() / eligible)
                        for metric in ("ndcg", "mrr", "hit_rate")
                    }
                )
        slices = {}
        for label, low, high in (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, 10000)):
            mask = (
                (frame["observed_events"] >= low) & (frame["observed_events"] <= high)
            ).to_numpy()
            if mask.any() and (denominator[mask].sum(axis=0) > 0).all():
                slices[label] = {
                    "sessions": int(mask.sum()),
                    **{
                        method: official_score(value[mask], denominator[mask])
                        for method, value in hits.items()
                    },
                }
        with Heartbeat(logger, stage="paired_uncertainty", interval_seconds=15):
            intervals = {
                label: paired_bootstrap(
                    hits["selected"],
                    hits[label],
                    denominator,
                    seed=seed,
                    replicates=bootstrap_replicates,
                )
                for label in ("core", "fusion")
            }
        report = {
            "status": "passed",
            "input_id": input_id,
            "seal_id": seal["seal_id"],
            "sessions": frame.height,
            "scores": scores,
            "paired_intervals": intervals,
            "prefix_slices": slices,
            "candidate_frontier": {
                str(budget): official_score(
                    frame.select([f"pool_{budget}_{o}" for o in OBJECTIVES]).to_numpy(), denominator
                )
                for budget in (100, 200, 400)
            },
            "elapsed_seconds": time.perf_counter() - start,
            "files": {name: sha256_file(output / name) for name in parts},
            "scope": "newly reserved temporal evaluation; "
            "all models refitted before their permitted cutoffs; "
            "underlying OTTO data had prior exploratory exposure; no leaderboard claim",
        }
        atomic_json(report_path, report)
        if publish:
            publish(report_path)
        return report
