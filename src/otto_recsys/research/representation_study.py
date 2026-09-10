"""Matched learned-feature ablations; all development decisions use fit/selection only."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.materialize import MODEL_METADATA
from otto_recsys.research.metrics import official_score, paired_bootstrap, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.representations import (
    FAMILIES,
    Representation,
    feature_names,
    prepare_sequences,
    train_representation,
)
from otto_recsys.research.retrievers import verified_file
from otto_recsys.research.study import load_model_cache
from otto_recsys.research.training import fit_model
from otto_recsys.runtime import Heartbeat


def augment_cache(
    source: Path,
    output: Path,
    queries: Queries,
    representations: dict[str, Representation],
    *,
    vector_hashes: dict[str, str],
    base_names: tuple[str, ...],
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    original = json.loads((source / "manifest.json").read_text())
    if original["status"] != "passed" or original["role"] != queries.role:
        raise ValueError("representation input cache has an incorrect temporal role")
    if queries.role not in ("fit", "selection"):
        raise ValueError("representation development cannot accept evaluation queries")
    names = (*base_names, *(name for family in FAMILIES for name in feature_names(family)))
    contract = {
        "source_cache_id": original["input_id"],
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "corpus_id": queries.manifest["input_id"],
        "role": queries.role,
        "representations": vector_hashes,
        "features": list(names),
        "code_sha256": sha256_file(Path(__file__)),
        "candidate_policy": "exact existing rows and order; never inject labels or candidates",
    }
    identity = canonical_json_sha256(contract)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("augmented cache contract differs")
    atomic_json(contract_path, contract)
    if publish:
        publish(contract_path)
    indices = {int(session): i for i, session in enumerate(queries.session)}
    count = 0
    files = {}
    with Heartbeat(logger, stage=f"embedding_features_{queries.role}", interval_seconds=15):
        for name in original["parts"]:
            source_path, destination = source / name, output / name
            if sha256_file(source_path) != original["files"][name]:
                raise ValueError("source candidate feature cache checksum mismatch")
            if not verified_file(destination, identity):
                frame = pl.read_parquet(source_path, columns=[*base_names, *MODEL_METADATA])
                sessions = frame["session"].to_numpy()
                aids = frame["aid"].to_numpy()
                ids, starts, groups = np.unique(sessions, return_index=True, return_counts=True)
                if (np.diff(sessions) < 0).any() or not set(ids).issubset(indices):
                    raise ValueError("cache groups do not belong to the declared prefix ledger")
                matrix = np.empty((frame.height, len(names) - len(base_names)), dtype=np.float32)
                for session, start, size in zip(ids, starts, groups, strict=True):
                    prefix = queries.prefix(indices[int(session)])
                    matrix[start : start + size] = np.column_stack(
                        [
                            representations[family].transform(prefix, aids[start : start + size])
                            for family in FAMILIES
                        ]
                    )
                frame = frame.hstack(pl.DataFrame(matrix, schema=list(names[len(base_names) :])))
                temporary = destination.with_suffix(".parquet.tmp")
                frame.write_parquet(temporary, compression="zstd")
                temporary.replace(destination)
                atomic_json(
                    destination.with_suffix(".json"),
                    {
                        "input_id": identity,
                        "sha256": sha256_file(destination),
                        "rows": frame.height,
                    },
                )
            receipt = json.loads(destination.with_suffix(".json").read_text())
            count += int(receipt["rows"])
            files[name] = receipt["sha256"]
            if publish:
                publish(destination)
                publish(destination.with_suffix(".json"))
            logger.info(
                "embedding_feature_part_complete",
                extra={
                    "role": queries.role,
                    "part": name,
                    "completed_rows": count,
                },
            )
    if count != original["rows"]:
        raise ValueError("augmentation must preserve every candidate row")
    result = {
        "status": "passed",
        "input_id": identity,
        "role": queries.role,
        "features": list(names),
        "rows": count,
        "sessions": original["sessions"],
        "parts": original["parts"],
        "files": files,
    }
    atomic_json(output / "manifest.json", result)
    if publish:
        publish(output / "manifest.json")
    return result


def screen_features(
    fit_x: np.ndarray, names: tuple[str, ...], base_names: tuple[str, ...], *, rows: int
) -> dict[str, Any]:
    """Unsupervised quality/redundancy screening, with no selection data or labels accepted."""
    if rows < 2 or fit_x.shape[0] < 2 or fit_x.shape[1] != len(names):
        raise ValueError("screening requires aligned fitting rows")
    indices = np.linspace(0, fit_x.shape[0] - 1, min(rows, fit_x.shape[0]), dtype=np.int64)
    sample = np.asarray(fit_x[indices], dtype=np.float64)
    finite = np.isfinite(sample).all(axis=0)
    sample[:, ~finite] = 0
    sample -= sample.mean(axis=0)
    norm = np.linalg.norm(sample, axis=0)
    sample /= np.maximum(norm, 1e-12)
    correlation = sample.T @ sample
    kept = [names.index(name) for name in base_names]
    report = []
    for j, name in enumerate(names):
        if name in base_names:
            continue
        reason = "retained"
        redundant_with = None
        if not finite[j]:
            reason = "nonfinite"
        elif norm[j] < 1e-10:
            reason = "constant_on_fit_sample"
        elif kept and np.abs(correlation[j, kept]).max() >= 0.99999:
            reason = "redundant_on_fit_sample"
            redundant_with = names[kept[int(np.argmax(np.abs(correlation[j, kept])))]]
        else:
            kept.append(j)
        report.append({"name": name, "decision": reason, "redundant_with": redundant_with})
    return {
        "scope": "fitting rows only; original baseline features fixed; no labels used",
        "rows": int(indices.size),
        "absolute_correlation_threshold": 0.99999,
        "features": [names[j] for j in kept],
        "decisions": report,
    }


def run_study(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Feature-only hypothesis test; no automatic evaluation access or Kaggle promotion."""
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        return _run_study(inputs, output, config, logger=logger, publish=publish)


def _run_study(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = inputs / "corpus"
    corpus_manifest = json.loads((corpus / "manifest.json").read_text())
    protocol = json.loads((corpus / "contract.json").read_text())["protocol"]
    base_names = tuple(config["baseline_features"])
    contract = {
        "config": config,
        "corpus_id": corpus_manifest["input_id"],
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "representations.py",
                "representation_study.py",
                "training.py",
                "metrics.py",
            )
        },
        "selection_scope": (
            "development only; the previously inspected evaluation is not a fresh holdout"
        ),
        "promotion": "requires a separately preregistered confirmatory temporal comparison",
    }
    contract_path = output / "study_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("learned feature study contract differs")
    atomic_json(contract_path, contract)
    if publish:
        publish(contract_path)
    study_id = canonical_json_sha256(contract)
    prepare_sequences(
        corpus / "history.parquet",
        output / "sequences",
        cutoff=int(protocol["history_end"]),
        expected_sha256=corpus_manifest["files"]["history.parquet"],
        threads=int(config["embedding"]["workers"]),
        logger=logger,
        publish=publish,
    )
    vectors = {}
    receipts = {}
    # Two independent native training tasks share one capped instance, with 16 workers each.
    # The candidate-feature stage waits for both immutable vector artifacts.
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {
            family: pool.submit(
                train_representation,
                output / f"sequences/{family}.txt",
                output / f"representations/{family}",
                config["embedding"],
                logger=logger,
                publish=publish,
            )
            for family in FAMILIES
        }
        for family in FAMILIES:
            receipts[family] = jobs[family].result()
    for family in FAMILIES:
        vectors[family] = Representation(output / f"representations/{family}/vectors.npz")
    queries = {role: Queries(corpus, role) for role in ("fit", "selection")}
    if set(queries["fit"].session) & set(queries["selection"].session):
        raise ValueError("fitting and selection sessions must be disjoint")
    working = output / "working"
    working.mkdir(exist_ok=True)
    caches = {}
    for role in ("fit", "selection"):
        augment_cache(
            inputs / f"{role}_cache",
            output / f"{role}_cache",
            queries[role],
            vectors,
            vector_hashes={f: r["sha256"] for f, r in receipts.items()},
            base_names=base_names,
            logger=logger,
            publish=publish,
        )
        with Heartbeat(logger, stage=f"load_augmented_{role}", interval_seconds=15):
            caches[role] = load_model_cache(output / f"{role}_cache", working / f"{role}.f32")
        if not np.array_equal(caches[role]["ids"], queries[role].session):
            raise ValueError("augmented cache must preserve the complete query ledger")
    fit, valid = caches["fit"], caches["selection"]
    if fit["names"] != valid["names"]:
        raise ValueError("fit and selection feature schemas differ")
    screen = screen_features(fit["x"], fit["names"], base_names, rows=int(config["screening_rows"]))
    atomic_json(output / "screening.json", screen)
    if publish:
        publish(output / "screening.json")
    retained = tuple(screen["features"])
    variants = {
        "baseline": base_names,
        **{
            f"with_{family}": tuple(
                n for n in retained if n in base_names or n.startswith(f"embedding_{family}_")
            )
            for family in FAMILIES
        },
        "with_both": retained,
    }
    denominator = queries["selection"].denominators
    fusion_hits = np.column_stack(
        [
            ranked_hits(
                valid["baseline"][:, j], valid["aids"], valid["target"][:, j], valid["groups"]
            )
            for j in range(3)
        ]
    )
    summaries: dict[str, Any] = {}
    statistics: dict[str, np.ndarray] = {}
    for variant, names in variants.items():
        columns = [fit["names"].index(name) for name in names]
        fit_x = np.ascontiguousarray(fit["x"][:, columns])
        valid_x = np.ascontiguousarray(valid["x"][:, columns])
        hits = np.zeros_like(denominator)
        states = {}
        for j, objective in enumerate(OBJECTIVES):
            directory = output / "models" / variant / objective
            logger.info(
                "representation_ranker_start", extra={"variant": variant, "objective": objective}
            )
            states[objective] = fit_model(
                fit_x,
                fit["target"][:, j],
                fit["groups"],
                valid_x,
                valid["target"][:, j],
                valid["aids"],
                valid["groups"],
                int(denominator[:, j].sum()),
                names=names,
                directory=directory,
                lineage={
                    "study_id": study_id,
                    "variant": variant,
                    "objective": objective,
                    "fit_role": "fit",
                    "selection_role": "selection",
                },
                config=config["training"],
                logger=logger,
                publish=publish,
            )
            model = lgb.Booster(model_file=str(directory / "model.txt"))
            prediction = np.asarray(
                model.predict(valid_x, num_threads=int(config["training"]["threads"]))
            )
            hits[:, j] = ranked_hits(
                prediction, valid["aids"], valid["target"][:, j], valid["groups"]
            )
        summary = official_score(hits, denominator)
        summary.update(
            features=len(names),
            models={
                o: {
                    "sha256": state["model_sha256"],
                    "best_iteration": state["best_iteration"],
                    "fit_seconds": state["retained_fit_seconds"],
                }
                for o, state in states.items()
            },
        )
        if variant != "baseline":
            summary["paired_selection_comparison"] = paired_bootstrap(
                hits,
                statistics["baseline"],
                denominator,
                seed=int(config["training"]["seed"]),
                replicates=1000,
            )
            summary["paired_selection_comparison"]["scope"] = (
                "exploratory selection-cohort comparison; selection optimism and "
                "multiple comparisons are not corrected"
            )
        summaries[variant], statistics[variant] = summary, hits
        directory = output / "models" / variant
        atomic_json(directory / "selection_metrics.json", summary)
        pl.DataFrame(
            {"session": valid["ids"], **{f"hits_{o}": hits[:, j] for j, o in enumerate(OBJECTIVES)}}
        ).write_parquet(directory / "selection_statistics.parquet")
        if publish:
            publish(directory / "selection_metrics.json")
            publish(directory / "selection_statistics.parquet")
        del fit_x, valid_x
    result = {
        "status": "passed",
        "study_id": study_id,
        "scope": "feature development on chronological selection data",
        "fit_sessions": len(fit["ids"]),
        "selection_sessions": len(valid["ids"]),
        "representations": receipts,
        "variants": summaries,
        "fusion": official_score(fusion_hits, denominator),
        "candidate_ceiling": official_score(
            np.minimum(20, np.add.reduceat(valid["target"], valid["starts"], axis=0)), denominator
        ),
        "best_selection_variant_by_objective": {
            o: min(
                summaries,
                key=lambda v: (
                    -summaries[v]["objectives"][o]["recall_at_20"],
                    summaries[v]["features"],
                    v,
                ),
            )
            for o in OBJECTIVES
        },
        "evaluation_access": False,
        "kaggle_promotion": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "results.json", result)
    if publish:
        publish(output / "results.json")
    return result
