"""Matched wider/forward co-visitation experiment on development queries only."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.directed_retrievers import build_retrievers
from otto_recsys.research.materialize import model_cache
from otto_recsys.research.metrics import official_score, paired_bootstrap, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.study import load_model_cache
from otto_recsys.research.training import fit_model


def fit_arm(
    cache_root: Path,
    output: Path,
    queries: dict[str, Queries],
    names: tuple[str, ...],
    config: dict[str, Any],
    lineage: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None],
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit identical task rankers and retain paired complete-query statistics."""
    output.mkdir(parents=True, exist_ok=True)
    caches = {
        role: load_model_cache(cache_root / f"{role}_cache", output / f"{role}.f32")
        for role in ("fit", "selection")
    }
    for role, cache in caches.items():
        if cache["manifest"]["role"] != role or not np.array_equal(
            cache["ids"], queries[role].session
        ):
            raise ValueError("candidate cache differs from the complete role ledger")
    if set(caches["fit"]["ids"]) & set(caches["selection"]["ids"]):
        raise ValueError("fit and selection sessions overlap")
    fit, valid = caches["fit"], caches["selection"]
    train_x = np.ascontiguousarray(fit["x"][:, [fit["names"].index(n) for n in names]])
    valid_x = np.ascontiguousarray(valid["x"][:, [valid["names"].index(n) for n in names]])
    denominator = queries["selection"].denominators
    hits = np.zeros_like(denominator)
    states = {}
    for j, objective in enumerate(OBJECTIVES):
        directory = output / objective
        states[objective] = fit_model(
            train_x,
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
                **lineage,
                "objective": objective,
                "fit_role": "fit",
                "selection_role": "selection",
                "fit_cache_id": fit["manifest"]["input_id"],
                "selection_cache_id": valid["manifest"]["input_id"],
            },
            config=config,
            logger=logger,
            publish=publish,
        )
        model = lgb.Booster(model_file=str(directory / "model.txt"))
        scores = np.asarray(model.predict(valid_x, num_threads=int(config["threads"])))
        hits[:, j] = ranked_hits(scores, valid["aids"], valid["target"][:, j], valid["groups"])
    coverage = np.minimum(20, np.add.reduceat(valid["target"], valid["starts"], axis=0))
    fusion = np.column_stack(
        [
            ranked_hits(
                valid["baseline"][:, j], valid["aids"], valid["target"][:, j], valid["groups"]
            )
            for j in range(3)
        ]
    )
    result = {
        **official_score(hits, denominator),
        "candidate_ceiling": official_score(coverage, denominator),
        "fixed_fusion": official_score(fusion, denominator),
        "features": list(names),
        "models": {
            o: {"sha256": s["model_sha256"], "best_iteration": s["best_iteration"]}
            for o, s in states.items()
        },
    }
    pl.DataFrame(
        {
            "session": valid["ids"],
            **{f"hits_{o}": hits[:, j] for j, o in enumerate(OBJECTIVES)},
            **{f"coverage_{o}": coverage[:, j] for j, o in enumerate(OBJECTIVES)},
        }
    ).write_parquet(output / "selection_statistics.parquet")
    publish(output / "selection_statistics.parquet")
    atomic_json(output / "selection_metrics.json", result)
    publish(output / "selection_metrics.json")
    del train_x, valid_x, fit, valid, caches
    for role in ("fit", "selection"):
        (output / f"{role}.f32").unlink(missing_ok=True)
    return result, hits


def run_study(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        return _run_study(inputs, output, config, logger=logger, publish=publish)


def _run_study(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None],
) -> dict[str, Any]:
    queries = {role: Queries(inputs / "corpus", role) for role in ("fit", "selection")}
    names = tuple(config["baseline_features"])
    if not names or len(names) != len(set(names)):
        raise ValueError("baseline feature schema must be nonempty and unique")
    contract = {
        "config": config,
        "corpus_id": queries["fit"].manifest["input_id"],
        "reference_cache_ids": {
            role: json.loads((inputs / f"{role}_cache/manifest.json").read_text())["input_id"]
            for role in queries
        },
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "retrieval_study.py",
                "retrievers.py",
                "directed_retrievers.py",
                "features.py",
                "materialize.py",
                "dataset.py",
                "training.py",
                "metrics.py",
                "study.py",
            )
        },
        "candidate_budget": 400,
        "negative_budget": 60,
        "scope": "development selection; not an independent holdout or Kaggle result",
        "promotion": "separate confirmatory temporal comparison required",
    }
    path = output / "study_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("retrieval study workspace belongs to a different contract")
    atomic_json(path, contract)
    publish(path)
    study_id = canonical_json_sha256(contract)
    summaries: dict[str, Any] = {}
    statistics: dict[str, np.ndarray] = {}
    for arm in ("baseline", "wide_symmetric", "wide_forward"):
        logger.info("retrieval_arm_start", extra={"arm": arm})
        root = inputs if arm == "baseline" else output / arm
        if arm != "baseline":
            build_retrievers(
                inputs / "corpus",
                root / "retrieval",
                {
                    **config["retrieval"],
                    "direction": "symmetric" if arm == "wide_symmetric" else "forward",
                },
                threads=int(config["graph_threads"]),
                memory_gib=int(config["graph_memory_gib"]),
                logger=logger,
                publish=publish,
            )
            for role in ("fit", "selection"):
                model_cache(
                    inputs / "corpus",
                    root / "retrieval",
                    root / f"{role}_cache",
                    role=role,
                    names=names,
                    budget=400,
                    negatives=60,
                    seed=int(config["training"]["seed"]),
                    workers=int(config["feature_workers"]),
                    logger=logger,
                    publish=publish,
                )
        summary, hits = fit_arm(
            root,
            output / "models" / arm,
            queries,
            names,
            config["training"],
            {"study_id": study_id, "arm": arm},
            logger=logger,
            publish=publish,
        )
        if (
            arm == "baseline"
            and "expected_baseline_score" in config
            and abs(summary["weighted_recall_at_20"] - config["expected_baseline_score"]) > 1e-12
        ):
            raise ValueError("matched baseline does not reproduce the certified selection score")
        if arm != "baseline":
            summary["paired_selection_comparison"] = paired_bootstrap(
                hits,
                statistics["baseline"],
                queries["selection"].denominators,
                seed=int(config["training"]["seed"]),
            )
        summaries[arm], statistics[arm] = summary, hits
        result = {
            "status": "passed" if arm == "wide_forward" else "running",
            "study_id": study_id,
            "scope": contract["scope"],
            "arms": summaries,
        }
        atomic_json(output / "results.json", result)
        publish(output / "results.json")
    return result
