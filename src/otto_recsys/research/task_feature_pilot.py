"""Bounded task-specific screening on certified, whole-session domain cache samples."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import OBJECTIVES, session_hash
from otto_recsys.research.materialize import MODEL_METADATA
from otto_recsys.research.metrics import official_score, paired_bootstrap, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.representation_study import screen_features
from otto_recsys.research.training import fit_model
from otto_recsys.runtime import Heartbeat


def sample_parts(parts: list[str], count: int) -> list[str]:
    """Evenly spaced full partitions; omit the potentially incomplete last partition."""
    if count < 2 or count >= len(parts) or len(set(parts)) != len(parts):
        raise ValueError("sample needs distinct full partitions and at least two positions")
    indices = np.floor(np.linspace(0, len(parts) - 2, count) + 0.5).astype(int)
    return [parts[i] for i in indices]


def load_sample(root: Path, role: str, config: dict[str, Any]) -> dict[str, Any]:
    """Reject altered caches, mixed roles, split queries and sampled selection negatives."""
    if role not in ("fit", "selection"):
        raise ValueError("only fitting and selection roles are permitted")
    directory = root / f"{role}_cache"
    path = directory / "manifest.json"
    if sha256_file(path) != config["source_manifests"][role]:
        raise ValueError("source cache manifest checksum differs from audited evidence")
    manifest = json.loads(path.read_text())
    contract = json.loads((directory / "contract.json").read_text())
    corpus = json.loads((root / "corpus/manifest.json").read_text())
    if (
        manifest["status"] != "passed"
        or manifest["role"] != role
        or contract["role"] != role
        or canonical_json_sha256(contract) != manifest["input_id"]
        or contract["corpus_id"] != corpus["input_id"]
        or contract["features"] != manifest["features"]
        or sha256_file(root / "corpus/manifest.json") != config["corpus_manifest_sha256"]
        or sha256_file(root / "corpus/queries.parquet") != config["queries_sha256"]
        or corpus["files"]["queries.parquet"] != config["queries_sha256"]
    ):
        raise ValueError("source cache role, schema, corpus or contract identity differs")
    names = tuple(manifest["features"])
    frames, selected = [], sample_parts(manifest["parts"], config["part_counts"][role])
    for part in selected:
        if Path(part).name != part or sha256_file(directory / part) != manifest["files"][part]:
            raise ValueError("candidate partition path or checksum differs")
        frame = pl.read_parquet(directory / part, columns=[*names, *MODEL_METADATA])
        ids = frame["session"].to_numpy()
        if (np.diff(ids) < 0).any() or np.unique(ids).size != config["sessions_per_part"]:
            raise ValueError("each sampled partition must contain sorted complete sessions")
        frames.append(frame)
    frame = pl.concat(frames)
    sessions, aids = frame["session"].to_numpy(), frame["aid"].to_numpy()
    ids, starts, groups = np.unique(sessions, return_index=True, return_counts=True)
    if (np.diff(sessions) < 0).any() or ids.size != len(selected) * config["sessions_per_part"]:
        raise ValueError("sampled sessions overlap or are split across partitions")
    positions = frame["candidate_position"].to_numpy()
    for start, count in zip(starts, groups, strict=True):
        stop = start + count
        if (
            np.unique(aids[start:stop]).size != count
            or (np.diff(positions[start:stop]) <= 0).any()
            or positions[start] < 0
            or positions[stop - 1] >= config["candidate_budget"]
            or (role == "selection" and count != config["candidate_budget"])
        ):
            raise ValueError("candidate identity/order or complete selection pool is invalid")
    ledger = pl.read_parquet(root / "corpus/queries.parquet").filter(
        (pl.col("split_role") == role) & pl.col("session").is_in(ids.tolist())
    )
    denominators = np.zeros((ids.size, 3), dtype=np.int64)
    for j, objective in enumerate(OBJECTIVES):
        rows = ledger.filter(pl.col("objective") == objective).sort("session")
        if not np.array_equal(rows["session"].to_numpy(), ids):
            raise ValueError("sample differs from the complete objective ledger")
        denominators[:, j] = rows["recall_denominator"].to_numpy()
    x = frame.select(names).to_numpy().astype(np.float32)
    target = frame.select([f"target_{o}" for o in OBJECTIVES]).to_numpy()
    if not np.isfinite(x).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("nonfinite features or nonbinary targets are forbidden")
    coverage = np.minimum(20, np.add.reduceat(target, starts, axis=0))
    official_score(coverage, denominators)
    return {
        "role": role,
        "names": names,
        "x": x,
        "target": target,
        "ids": ids,
        "sessions": sessions,
        "aids": aids,
        "groups": groups,
        "denominators": denominators,
        "coverage": coverage,
        "baseline": frame.select([f"baseline_{o}" for o in OBJECTIVES]).to_numpy(),
        "identity": {
            "cache_id": manifest["input_id"],
            "parts": selected,
            "part_hashes": {p: manifest["files"][p] for p in selected},
            "sessions": int(ids.size),
            "rows": int(frame.height),
        },
    }


def select_columns(
    names: tuple[str, ...], base: tuple[str, ...], gains: np.ndarray, maximum: int
) -> dict[str, Any]:
    """Equal-budget shared versus per-task shortlists from fitting-fold gains only."""
    if gains.shape != (3, 3, len(names)) or maximum < 1 or not set(base).issubset(names):
        raise ValueError("three fitting folds and three objectives must align with feature names")
    if not np.isfinite(gains).all() or (gains < 0).any():
        raise ValueError("feature gains must be finite and nonnegative")
    norm = gains / np.maximum(gains.sum(axis=2, keepdims=True), 1e-12)
    mean = norm.mean(axis=0)
    support = (gains > 0).sum(axis=0)
    added = [i for i, n in enumerate(names) if n not in base]

    def choose(values: np.ndarray, eligible: np.ndarray) -> list[str]:
        order = sorted((i for i in added if eligible[i]), key=lambda i: (-values[i], names[i]))
        return [*base, *(names[i] for i in order[:maximum])]

    return {
        "shared": choose(np.array([0.1, 0.3, 0.6]) @ mean, (support >= 2).any(axis=0)),
        "per_task": {o: choose(mean[j], support[j] >= 2) for j, o in enumerate(OBJECTIVES)},
        "normalized_mean_gain": {
            o: dict(zip(names, mean[j].tolist(), strict=True)) for j, o in enumerate(OBJECTIVES)
        },
        "positive_gain_folds": {
            o: dict(zip(names, support[j].tolist(), strict=True)) for j, o in enumerate(OBJECTIVES)
        },
        "interpretation": "Fitting-only shortlist heuristic; gain is not held-out feature utility.",
    }


def screen(
    fit: dict[str, Any],
    output: Path,
    config: dict[str, Any],
    study_id: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    if fit["role"] != "fit":
        raise ValueError("utility screening accepts fitting data only")
    base = tuple(config["baseline_features"])
    allowed = (*base, *(n for n in fit["names"] if n.startswith("domain_")))
    indices = [fit["names"].index(n) for n in allowed]
    quality = screen_features(fit["x"][:, indices], allowed, base, rows=config["quality_rows"])
    names = tuple(quality["features"])
    x = np.ascontiguousarray(fit["x"][:, [fit["names"].index(n) for n in names]])
    fold = (session_hash(fit["sessions"], config["seed"]) % 3).astype(np.int8)
    gains, diagnostics = np.zeros((3, 3, len(names))), []
    output.mkdir(parents=True, exist_ok=True)
    for f in range(3):
        train, valid = fold != f, fold == f
        if not train.any() or not valid.any():
            raise ValueError("every fitting fold requires complete training and validation groups")
        for j, objective in enumerate(OBJECTIVES):
            model_path, receipt_path = (
                output / f"{f}-{objective}.txt",
                output / f"{f}-{objective}.json",
            )
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["study_id"] != study_id or sha256_file(model_path) != receipt["sha256"]:
                    raise ValueError("screening checkpoint identity or checksum differs")
                model = lgb.Booster(model_file=str(model_path))
                if model.feature_name() != list(names):
                    raise ValueError("screening checkpoint feature order differs")
            else:
                logger.info("task_screen_start", extra={"fold": f, "objective": objective})
                with Heartbeat(logger, stage=f"screen_{f}_{objective}", interval_seconds=15):
                    model = lgb.train(
                        {
                            "objective": "binary",
                            "metric": "None",
                            "verbosity": -1,
                            "num_threads": config["threads"],
                            "num_leaves": 15,
                            "learning_rate": 0.08,
                            "min_data_in_leaf": 40,
                            "seed": config["seed"] + f,
                            "deterministic": True,
                            "force_col_wise": True,
                        },
                        lgb.Dataset(
                            x[train], label=fit["target"][train, j], feature_name=list(names)
                        ),
                        num_boost_round=config["screen_rounds"],
                    )
                prediction = np.clip(
                    model.predict(x[valid], num_threads=config["threads"]), 1e-7, 1 - 1e-7
                )
                truth = fit["target"][valid, j]
                loss = float(
                    -np.mean(truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction))
                )
                temporary = model_path.with_suffix(".tmp")
                model.save_model(str(temporary))
                temporary.replace(model_path)
                receipt = {
                    "study_id": study_id,
                    "sha256": sha256_file(model_path),
                    "fold": f,
                    "objective": objective,
                    "sampled_candidate_logloss": loss,
                    "train_sessions": int(np.unique(fit["sessions"][train]).size),
                    "validation_sessions": int(np.unique(fit["sessions"][valid]).size),
                }
                atomic_json(receipt_path, receipt)
            gains[f, j] = model.feature_importance(importance_type="gain")
            diagnostics.append(receipt)
    result = {
        "study_id": study_id,
        "quality": quality,
        "pilots": diagnostics,
        **select_columns(names, base, gains, config["maximum_added"]),
    }
    atomic_json(output / "selection.json", result)
    return result


def run(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        return _run(inputs, output, config, logger=logger, publish=publish)


def _run(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = {
        "config": config,
        "lightgbm": lgb.__version__,
        "code": {
            n: sha256_file(Path(__file__).with_name(n))
            for n in (
                "task_feature_pilot.py",
                "training.py",
                "metrics.py",
                "representation_study.py",
                "dataset.py",
            )
        },
    }
    identity = canonical_json_sha256(contract)
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("pilot output has a different experiment contract")
    atomic_json(path, contract)
    with Heartbeat(logger, stage="load_fitting_sample", interval_seconds=15):
        fit = load_sample(inputs, "fit", config)
    screening = screen(fit, output / "screening", config, identity, logger)
    # Freeze shortlists before loading any chronological selection candidate values/targets.
    with Heartbeat(logger, stage="load_selection_sample", interval_seconds=15):
        valid = load_sample(inputs, "selection", config)
    if fit["names"] != valid["names"] or np.intersect1d(fit["ids"], valid["ids"]).size:
        raise ValueError("fit and selection require aligned schemas and disjoint sessions")
    arms = {
        "baseline": {o: config["baseline_features"] for o in OBJECTIVES},
        "shared": {o: screening["shared"] for o in OBJECTIVES},
        "per_task": screening["per_task"],
    }
    result: dict[str, Any] = {
        "status": "running",
        "study_id": identity,
        "scope": "Small stratified partition sample of previously used development cohorts.",
        "evaluation_access": False,
        "kaggle_promotion": False,
        "inputs": {"fit": fit["identity"], "selection": valid["identity"]},
        "candidate_ceiling": official_score(valid["coverage"], valid["denominators"]),
        "screening_sha256": sha256_file(output / "screening/selection.json"),
        "arms": {},
    }
    all_hits = {}
    for arm, feature_sets in arms.items():
        hits = np.zeros_like(valid["denominators"])
        model_reports = {}
        for j, objective in enumerate(OBJECTIVES):
            names = tuple(feature_sets[objective])
            columns = [fit["names"].index(n) for n in names]
            directory = output / "models" / arm / objective
            logger.info(
                "pilot_ranker_start",
                extra={"arm": arm, "objective": objective, "features": len(names)},
            )
            model_reports[objective] = fit_model(
                np.ascontiguousarray(fit["x"][:, columns]),
                fit["target"][:, j],
                fit["groups"],
                np.ascontiguousarray(valid["x"][:, columns]),
                valid["target"][:, j],
                valid["aids"],
                valid["groups"],
                int(valid["denominators"][:, j].sum()),
                names=names,
                directory=directory,
                lineage={
                    "study_id": identity,
                    "arm": arm,
                    "objective": objective,
                    "fit_role": "fit",
                    "selection_role": "selection",
                    "fit_sample": fit["identity"],
                    "selection_sample": valid["identity"],
                },
                config=config["training"],
                logger=logger,
                publish=publish,
            )
            model = lgb.Booster(model_file=str(directory / "model.txt"))
            prediction = np.asarray(
                model.predict(valid["x"][:, columns], num_threads=config["threads"])
            )
            hits[:, j] = ranked_hits(
                prediction, valid["aids"], valid["target"][:, j], valid["groups"]
            )
        all_hits[arm] = hits
        summary = {
            **official_score(hits, valid["denominators"]),
            "features": feature_sets,
            "models": {
                o: {k: s[k] for k in ("model_sha256", "best_iteration")}
                for o, s in model_reports.items()
            },
        }
        if arm != "baseline":
            summary["versus_baseline"] = paired_bootstrap(
                hits, all_hits["baseline"], valid["denominators"], seed=config["seed"]
            )
        if arm == "per_task":
            summary["versus_shared"] = paired_bootstrap(
                hits, all_hits["shared"], valid["denominators"], seed=config["seed"]
            )
        result["arms"][arm] = summary
        statistics = {
            "session": valid["ids"],
            **{f"denominator_{o}": valid["denominators"][:, j] for j, o in enumerate(OBJECTIVES)},
            **{f"coverage_{o}": valid["coverage"][:, j] for j, o in enumerate(OBJECTIVES)},
            **{f"hits_{o}": hits[:, j] for j, o in enumerate(OBJECTIVES)},
        }
        stats_path = output / "models" / arm / "statistics.parquet"
        temporary = stats_path.with_suffix(".tmp")
        pl.DataFrame(statistics).write_parquet(temporary)
        temporary.replace(stats_path)
        summary["statistics_sha256"] = sha256_file(stats_path)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["status"] = "passed" if arm == "per_task" else "running"
        atomic_json(output / "results.json", result)
        if publish:
            publish(stats_path)
            publish(output / "results.json")
        logger.info(
            "pilot_arm_complete",
            extra={
                "arm": arm,
                "score": summary["weighted_recall_at_20"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
        )
    return result
