"""Frozen feature validation and resumable, gated leave-one-block-out ablations."""

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
from otto_recsys.research.dataset import OBJECTIVES
from otto_recsys.research.metrics import official_score, paired_bootstrap, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.task_feature_pilot import load_sample
from otto_recsys.research.training import fit_model
from otto_recsys.runtime import Heartbeat


def frozen_arms(inputs: Path, config: dict[str, Any]) -> dict[str, list[str]]:
    """Verify the prior fitting-only shortlist; never repeat or revise screening."""
    path = inputs / "frozen_screening.json"
    if sha256_file(path) != config["source_screening_sha256"]:
        raise ValueError("frozen screening checksum differs")
    prior = json.loads(path.read_text())
    base, added = config["baseline_features"], config["added_features"]
    shared = [*base, *added]
    blocks = config["ablation_blocks"]
    flattened = [name for values in blocks.values() for name in values]
    if (
        prior["study_id"] != config["source_study_id"]
        or prior["shared"] != shared
        or not base
        or not added
        or len(set(shared)) != len(shared)
        or set(flattened) != set(added)
        or len(flattened) != len(added)
        or any(not values for values in blocks.values())
        or any(not key.replace("_", "").isalnum() for key in blocks)
    ):
        raise ValueError("frozen schema or disjoint ablation blocks differ")
    return {
        "baseline": base,
        "shared": shared,
        **{
            f"without_{key}": [n for n in shared if n not in values]
            for key, values in blocks.items()
        },
    }


def advancement(arms: dict[str, Any], minimum_gain: float) -> dict[str, Any]:
    """A point gate buys further development experiments, never promotion."""
    if not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum gain must be finite and nonnegative")
    baseline, shared = arms["baseline"], arms["shared"]
    gain = shared["weighted_recall_at_20"] - baseline["weighted_recall_at_20"]
    order_gain = (
        shared["objectives"]["orders"]["recall_at_20"]
        - baseline["objectives"]["orders"]["recall_at_20"]
    )
    if not np.isfinite([gain, order_gain]).all():
        raise ValueError("nonfinite validation metric")
    return {
        "advance_to_ablations": bool(gain >= minimum_gain and order_gain >= 0),
        "weighted_gain": gain,
        "order_gain": order_gain,
        "minimum_weighted_gain": minimum_gain,
        "scope": "Prespecified development point gate; no independent confirmation or promotion.",
    }


def query_slices(inputs: Path, ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Observed prefix support and equal-count chronological query-time blocks."""
    ledger = (
        pl.read_parquet(
            inputs / "corpus/queries.parquet",
            columns=["session", "split_role", "objective", "query_ts", "observed_events"],
        )
        .filter(
            (pl.col("split_role") == "selection")
            & (pl.col("objective") == "clicks")
            & pl.col("session").is_in(ids.tolist())
        )
        .sort("session")
    )
    if not np.array_equal(ledger["session"].to_numpy(), ids):
        raise ValueError("slice ledger differs from complete selection sessions")
    ts, events = ledger["query_ts"].to_numpy(), ledger["observed_events"].to_numpy()
    order = np.lexsort((ids, ts))
    quartile = np.empty(ids.size, dtype=np.int8)
    quartile[order] = np.arange(ids.size) * 4 // ids.size
    masks = {f"time_q{i + 1}": quartile == i for i in range(4)}
    masks.update(
        prefix_1=events == 1, prefix_2_to_5=(events >= 2) & (events <= 5), prefix_6_plus=events >= 6
    )
    return masks, {"query_ts": ts, "observed_events": events, "time_quartile": quartile}


def run(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
    phase: str = "validation",
) -> dict[str, Any]:
    if phase not in {"validation", "ablation"}:
        raise ValueError("only validation and ablation phases are permitted")
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        return _run(inputs, output, config, logger=logger, publish=publish, phase=phase)


def _run(
    inputs: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None,
    phase: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    schemas = frozen_arms(inputs, config)
    contract = {
        "config": config,
        "schemas": schemas,
        "lightgbm": lgb.__version__,
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "shared_feature_validation.py",
                "task_feature_pilot.py",
                "training.py",
                "metrics.py",
                "dataset.py",
            )
        },
    }
    identity = canonical_json_sha256(contract)
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("shared validation output has a different experiment contract")
    previous_path = output / "validation_results.json"
    if phase == "ablation":
        if not previous_path.is_file():
            raise ValueError("ablation requires the completed validation checkpoint")
        previous = json.loads(previous_path.read_text())
        if (
            previous["status"] != "passed"
            or previous["study_id"] != identity
            or not advancement(previous["arms"], config["minimum_weighted_gain"])[
                "advance_to_ablations"
            ]
        ):
            raise ValueError("ablation requires a matching passed validation gate")
        for arm in ("baseline", "shared"):
            for objective, model in previous["arms"][arm]["models"].items():
                directory = output / "models" / arm / objective
                manifest = json.loads((directory / "manifest.json").read_text())
                checkpoint = directory / "checkpoints" / f"{manifest['iteration']:06d}.txt"
                if (
                    sha256_file(checkpoint) != manifest["sha256"]
                    or sha256_file(directory / "model.txt") != model["model_sha256"]
                    or not manifest["complete"]
                ):
                    raise ValueError("completed validation model is missing or altered")
    atomic_json(path, contract)
    if publish:
        publish(path)
    # Schema identity is fixed before either sample is loaded.
    with Heartbeat(logger, stage="load_frozen_validation_inputs", interval_seconds=15):
        fit, valid = load_sample(inputs, "fit", config), load_sample(inputs, "selection", config)
    if fit["names"] != valid["names"] or np.intersect1d(fit["ids"], valid["ids"]).size:
        raise ValueError("fitting and selection need aligned schemas and disjoint sessions")
    masks, context = query_slices(inputs, valid["ids"])
    result: dict[str, Any] = {
        "status": "running",
        "study_id": identity,
        "phase": phase,
        "evaluation_access": False,
        "kaggle_promotion": False,
        "scope": "Systematic larger sample of reused development cohorts; no fresh holdout.",
        "source_screening_sha256": config["source_screening_sha256"],
        "inputs": {"fit": fit["identity"], "selection": valid["identity"]},
        "candidate_ceiling": official_score(valid["coverage"], valid["denominators"]),
        "slice_scope": "Query-time quartiles and observed prefix lengths; descriptive, one seed.",
        "arms": {},
    }
    all_hits: dict[str, np.ndarray] = {}
    chosen = list(schemas) if phase == "ablation" else ["baseline", "shared"]
    for arm in chosen:
        names = tuple(schemas[arm])
        columns = [fit["names"].index(n) for n in names]
        fit_x = np.ascontiguousarray(fit["x"][:, columns])
        valid_x = np.ascontiguousarray(valid["x"][:, columns])
        hits, reports = np.zeros_like(valid["denominators"]), {}
        for j, objective in enumerate(OBJECTIVES):
            directory = output / "models" / arm / objective
            logger.info(
                "shared_ranker_start",
                extra={"arm": arm, "objective": objective, "features": len(names)},
            )
            reports[objective] = fit_model(
                fit_x,
                fit["target"][:, j],
                fit["groups"],
                valid_x,
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
            prediction = np.asarray(model.predict(valid_x, num_threads=config["threads"]))
            hits[:, j] = ranked_hits(
                prediction, valid["aids"], valid["target"][:, j], valid["groups"]
            )
        del fit_x, valid_x
        all_hits[arm] = hits
        summary = {
            **official_score(hits, valid["denominators"]),
            "features": {o: list(names) for o in OBJECTIVES},
            "models": {
                o: {k: report[k] for k in ("model_sha256", "best_iteration")}
                for o, report in reports.items()
            },
            "slices": {
                key: {
                    "sessions": int(mask.sum()),
                    **(
                        official_score(hits[mask], valid["denominators"][mask])
                        if (valid["denominators"][mask].sum(axis=0) > 0).all()
                        else {"status": "insufficient_target_support"}
                    ),
                }
                for key, mask in masks.items()
            },
        }
        if arm != "baseline":
            summary["versus_baseline"] = paired_bootstrap(
                hits, all_hits["baseline"], valid["denominators"], seed=config["seed"]
            )
        if arm.startswith("without_"):
            summary["full_minus_drop"] = paired_bootstrap(
                all_hits["shared"], hits, valid["denominators"], seed=config["seed"]
            )
        stats_path = output / "models" / arm / "statistics.parquet"
        temporary = stats_path.with_suffix(".tmp")
        pl.DataFrame(
            {
                "session": valid["ids"],
                **context,
                **{
                    f"denominator_{o}": valid["denominators"][:, j]
                    for j, o in enumerate(OBJECTIVES)
                },
                **{f"coverage_{o}": valid["coverage"][:, j] for j, o in enumerate(OBJECTIVES)},
                **{f"hits_{o}": hits[:, j] for j, o in enumerate(OBJECTIVES)},
            }
        ).write_parquet(temporary)
        temporary.replace(stats_path)
        summary["statistics_sha256"] = sha256_file(stats_path)
        result["arms"][arm] = summary
        if "shared" in result["arms"]:
            result["decision"] = advancement(result["arms"], config["minimum_weighted_gain"])
        if (
            phase == "ablation"
            and arm == "shared"
            and not result["decision"]["advance_to_ablations"]
        ):
            raise ValueError("replayed validation gate does not support ablations")
        result["elapsed_seconds"] = time.perf_counter() - started
        result["status"] = "passed" if arm == chosen[-1] else "running"
        atomic_json(output / "results.json", result)
        if publish:
            publish(stats_path)
            publish(output / "results.json")
        logger.info(
            "shared_arm_complete",
            extra={
                "arm": arm,
                "score": summary["weighted_recall_at_20"],
                "completed_arms": len(all_hits),
                "total_arms": len(chosen),
                "elapsed_seconds": result["elapsed_seconds"],
            },
        )
    atomic_json(output / f"{phase}_results.json", result)
    if publish:
        publish(output / f"{phase}_results.json")
    return result
