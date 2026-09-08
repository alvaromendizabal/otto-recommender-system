"""Matched feature-family ablations and an immutable final evaluation choice."""

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
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.materialize import MODEL_METADATA
from otto_recsys.research.metrics import official_score, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.screening import core_features
from otto_recsys.research.training import fit_model
from otto_recsys.runtime import Heartbeat


def load_model_cache(directory: Path, working: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("ranker requires a completed feature cache")
    names = tuple(manifest["features"])
    rows = int(manifest["rows"])
    x = np.memmap(working, dtype=np.float32, mode="w+", shape=(rows, len(names)))
    target = np.empty((rows, 3), dtype=np.int8)
    aids = np.empty(rows, dtype=np.int32)
    sessions = np.empty(rows, dtype=np.int32)
    position = np.empty(rows, dtype=np.int16)
    baseline = np.empty((rows, 3), dtype=np.float32)
    offset = 0
    for name in manifest["parts"]:
        path = directory / name
        if sha256_file(path) != manifest["files"][name]:
            raise ValueError("model feature cache failed checksum verification")
        frame = pl.read_parquet(path, columns=[*names, *MODEL_METADATA])
        stop = offset + frame.height
        x[offset:stop] = frame.select(names).to_numpy()
        target[offset:stop] = frame.select([f"target_{o}" for o in OBJECTIVES]).to_numpy()
        baseline[offset:stop] = frame.select([f"baseline_{o}" for o in OBJECTIVES]).to_numpy()
        aids[offset:stop] = frame["aid"].to_numpy()
        sessions[offset:stop] = frame["session"].to_numpy()
        position[offset:stop] = frame["candidate_position"].to_numpy()
        offset = stop
    if offset != rows or (np.diff(sessions) < 0).any():
        raise ValueError("model cache must contain every row in sorted whole-session groups")
    ids, starts, groups = np.unique(sessions, return_index=True, return_counts=True)
    x.flush()
    return {
        "x": x,
        "target": target,
        "aids": aids,
        "sessions": sessions,
        "position": position,
        "baseline": baseline,
        "ids": ids,
        "starts": starts,
        "groups": groups,
        "names": names,
        "manifest": manifest,
    }


def variants(names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    family = {f.name: f.family for f in feature_catalog()}
    result = {"core": tuple(n for n in core_features() if n in names), "full": names}
    for group in ("graph", "history", "repeat", "context", "interaction", "source"):
        selected = tuple(n for n in names if family[n] != group)
        if selected and selected != names:
            result[f"without_{group}"] = selected
    return result


def _run_ablations(
    root: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    output = root / "models"
    output.mkdir(parents=True, exist_ok=True)
    working = root / "working"
    working.mkdir(exist_ok=True)
    start = time.perf_counter()
    with Heartbeat(logger, stage="model_cache_load", interval_seconds=15):
        fit = load_model_cache(root / "fit_cache", working / "fit.f32")
        valid = load_model_cache(root / "selection_cache", working / "selection.f32")
        fit_queries = Queries(root / "corpus", "fit")
        valid_queries = Queries(root / "corpus", "selection")
    if (
        fit["names"] != valid["names"]
        or set(fit["ids"]) & set(valid["ids"])
        or not np.array_equal(fit["ids"], fit_queries.session)
        or not np.array_equal(valid["ids"], valid_queries.session)
    ):
        raise ValueError("ranker caches must match disjoint complete temporal query ledgers")
    if fit["manifest"]["role"] != "fit" or valid["manifest"]["role"] != "selection":
        raise ValueError("incorrect fitting or selection cache role")
    names = fit["names"]
    experiments = variants(names)
    denominator = valid_queries.denominators
    baseline_hits = np.column_stack(
        [
            ranked_hits(
                valid["baseline"][:, j], valid["aids"], valid["target"][:, j], valid["groups"]
            )
            for j in range(3)
        ]
    )
    frontier = {}
    for budget in (100, 200, 400):
        covered = np.add.reduceat(
            valid["target"] * (valid["position"][:, None] < budget), valid["starts"], axis=0
        )
        frontier[str(budget)] = official_score(np.minimum(20, covered), denominator)
    contract = {
        "fit_cache_id": fit["manifest"]["input_id"],
        "selection_cache_id": valid["manifest"]["input_id"],
        "selected_features_sha256": sha256_file(root / "screening/selection.json"),
        "corpus_id": fit_queries.manifest["input_id"],
        "config": config,
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("study.py", "training.py", "metrics.py")
        },
        "candidate_budget": 400,
        "feature_variants": {k: list(v) for k, v in experiments.items()},
        "evaluation_access": "no evaluation labels accepted by this stage",
    }
    input_id = canonical_json_sha256(contract)
    contract_path = output / "study_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("ablation workspace belongs to a different experiment")
    atomic_json(contract_path, contract)
    if publish:
        publish(contract_path)
    summaries: dict[str, Any] = {}
    model_states: dict[str, Any] = {}
    for variant, columns in experiments.items():
        indices = [names.index(n) for n in columns]
        train_x = np.ascontiguousarray(fit["x"][:, indices])
        valid_x = np.ascontiguousarray(valid["x"][:, indices])
        hits = np.zeros_like(denominator)
        model_states[variant] = {}
        for j, objective in enumerate(OBJECTIVES):
            logger.info(
                "research_model_start",
                extra={"stage": f"{variant}/{objective}", "features": len(columns)},
            )
            lineage = {
                "study_id": input_id,
                "variant": variant,
                "objective": objective,
                "fit_role": "fit",
                "selection_role": "selection",
                "fit_cache_id": fit["manifest"]["input_id"],
                "selection_cache_id": valid["manifest"]["input_id"],
                "fit_sessions": len(fit["ids"]),
                "selection_sessions": len(valid["ids"]),
                "temporal_session_disjointness_verified": True,
            }
            state = fit_model(
                train_x,
                fit["target"][:, j],
                fit["groups"],
                valid_x,
                valid["target"][:, j],
                valid["aids"],
                valid["groups"],
                int(denominator[:, j].sum()),
                names=columns,
                directory=output / variant / objective,
                lineage=lineage,
                config=config,
                logger=logger,
                publish=publish,
            )
            booster = lgb.Booster(model_file=str(output / variant / objective / "model.txt"))
            prediction = booster.predict(valid_x, num_threads=int(config["threads"]))
            hits[:, j] = ranked_hits(
                np.asarray(prediction), valid["aids"], valid["target"][:, j], valid["groups"]
            )
            model_states[variant][objective] = state
        summary = official_score(hits, denominator)
        summary.update(
            features=len(columns),
            models={
                o: {
                    "best_iteration": s["best_iteration"],
                    "fit_seconds": s["retained_fit_seconds"],
                    "model_sha256": s["model_sha256"],
                }
                for o, s in model_states[variant].items()
            },
        )
        summaries[variant] = summary
        statistics = pl.DataFrame(
            {"session": valid["ids"], **{f"hits_{o}": hits[:, j] for j, o in enumerate(OBJECTIVES)}}
        )
        statistics.write_parquet(output / variant / "selection_statistics.parquet")
        atomic_json(output / variant / "selection_metrics.json", summary)
        if publish:
            publish(output / variant / "selection_statistics.parquet")
            publish(output / variant / "selection_metrics.json")
        del train_x, valid_x
    chosen = {}
    for objective in OBJECTIVES:
        variant = min(
            summaries,
            key=lambda v: (
                -summaries[v]["objectives"][objective]["recall_at_20"],
                summaries[v]["features"],
                v,
            ),
        )
        model = output / variant / objective / "model.txt"
        chosen[objective] = {
            "variant": variant,
            "path": str(model.relative_to(root)),
            "sha256": sha256_file(model),
            "features": list(experiments[variant]),
        }
    retrieval = json.loads((root / "retrieval/manifest.json").read_text())
    seal = {
        "study_id": input_id,
        "corpus_id": fit_queries.manifest["input_id"],
        "retrieval_id": retrieval["input_id"],
        "candidate_budget": 400,
        "models": chosen,
        "core_models": {
            o: {
                "path": f"models/core/{o}/model.txt",
                "features": list(experiments["core"]),
                "sha256": sha256_file(output / "core" / o / "model.txt"),
            }
            for o in OBJECTIVES
        },
        "feature_code_sha256": sha256_file(Path(__file__).with_name("features.py")),
        "selection_rule": "best per-objective selection Recall@20; "
        "fewer features then name break ties",
        "evaluation_labels_consulted": False,
    }
    seal["seal_id"] = canonical_json_sha256(seal)
    seal_path = root / "evaluation_seal.json"
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise ValueError("final model choice is already sealed to a different experiment")
    atomic_json(seal_path, seal)
    report = {
        "status": "passed",
        "study_id": input_id,
        "selection_sessions": len(valid["ids"]),
        "candidate_budget": 400,
        "candidate_frontier": frontier,
        "fusion_baseline": official_score(baseline_hits, denominator),
        "variants": summaries,
        "chosen": {o: chosen[o]["variant"] for o in OBJECTIVES},
        "seal_id": seal["seal_id"],
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "model selection only; reserved temporal evaluation has not run",
    }
    atomic_json(output / "ablation_report.json", report)
    if publish:
        publish(output / "ablation_report.json")
        publish(seal_path)
    del fit, valid
    for path in (working / "fit.f32", working / "selection.f32"):
        path.unlink()
    return report


def run_ablations(
    root: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Reject duplicate study writers before creating shared working matrices."""
    directory = root / "models"
    directory.mkdir(parents=True, exist_ok=True)
    with workspace_lock(directory):
        return _run_ablations(root, config, logger=logger, publish=publish)
