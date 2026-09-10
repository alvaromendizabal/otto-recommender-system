"""Preregistered domain comparisons on unchanged development candidate rows."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.domain_feature_cache import augment_cache
from otto_recsys.research.domain_features import NormalizedGraphSignals, feature_names
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals
from otto_recsys.research.materialize import MODEL_METADATA
from otto_recsys.research.metrics import official_score, paired_bootstrap, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.representation_study import screen_features
from otto_recsys.research.retrieval_study import fit_arm
from otto_recsys.runtime import Heartbeat

ARM_FAMILIES = {
    "sequence": ("funnel", "episode"),
    "raw_graph": ("raw_graph",),
    "normalized_graph": ("normalized_graph",),
    "combined": ("funnel", "episode", "normalized_graph"),
}
REFERENCE_FILES = (
    *(f"{o}/{f}" for o in OBJECTIVES for f in ("model.txt", "manifest.json", "contract.json")),
    "selection_statistics.parquet",
    "selection_metrics.json",
)


def replay_baseline(
    inputs: Path,
    output: Path,
    queries: Queries,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None],
) -> tuple[dict[str, Any], np.ndarray]:
    """Re-predict every original selection candidate; never refit the frozen baseline."""
    reference = inputs / "reference"
    for name in REFERENCE_FILES:
        if sha256_file(reference / name) != config["reference_files"][name]:
            raise ValueError("baseline reference checksum differs")
    original = {
        role: json.loads((inputs / f"{role}_cache/manifest.json").read_text())
        for role in ("fit", "selection")
    }
    if any(m["status"] != "passed" or m["role"] != role for role, m in original.items()):
        raise ValueError("baseline replay requires certified cache roles")
    names = config["baseline_features"]
    models = []
    for objective in OBJECTIVES:
        state = json.loads((reference / objective / "manifest.json").read_text())
        contract = json.loads((reference / objective / "contract.json").read_text())
        lineage = contract["lineage"]
        if (
            not state["complete"]
            or state["features"] != names
            or contract["features"] != names
            or state["input_id"] != canonical_json_sha256(contract)
            or state["model_sha256"] != config["reference_files"][f"{objective}/model.txt"]
            or contract["config"] != config["training"]
            or contract["lightgbm"] != lgb.__version__
            or contract["code_sha256"] != sha256_file(Path(__file__).with_name("training.py"))
            or lineage["arm"] != "baseline"
            or lineage["objective"] != objective
            or any(
                lineage[f"{role}_role"] != role
                or lineage[f"{role}_cache_id"] != original[role]["input_id"]
                for role in original
            )
        ):
            raise ValueError("baseline reference training lineage differs")
        model = lgb.Booster(model_file=str(reference / objective / "model.txt"))
        if model.feature_name() != names or model.current_iteration() != state["best_iteration"]:
            raise ValueError("baseline native model schema or iteration differs")
        models.append(model)
    records, fusion_parts, count = [], [], 0
    manifest = original["selection"]
    with Heartbeat(logger, stage="baseline_prediction_replay", interval_seconds=15):
        for part in manifest["parts"]:
            path = inputs / "selection_cache" / part
            if Path(part).name != part or sha256_file(path) != manifest["files"][part]:
                raise ValueError("baseline candidate checksum differs")
            frame = pl.read_parquet(path, columns=[*names, *MODEL_METADATA])
            sessions, aids = frame["session"].to_numpy(), frame["aid"].to_numpy()
            ids, starts, groups = np.unique(sessions, return_index=True, return_counts=True)
            if (np.diff(sessions) < 0).any():
                raise ValueError("baseline candidate groups must be sorted")
            x = frame.select(names).to_numpy()
            target = frame.select([f"target_{o}" for o in OBJECTIVES]).to_numpy()
            hits = np.column_stack(
                [
                    ranked_hits(
                        np.asarray(model.predict(x, num_threads=config["training"]["threads"])),
                        aids,
                        target[:, j],
                        groups,
                    )
                    for j, model in enumerate(models)
                ]
            )
            coverage = np.minimum(20, np.add.reduceat(target, starts, axis=0))
            fusion_parts.append(
                np.column_stack(
                    [
                        ranked_hits(frame[f"baseline_{o}"].to_numpy(), aids, target[:, j], groups)
                        for j, o in enumerate(OBJECTIVES)
                    ]
                )
            )
            records.append(
                pl.DataFrame(
                    {
                        "session": ids,
                        **{f"hits_{o}": hits[:, j] for j, o in enumerate(OBJECTIVES)},
                        **{f"coverage_{o}": coverage[:, j] for j, o in enumerate(OBJECTIVES)},
                    }
                )
            )
            count += frame.height
    replay = pl.concat(records)
    if count != manifest["rows"] or not np.array_equal(
        replay["session"].to_numpy(), queries.session
    ):
        raise ValueError("baseline replay dropped or duplicated selection rows")
    if not replay.equals(pl.read_parquet(reference / "selection_statistics.parquet")):
        raise ValueError("baseline predictions do not reproduce every reference session statistic")
    hits = replay.select([f"hits_{o}" for o in OBJECTIVES]).to_numpy()
    summary = json.loads((reference / "selection_metrics.json").read_text())
    recomputed = {
        **official_score(hits, queries.denominators),
        "candidate_ceiling": official_score(
            replay.select([f"coverage_{o}" for o in OBJECTIVES]).to_numpy(), queries.denominators
        ),
        "fixed_fusion": official_score(np.vstack(fusion_parts), queries.denominators),
    }
    if (
        any(summary[key] != value for key, value in recomputed.items())
        or abs(summary["weighted_recall_at_20"] - config["expected_baseline_score"]) > 1e-12
    ):
        raise ValueError("baseline metrics do not reproduce the certified score")
    output.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, destination)
        publish(destination)
    summary["reused_reference"] = True
    summary["prediction_replay"] = (
        "all original selection candidate rows; identical per-session statistics"
    )
    atomic_json(output / "selection_metrics.json", summary)
    publish(output / "selection_metrics.json")
    return summary, hits


def fit_sample(directory: Path, rows: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Evenly sample fitting rows across every part without materializing a full matrix."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["role"] != "fit" or rows < 2:
        raise ValueError("quality screening accepts fitting rows only")
    names = tuple(manifest["features"])
    indices = np.linspace(0, manifest["rows"] - 1, min(rows, manifest["rows"]), dtype=np.int64)
    blocks, offset = [], 0
    for part in manifest["parts"]:
        path = directory / part
        if sha256_file(path) != manifest["files"][part]:
            raise ValueError("fitting sample checksum differs")
        frame = pl.read_parquet(path, columns=list(names))
        selected = indices[(indices >= offset) & (indices < offset + frame.height)] - offset
        if selected.size:
            blocks.append(frame[selected.tolist()].to_numpy())
        offset += frame.height
    if offset != manifest["rows"]:
        raise ValueError("fitting sample row count differs")
    return np.vstack(blocks), names


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
    if config["arms"] != {k: list(v) for k, v in ARM_FAMILIES.items()}:
        raise ValueError("domain comparisons differ from the preregistered arms")
    queries = {role: Queries(inputs / "corpus", role) for role in ("fit", "selection")}
    base_names = tuple(config["baseline_features"])
    graph_hashes = {f: sha256_file(inputs / f"graphs/{f}/manifest.json") for f in FAMILIES}
    contract = {
        "config": config,
        "graphs": graph_hashes,
        "corpus_id": queries["fit"].manifest["input_id"],
        "source_manifests": {
            role: sha256_file(inputs / f"{role}_cache/manifest.json") for role in queries
        },
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "domain_feature_study.py",
                "domain_feature_cache.py",
                "domain_features.py",
                "graph_signals.py",
                "features.py",
                "retrieval_study.py",
                "representation_study.py",
                "training.py",
                "metrics.py",
                "study.py",
                "dataset.py",
            )
        },
        "candidate_policy": "unchanged baseline rows, targets, features and negative sampling",
        "scope": "exploratory chronological selection; no evaluation or Kaggle promotion",
        "selection_exposure": "previously used in baseline, representation and graph studies",
    }
    path = output / "study_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("domain study contract differs")
    atomic_json(path, contract)
    publish(path)
    identity = canonical_json_sha256(contract)
    baseline, baseline_hits = replay_baseline(
        inputs,
        output / "models/baseline",
        queries["selection"],
        config,
        logger=logger,
        publish=publish,
    )
    summaries = {"baseline": baseline}
    statistics = {"baseline": baseline_hits}
    with Heartbeat(logger, stage="load_normalized_graphs", interval_seconds=15):
        graphs = NormalizedGraphSignals(
            {f: GraphSignals(inputs / f"graphs/{f}", f) for f in FAMILIES}
        )
    for role in queries:
        augment_cache(
            inputs / f"{role}_cache",
            output / f"{role}_cache",
            queries[role],
            graphs,
            graph_hashes=graph_hashes,
            base_names=base_names,
            logger=logger,
            publish=publish,
        )
    del graphs
    with Heartbeat(logger, stage="domain_fit_only_screening", interval_seconds=15):
        sample, all_names = fit_sample(output / "fit_cache", int(config["screening_rows"]))
        screens = {}
        for arm, families in ARM_FAMILIES.items():
            names = (*base_names, *(n for f in families for n in feature_names(f)))
            screens[arm] = screen_features(
                sample[:, [all_names.index(n) for n in names]],
                names,
                base_names,
                rows=sample.shape[0],
            )
    del sample
    atomic_json(output / "screening.json", screens)
    publish(output / "screening.json")
    for arm in ARM_FAMILIES:
        logger.info("domain_feature_arm_start", extra={"stage": arm})
        summary, hits = fit_arm(
            output,
            output / "models" / arm,
            queries,
            tuple(screens[arm]["features"]),
            config["training"],
            {"study_id": identity, "arm": arm},
            logger=logger,
            publish=publish,
        )
        summary["paired_selection_comparison"] = paired_bootstrap(
            hits, baseline_hits, queries["selection"].denominators, seed=config["training"]["seed"]
        )
        if arm == "normalized_graph":
            summary["paired_raw_graph_comparison"] = paired_bootstrap(
                hits,
                statistics["raw_graph"],
                queries["selection"].denominators,
                seed=config["training"]["seed"],
            )
        if summary["candidate_ceiling"] != baseline["candidate_ceiling"]:
            raise ValueError("domain experiment changed candidate coverage")
        summaries[arm], statistics[arm] = summary, hits
        result = {
            "status": "passed" if arm == "combined" else "running",
            "study_id": identity,
            "scope": contract["scope"],
            "arms": summaries,
            "evaluation_access": False,
            "kaggle_promotion": False,
        }
        atomic_json(output / "results.json", result)
        publish(output / "results.json")
    return result
