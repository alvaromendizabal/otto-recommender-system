"""Add wider affinities without replacing baseline candidates or features."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import Queries
from otto_recsys.research.graph_feature_cache import augment_cache
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals
from otto_recsys.research.metrics import paired_bootstrap
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.representation_study import screen_features
from otto_recsys.research.retrieval_study import fit_arm
from otto_recsys.research.study import load_model_cache
from otto_recsys.runtime import Heartbeat


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
                "graph_feature_study.py",
                "graph_feature_cache.py",
                "graph_signals.py",
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
    }
    path = output / "study_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("graph feature study contract differs")
    atomic_json(path, contract)
    publish(path)
    identity = canonical_json_sha256(contract)
    summary, baseline_hits = fit_arm(
        inputs,
        output / "models/baseline",
        queries,
        base_names,
        config["training"],
        {"study_id": identity, "arm": "baseline"},
        logger=logger,
        publish=publish,
    )
    if abs(summary["weighted_recall_at_20"] - config["expected_baseline_score"]) > 1e-12:
        raise ValueError("baseline does not reproduce the certified selection score")
    summaries = {"baseline": summary}
    with Heartbeat(logger, stage="load_historical_graphs", interval_seconds=15):
        graphs = {f: GraphSignals(inputs / f"graphs/{f}", f) for f in FAMILIES}
    for graph in graphs.values():
        if graph.cutoff != queries["fit"].manifest["protocol"]["history_end"]:
            raise ValueError("graph cutoff differs from the query corpus")
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
    with Heartbeat(logger, stage="graph_feature_screening", interval_seconds=15):
        cache = load_model_cache(output / "fit_cache", output / "screening.f32")
        screen = screen_features(
            cache["x"], cache["names"], base_names, rows=int(config["screening_rows"])
        )
        del cache
        (output / "screening.f32").unlink()
    atomic_json(output / "screening.json", screen)
    publish(output / "screening.json")
    for arm in (*FAMILIES, "both"):
        names = tuple(
            n
            for n in screen["features"]
            if n in base_names or arm == "both" or n.startswith(f"wide_{arm}_")
        )
        logger.info("graph_feature_arm_start", extra={"stage": arm})
        summary, hits = fit_arm(
            output,
            output / "models" / arm,
            queries,
            names,
            config["training"],
            {"study_id": identity, "arm": arm},
            logger=logger,
            publish=publish,
        )
        summary["paired_selection_comparison"] = paired_bootstrap(
            hits,
            baseline_hits,
            queries["selection"].denominators,
            seed=int(config["training"]["seed"]),
        )
        if summary["candidate_ceiling"] != summaries["baseline"]["candidate_ceiling"]:
            raise ValueError("feature-only experiment changed candidate coverage")
        summaries[arm] = summary
        result = {
            "status": "passed" if arm == "both" else "running",
            "study_id": identity,
            "scope": contract["scope"],
            "arms": summaries,
            "evaluation_access": False,
            "kaggle_promotion": False,
        }
        atomic_json(output / "results.json", result)
        publish(output / "results.json")
    return result
