"""Fitting-only candidate-ceiling study for nested 400/800/1200 graph frontiers."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.research.candidate_frontier import CandidateFrontier, candidate_ceiling
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals
from otto_recsys.research.metrics import official_score
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat


def run(
    *,
    repo: Path,
    corpus: Path,
    retrieval: Path,
    graphs: Path,
    preflight: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads((repo / "configs/candidate_frontier_pilot.json").read_text())
    queries = Queries(corpus, "fit")
    order = queries.indices(int(config["cohort_seed"]))[: int(config["sessions"])]
    ids = queries.session[order]
    if ids.size != int(config["sessions"]) or np.unique(ids).size != ids.size:
        raise ValueError("candidate-frontier cohort must contain unique complete fitting sessions")
    smoke = (
        pl.read_parquet(preflight / "early_fit_queries.parquet", columns=["session"])
        .unique()
        .sort("session")["session"]
        .to_numpy()
    )
    if ids.size >= 256 and not np.array_equal(np.sort(ids[:256]), smoke):
        raise ValueError("candidate-frontier cohort is not nested over the verified 256-session smoke")

    engine = FeatureEngine(retrieval)
    wide = {family: GraphSignals(graphs / family, family) for family in FAMILIES}
    frontier = CandidateFrontier(
        engine,
        wide,
        bridge_per_source=int(config["twohop_bridge_per_source"]),
        rrf_k=float(config["rrf_k"]),
    )
    arms = {
        "baseline_400": (400, 1),
        "wide_onehop_800": (800, 1),
        "wide_onehop_1200": (1200, 1),
        "wide_twohop_800": (800, 2),
        "wide_twohop_1200": (1200, 2),
    }
    hits = {name: np.zeros((ids.size, 3), dtype=np.int32) for name in arms}
    counts = {name: np.zeros(ids.size, dtype=np.int32) for name in arms}
    rows: list[dict[str, Any]] = []
    progress: dict[str, Any] = {"sessions": 0, "total": int(ids.size)}
    with Heartbeat(
        __import__("logging").getLogger("candidate_frontier"),
        stage="candidate_frontier_ceiling",
        interval_seconds=15,
        progress_provider=progress.copy,
    ):
        for slot, index in enumerate(order):
            prefix = queries.prefix(int(index))
            session = int(prefix.session)
            labels = tuple(
                queries.labels.get((session, objective), np.empty(0, dtype=np.int64))
                for objective in OBJECTIVES
            )
            denominator = queries.denominators[int(index)]
            baseline_ids: np.ndarray | None = None
            row: dict[str, Any] = {"session": session}
            for name, (budget, hops) in arms.items():
                candidates = frontier.candidates(prefix, budget=budget, hops=hops)
                counts[name][slot] = candidates.aid.size
                hits[name][slot] = candidate_ceiling(candidates.aid, labels, denominator)
                if name == "baseline_400":
                    baseline_ids = candidates.aid
                elif baseline_ids is None or not np.isin(baseline_ids, candidates.aid).all():
                    raise ValueError("expanded candidate frontier lost a baseline candidate")
                row[f"{name}_candidates"] = int(candidates.aid.size)
                for j, objective in enumerate(OBJECTIVES):
                    row[f"{name}_{objective}_hits"] = int(hits[name][slot, j])
            for j, objective in enumerate(OBJECTIVES):
                row[f"denominator_{objective}"] = int(denominator[j])
            rows.append(row)
            progress["sessions"] = slot + 1

    denominators = queries.denominators[order]
    summaries: dict[str, Any] = {}
    for name in arms:
        score = official_score(hits[name], denominators)
        summaries[name] = {
            **score,
            "average_candidates": float(counts[name].mean()),
            "minimum_candidates": int(counts[name].min()),
            "maximum_candidates": int(counts[name].max()),
        }
    baseline = summaries["baseline_400"]
    for name, summary in summaries.items():
        summary["weighted_gain_vs_baseline"] = (
            float(summary["weighted_recall_at_20"])
            - float(baseline["weighted_recall_at_20"])
        )
    best = max(
        (name for name in arms if name != "baseline_400"),
        key=lambda name: summaries[name]["weighted_recall_at_20"],
    )
    best_gain = float(summaries[best]["weighted_gain_vs_baseline"])
    objective_nonnegative = all(
        summaries[best]["objectives"][objective]["recall_at_20"]
        >= baseline["objectives"][objective]["recall_at_20"]
        for objective in OBJECTIVES
    )
    statistics = pl.DataFrame(rows).sort("session")
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "frontier_statistics.parquet"
    statistics.write_parquet(stats_path, compression="zstd")
    result = {
        "status": "CANDIDATE_FRONTIER_PILOT_PASSED",
        "scope": "Fitting-only metric after label-blind candidate generation; no model fits.",
        "sessions": int(ids.size),
        "cohort_seed": int(config["cohort_seed"]),
        "history_end": int(engine.cutoff),
        "selection_access": False,
        "model_fits": 0,
        "arms": summaries,
        "best_expanded_arm": best,
        "decision": {
            "advance_to_1024_sessions": bool(
                best_gain >= float(config["minimum_weighted_ceiling_gain"])
                and objective_nonnegative
            ),
            "minimum_weighted_ceiling_gain": float(config["minimum_weighted_ceiling_gain"]),
            "best_weighted_gain": best_gain,
            "all_objectives_nonnegative": objective_nonnegative,
        },
        "statistics_rows": statistics.height,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "next_gate": (
            "If the smoke gain passes, repeat on the frozen 1,024 fitting sessions before "
            "materializing expanded source features or fitting a ranker. If it fails, add "
            "action-pair graph variants/neural retrieval instead of tuning this frontier."
        ),
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        repo=args.repo,
        corpus=args.corpus,
        retrieval=args.retrieval,
        graphs=args.graphs,
        preflight=args.preflight,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
