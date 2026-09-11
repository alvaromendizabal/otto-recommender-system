"""Run a checkpointed 256-fitting-session candidate-coverage experiment; never fit models."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.research.candidate_frontier import ARMS, CandidateFrontier, candidate_ceiling
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals
from otto_recsys.research.metrics import official_score, paired_bootstrap
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(repo: Path, workspace: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    config = json.loads((repo / "configs/candidate_frontier_pilot.json").read_text())
    if (config["role"] != "fit" or config["sessions"] != 256
            or config["selection_access"] is not False
            or config["evaluation_access"] is not False):
        raise ValueError("Only the preregistered 256-session fitting smoke is permitted")
    root = workspace / "otto-artifacts/shared-feature-smoke/early-smoke-91d3dcd1"
    corpus = root / "inputs/corpus"
    retrieval, graphs = root / "inputs/retrieval", root / "output/graphs"
    scale = workspace / "otto-artifacts/shared-feature-scale/early-scale-29439ec2/output"
    smoke = json.loads((repo / "reports/research/early_shared_feature_smoke.json").read_text())
    scale_path = repo / "reports/research/early_shared_feature_scale.json"
    scale_report = json.loads(scale_path.read_text())
    first = scale_report["outputs"]["parts"][0]
    if first["sessions"] != 256 or first["rows"] != 102400:
        raise ValueError("Certified baseline partition must have 256 complete sessions")
    baseline_file = scale / first["part"]
    if sha(baseline_file) != first["sha256"]:
        raise ValueError("Certified baseline matrix checksum mismatch")
    baseline = pl.read_parquet(baseline_file, columns=[
        "session", "aid", "candidate_position", *[f"target_{o}" for o in OBJECTIVES]])
    baseline = baseline.sort("session", "candidate_position")
    queries = Queries(corpus, "fit")
    order = queries.indices(config["cohort_seed"])[:256]
    ids = queries.session[order]
    if not np.array_equal(np.sort(ids), baseline["session"].unique().sort().to_numpy()):
        raise ValueError("Frozen first-256 cohort differs from the certified baseline")
    denominator = queries.denominators[order]
    if denominator.sum(axis=0).tolist() != smoke["roles"]["fit"]["pooled_denominators"]:
        raise ValueError("Frozen cohort denominator differs")
    for directory in (retrieval, *(graphs / f for f in FAMILIES)):
        receipt = json.loads((directory / "manifest.json").read_text())
        if (receipt["status"] != "passed" or receipt["query_labels_used"] is not False
                or receipt["history_end"] != config["history_end_ms"]
                or receipt["observed_history_max_ts"] >= config["history_end_ms"]):
            raise ValueError("Historical graph certification mismatch")
        if (directory != retrieval
                and receipt["input_id"] != smoke["graphs"][directory.name]["input_id"]):
            raise ValueError("Wide graph identity differs from certified smoke")
    lineage = {
        "config": config, "sessions": ids.tolist(),
        "corpus": {n: sha(corpus / n) for n in (
            "manifest.json", "observed.parquet", "queries.parquet", "labels.parquet")},
        "baseline_matrix": sha(baseline_file),
        "retrieval_manifest": sha(retrieval / "manifest.json"),
        "graph_manifests": {f: sha(graphs / f / "manifest.json") for f in FAMILIES},
        "code": {n: sha(repo / n) for n in (
            "src/otto_recsys/research/candidate_frontier.py",
            "scripts/run_candidate_frontier_pilot.py")},
    }
    input_id = identity(lineage)
    output.mkdir(parents=True, exist_ok=True)
    contract = output / "contract.json"
    if contract.exists() and json.loads(contract.read_text()) != lineage:
        raise ValueError("Existing output contract differs; preserve previous evidence")
    atomic_json(contract, lineage)
    logger = logging.getLogger("frontier")
    progress: dict[str, Any] = {"phase": "load_certified_graphs", "done": 0, "total": 256}
    records = []
    reused = 0
    with Heartbeat(logger, stage="candidate_frontier", interval_seconds=15,
                   progress_provider=progress.copy):
        engine = FeatureEngine(retrieval)
        wide = {f: GraphSignals(graphs / f, f) for f in FAMILIES}
        frontier = CandidateFrontier(engine, wide,
                                    bridge_per_source=config["twohop_bridge_per_source"],
                                    rrf_k=config["rrf_k"],
                                    max_discovery_items=config["max_discovery_items"])
        progress["phase"] = "generate_and_replay"
        for slot, index in enumerate(order):
            session = int(ids[slot])
            path = output / "sessions" / f"{session}.json"
            prefix = queries.prefix(int(index))
            if path.exists():
                record = json.loads(path.read_text())
                if record["input_id"] != input_id or record["session"] != session:
                    raise ValueError("Checkpoint belongs to another experiment")
                pools = {name: np.asarray(record["candidates"][name], dtype=np.int64)
                         for name in ARMS}
                if identity(record["candidates"]) != record["candidate_digest"]:
                    raise ValueError("Checkpoint candidate digest mismatch")
                reused += 1
            else:
                tick = time.monotonic()
                first_pass = frontier.frontiers(prefix)
                second_pass = frontier.frontiers(prefix)
                pools = {name: first_pass[name].aid for name in ARMS}
                for name in ARMS:
                    if not np.array_equal(pools[name], second_pass[name].aid):
                        raise ValueError("Candidate replay mismatch")
                candidates = {name: values.tolist() for name, values in pools.items()}
                record = {"input_id": input_id, "session": session, "candidates": candidates,
                          "candidate_digest": identity(candidates),
                          "generation_and_replay_seconds": time.monotonic() - tick}
            original = baseline.filter(pl.col("session") == session)
            if original.height != 400 or not np.array_equal(
                    pools[ARMS[0]], original["aid"].to_numpy()):
                raise ValueError("Exact baseline candidates/order were not preserved")
            # Targets are read only after every candidate pool has been generated or restored.
            labels = tuple(queries.labels.get((session, o), np.array([], dtype=np.int64))
                           for o in OBJECTIVES)
            for j, objective in enumerate(OBJECTIVES):
                if not np.array_equal(np.isin(pools[ARMS[0]], labels[j]),
                                      original[f"target_{objective}"].to_numpy()):
                    raise ValueError("Baseline target identity differs")
            hits = {}
            for name in ARMS:
                if not np.array_equal(pools[name][:400], pools[ARMS[0]]):
                    raise ValueError("Expanded candidate pool lost its baseline")
                hits[name] = candidate_ceiling(pools[name], labels, denominator[slot]).tolist()
            record.update(hits=hits, denominator=denominator[slot].tolist())
            atomic_json(path, record)
            records.append(record)
            progress["done"] = slot + 1
            if (slot + 1) % 32 == 0:
                print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  **progress}), flush=True)
    arrays = {name: np.asarray([r["hits"][name] for r in records], dtype=np.int64)
              for name in ARMS}
    summaries = {}
    for name in ARMS:
        sizes = np.array([len(r["candidates"][name]) for r in records])
        summaries[name] = {**official_score(arrays[name], denominator),
                           "mean_candidates": float(sizes.mean()),
                           "min_candidates": int(sizes.min()),
                           "max_candidates": int(sizes.max())}
    base = summaries[ARMS[0]]["weighted_recall_at_20"]
    comparisons = {}
    for name in ARMS[1:]:
        if (arrays[name] < arrays[ARMS[0]]).any():
            raise ValueError("Nested candidate oracle declined")
        comparisons[name] = paired_bootstrap(arrays[name], arrays[ARMS[0]], denominator,
                                             seed=config["cohort_seed"], replicates=1000)
    gain = max(v["weighted_recall_at_20"] - base for n, v in summaries.items() if n != ARMS[0])
    statistics = [{"session": r["session"], "denominator": r["denominator"],
                   "hits": r["hits"]} for r in records]
    atomic_json(output / "statistics.json", statistics)
    result = {
        "status": "CANDIDATE_FRONTIER_PILOT_COMPLETED", "role": "fit", "sessions": 256,
        "selection_access": False, "evaluation_access": False, "model_fits": 0,
        "graph_rebuilds": 0, "feature_retention_decisions": 0, "input_id": input_id,
        "history_end_ms": config["history_end_ms"], "arms": summaries,
        "paired_descriptive_comparisons": comparisons,
        "denominators": denominator.sum(axis=0).tolist(),
        "decision": {"advance_unchanged_fitting_replication": gain >= 0.005,
                     "minimum_oracle_gain": 0.005, "best_oracle_gain": gain},
        "verified_candidate_replay": True, "verified_baseline_candidate_target_parity": True,
        "reused_sessions": reused, "new_sessions": 256 - reused,
        "statistics_sha256": sha(output / "statistics.json"),
        "elapsed_seconds": time.monotonic() - started,
        "generation_and_replay_seconds": sum(r["generation_and_replay_seconds"] for r in records),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "limitations": ["Candidate oracle ceiling, not achieved ranking or Kaggle score.",
                        "Small reused fitting cohort; only 27 order-denominator units.",
                        "No new ranker features, feature ablation or promotion in this smoke.",
                        "Budget growth guarantees nondecreasing oracle; ranking may still worsen."]}
    atomic_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/home/sagemaker-user"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    def deadline(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError("300-second work limit; completed session checkpoints preserved")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        run(args.repo, args.workspace, args.output)
    except Exception as error:
        atomic_json(args.output / "failure.json", {"status": "STOPPED_REVIEW_REQUIRED",
                                                   "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
