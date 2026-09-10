"""Recompute graph-study metrics from complete-session statistics and pinned inputs.

This is an aggregate/lineage audit. It does not independently reconstruct raw graph
edges, candidate feature values, or native-model predictions on all candidate rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file

OBJECTIVES = ("clicks", "carts", "orders")
ARMS = ("baseline", "symmetric", "forward", "both")
WEIGHTS = np.array([0.1, 0.3, 0.6])


def check_equal(actual: Any, expected: Any, description: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"graph study audit mismatch: {description}")


def check_metrics(hits: np.ndarray, den: np.ndarray, report: dict[str, Any]) -> float:
    if (
        hits.shape != den.shape or (hits < 0).any() or (hits > den).any()
        or not np.equal(hits, np.floor(hits)).all() or (den.sum(axis=0) == 0).any()
    ):
        raise ValueError("graph study statistics contain invalid hit counts")
    recall = hits.sum(axis=0) / den.sum(axis=0)
    score = float(recall @ WEIGHTS)
    check_equal(score, report["weighted_recall_at_20"], "weighted Recall@20")
    for j, objective in enumerate(OBJECTIVES):
        measured = report["objectives"][objective]
        check_equal(recall[j], measured["recall_at_20"], objective)
        check_equal(hits[:, j].sum(), measured["hits"], f"{objective} numerator")
        check_equal(den[:, j].sum(), measured["denominator"], f"{objective} denominator")
    return score


def audit(directory: Path) -> dict[str, Any]:
    launch = json.loads((directory / "launch.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    contract = json.loads((directory / "study_contract.json").read_text())
    if (
        launch["task"] != "graph_features" or results["status"] != "passed"
        or results["evaluation_access"] is not False or results["kaggle_promotion"] is not False
        or set(results["arms"]) != set(ARMS)
    ):
        raise ValueError("audit requires a complete four-arm development study")
    graph_hashes = {
        family: next(row["sha256"] for row in inventory if row["path"] == "manifest.json")
        for family, inventory in launch["graphs"].items()
    }
    if (
        canonical_json_sha256(contract) != results["study_id"]
        or contract["config"] != launch["study"] or contract["graphs"] != graph_hashes
    ):
        raise ValueError("study contract differs from the managed launch or result identity")
    expected = next(row for row in launch["corpus"] if row["key"].endswith("/queries.parquet"))
    ledger_path = directory / "queries.parquet"
    if sha256_file(ledger_path) != expected["sha256"]:
        raise ValueError("query ledger differs from the pinned managed input")
    ledger = pl.read_parquet(ledger_path).filter(pl.col("split_role") == "selection")
    rows = [ledger.filter(pl.col("objective") == o).sort("session") for o in OBJECTIVES]
    sessions = rows[0]["session"].to_numpy()
    if not sessions.size or np.unique(sessions).size != sessions.size:
        raise ValueError("selection ledger requires unique nonempty sessions")
    if any(not np.array_equal(row["session"].to_numpy(), sessions) for row in rows):
        raise ValueError("selection objectives have different session ledgers")
    den = np.column_stack([row["recall_denominator"].to_numpy() for row in rows])
    truth = np.column_stack([row["true_items"].to_numpy() for row in rows])
    if not np.array_equal(den, np.minimum(20, truth)) or (den < 0).any():
        raise ValueError("selection denominators must include all capped future targets")
    base_names = launch["study"]["baseline_features"]
    hits_by_arm: dict[str, np.ndarray] = {}
    coverage_by_arm: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    verified_files = {
        "launch.json", "results.json", "queries.parquet", "screening.json", "study_contract.json",
    }
    for arm in ARMS:
        summary = results["arms"][arm]
        names = summary["features"]
        if names[:len(base_names)] != base_names or len(names) != len(set(names)):
            raise ValueError("arm changed the ordered baseline feature schema")
        if arm == "baseline" and names != base_names:
            raise ValueError("baseline includes additional feature columns")
        if arm in ("symmetric", "forward") and any(
            not name.startswith(f"wide_{arm}_") for name in names[len(base_names):]
        ):
            raise ValueError("single-family arm contains features from another family")
        relative = f"models/{arm}/selection_statistics.parquet"
        frame = pl.read_parquet(directory / relative)
        verified_files.add(relative)
        if not np.array_equal(frame["session"].to_numpy(), sessions):
            raise ValueError("arm dropped, reordered or duplicated selection sessions")
        hits = frame.select([f"hits_{o}" for o in OBJECTIVES]).to_numpy()
        coverage = frame.select([f"coverage_{o}" for o in OBJECTIVES]).to_numpy()
        score = check_metrics(hits, den, summary)
        check_metrics(coverage, den, summary["candidate_ceiling"])
        if (hits > coverage).any():
            raise ValueError("ranker hits exceed per-session candidate coverage")
        if coverage_by_arm and not np.array_equal(coverage, coverage_by_arm["baseline"]):
            raise ValueError("feature-only arms changed per-session candidate coverage")
        if arm == "baseline":
            check_equal(score, launch["study"]["expected_baseline_score"], "certified baseline")
        model_evidence = {}
        for objective in OBJECTIVES:
            model = f"models/{arm}/{objective}/model.txt"
            manifest = f"models/{arm}/{objective}/manifest.json"
            model_contract_path = f"models/{arm}/{objective}/contract.json"
            state = json.loads((directory / manifest).read_text())
            model_contract = json.loads((directory / model_contract_path).read_text())
            digest = sha256_file(directory / model)
            if (
                digest != summary["models"][objective]["sha256"]
                or digest != state["model_sha256"] or not state["complete"]
                or state["features"] != names
                or state["best_iteration"] != summary["models"][objective]["best_iteration"]
                or state["input_id"] != canonical_json_sha256(model_contract)
                or model_contract["lineage"]["study_id"] != results["study_id"]
                or model_contract["lineage"]["arm"] != arm
                or model_contract["lineage"]["objective"] != objective
                or model_contract["config"] != launch["study"]["training"]
            ):
                raise ValueError("model bytes or schema differ from the reported arm")
            native = lgb.Booster(model_file=str(directory / model))
            if (
                native.feature_name() != names
                or native.current_iteration() != state["best_iteration"]
            ):
                raise ValueError("native model feature schema or tree count differs")
            native_gains = dict(
                zip(names, native.feature_importance(importance_type="gain"), strict=True)
            )
            total = sum(native_gains.values())
            extra = sum(v for n, v in native_gains.items() if n.startswith("wide_"))
            model_evidence[objective] = {
                "model_sha256": digest,
                "best_iteration": state["best_iteration"],
                "new_feature_gain_share": float(extra / total) if total else 0.0,
                "gain_source": "recomputed from the verified native model file",
                "top_new_features": sorted(
                    ((n, float(v)) for n, v in native_gains.items() if n.startswith("wide_")),
                    key=lambda row: (-row[1], row[0]),
                )[:10],
            }
            verified_files.update((model, manifest, model_contract_path))
        hits_by_arm[arm], coverage_by_arm[arm] = hits, coverage
        summaries[arm] = {"weighted_recall_at_20": score, "features": len(names),
                          "models": model_evidence}
    for arm in ARMS[1:]:
        reported = results["arms"][arm]["paired_selection_comparison"]
        if reported["seed"] != launch["study"]["training"]["seed"]:
            raise ValueError("paired comparison seed differs from the experiment")
        repeats = reported["requested_replicates"]
        if repeats < 100:
            raise ValueError("paired comparison requires at least 100 replicates")
        rng = np.random.default_rng(reported["seed"])
        delta = hits_by_arm[arm] - hits_by_arm["baseline"]
        gains = []
        for _ in range(repeats):
            sample = rng.integers(0, sessions.size, size=sessions.size)
            denominator = den[sample].sum(axis=0)
            if (denominator == 0).any():
                continue
            gains.append(delta[sample].sum(axis=0) / denominator)
        if len(gains) < 0.95 * repeats:
            raise ValueError("too many bootstrap samples lack an objective denominator")
        distribution = np.asarray(gains)
        point = delta.sum(axis=0) / den.sum(axis=0)
        interval = np.quantile(distribution @ WEIGHTS, [0.025, 0.975])
        check_equal(point @ WEIGHTS, reported["absolute_gain"], "paired gain")
        check_equal(interval, reported["gain_interval"], "paired interval")
        check_equal(len(gains), reported["valid_replicates"], "valid bootstrap replicates")
        summaries[arm]["paired_gain"] = {
            "weighted": float(point @ WEIGHTS), "weighted_ci95": interval.tolist(),
            "objectives": {
                o: {"gain": float(point[j]),
                    "ci95": np.quantile(distribution[:, j], [0.025, 0.975]).tolist()}
                for j, o in enumerate(OBJECTIVES)
            },
        }
    screen = json.loads((directory / "screening.json").read_text())
    screening_findings = [
        {"feature": row["name"], "redundant_with": row["redundant_with"], "arm": arm}
        for row in screen["decisions"]
        for arm in ("symmetric", "forward")
        if row["name"].startswith(f"wide_{arm}_")
        and row["decision"] == "redundant_on_fit_sample"
        and row["redundant_with"] not in results["arms"][arm]["features"]
    ]
    return {
        "status": "passed", "study_id": results["study_id"],
        "scope": "Independent aggregate metrics and model-byte lineage; development selection only",
        "source_commit": launch["source_commit"], "sessions_per_arm": int(sessions.size),
        "arms_verified": len(ARMS), "models_verified": len(ARMS) * len(OBJECTIVES),
        "candidate_coverage_equal_per_session_and_objective": True,
        "per_task_intervals": (
            "Paired whole-session percentile 95%; descriptive, unadjusted for "
            "selection or multiple comparisons"
        ),
        "arms": summaries, "cross_family_screening_findings": screening_findings,
        "limitations": [
            "No independent reconstruction of raw graph edges or all candidate feature values.",
            "No replay of model predictions on the full candidate matrix.",
            "Gain importance is descriptive and does not establish individual causal utility.",
            "Repeated selection-cohort use requires separate confirmation before promotion.",
        ],
        "files": {name: sha256_file(directory / name) for name in sorted(verified_files)},
        "evaluation_access": False, "kaggle_promotion": False,
    }
