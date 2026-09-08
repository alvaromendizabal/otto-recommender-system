"""Post-selection explanations and matched feature-cost measurements."""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import OBJECTIVES, Queries, session_hash
from otto_recsys.research.evaluation import verify_seal
from otto_recsys.research.features import FeatureEngine, feature_catalog
from otto_recsys.research.metrics import official_score, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat


def explain(
    root: Path,
    *,
    seed: int,
    threads: int,
    logger: logging.Logger,
    sample_sessions: int = 1000,
    shap_rows: int = 4096,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Interpret frozen models on selection data; never revise the evaluation seal."""
    seal = verify_seal(root)
    if min(sample_sessions, shap_rows, threads) < 1:
        raise ValueError("explanation sample and thread settings must be positive")
    evaluated = json.loads((root / "evaluation/report.json").read_text())
    if evaluated["status"] != "passed" or evaluated["seal_id"] != seal["seal_id"]:
        raise ValueError("explanations require completed evaluation of the frozen model choice")
    output = root / "interpretation"
    output.mkdir(exist_ok=True)
    cache = json.loads((root / "selection_cache/manifest.json").read_text())
    contract = {
        "seal_id": seal["seal_id"],
        "selection_cache_id": cache["input_id"],
        "seed": seed,
        "sample_sessions": sample_sessions,
        "shap_rows": shap_rows,
        "code_sha256": sha256_file(Path(__file__)),
        "role": "post-selection interpretation; no model tuning or evaluation-label access",
    }
    input_id = canonical_json_sha256(contract)
    with workspace_lock(output):
        contract_path = output / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("explanation workspace has a different frozen-model contract")
        atomic_json(contract_path, contract)
        report_path = output / "report.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if previous["input_id"] == input_id and all(
                (output / n).is_file() and sha256_file(output / n) == h
                for n, h in previous["files"].items()
            ):
                return previous
        started = time.perf_counter()
        queries = Queries(root / "corpus", "selection")
        chosen_indices = np.argsort(session_hash(queries.session, seed + 903))[:sample_sessions]
        sessions = np.sort(queries.session[chosen_indices])
        denominator = queries.denominators[np.searchsorted(queries.session, sessions)]
        names = tuple(cache["features"])
        pieces = []
        with Heartbeat(logger, stage="interpretation_inputs", interval_seconds=15):
            for name in cache["parts"]:
                path = root / "selection_cache" / name
                if sha256_file(path) != cache["files"][name]:
                    raise ValueError("explanation candidate cache failed checksum validation")
                frame = pl.read_parquet(path)
                subset = frame.filter(pl.col("session").is_in(sessions.tolist()))
                if subset.height:
                    pieces.append(subset)
        frame = pl.concat(pieces).sort("session", "candidate_position")
        ids, groups = np.unique(frame["session"].to_numpy(), return_counts=True)
        if not np.array_equal(ids, sessions) or not np.all(groups == groups[0]):
            raise ValueError("group-block permutation requires complete equal-budget queries")
        x = frame.select(names).to_numpy().astype(np.float32)
        target = frame.select([f"target_{o}" for o in OBJECTIVES]).to_numpy()
        aids = frame["aid"].to_numpy()
        entries = [seal["models"][o] for o in OBJECTIVES]
        models = [lgb.Booster(model_file=str(root / m["path"])) for m in entries]
        if any(
            model.feature_name() != entry["features"]
            for model, entry in zip(models, entries, strict=True)
        ):
            raise ValueError("explanation model schema differs from the frozen feature order")
        columns = [[names.index(n) for n in m["features"]] for m in entries]
        family = {f.name: f.family for f in feature_catalog()}

        def hits_for(matrix: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    ranked_hits(
                        np.asarray(model.predict(matrix[:, index], num_threads=threads)),
                        aids,
                        target[:, j],
                        groups,
                    )
                    for j, (model, index) in enumerate(zip(models, columns, strict=True))
                ]
            )

        with Heartbeat(logger, stage="group_permutation", interval_seconds=15):
            original = official_score(hits_for(x), denominator)
            rng = np.random.default_rng(seed)
            permutations: dict[str, Any] = {}
            for group in sorted(set(family[n] for m in entries for n in m["features"])):
                positions = [i for i, n in enumerate(names) if family[n] == group]
                differences = []
                for _ in range(2):
                    altered = x.copy()
                    cube = altered.reshape(sessions.size, int(groups[0]), len(names))
                    replacement = x.reshape(sessions.size, int(groups[0]), len(names))[
                        rng.permutation(sessions.size)
                    ]
                    cube[:, :, positions] = replacement[:, :, positions]
                    score = official_score(hits_for(altered), denominator)
                    differences.append(
                        original["weighted_recall_at_20"] - score["weighted_recall_at_20"]
                    )
                permutations[group] = {
                    "mean_weighted_recall_drop": float(np.mean(differences)),
                    "repeat_drops": differences,
                }
        rng = np.random.default_rng(seed + 917)
        background = x[rng.choice(x.shape[0], min(shap_rows, x.shape[0]), replace=False)]
        importance = []
        additivity = {}
        with Heartbeat(logger, stage="native_tree_shap", interval_seconds=15):
            for objective, entry, model, index in zip(
                OBJECTIVES, entries, models, columns, strict=True
            ):
                native = np.asarray(
                    model.predict(background[:, index], pred_contrib=True, num_threads=threads)
                )
                prediction = np.asarray(model.predict(background[:, index], num_threads=threads))
                error = float(np.max(np.abs(native.sum(axis=1) - prediction)))
                if error > 1e-5:
                    raise ValueError("native SHAP contributions do not sum to the model score")
                additivity[objective] = error
                gain = model.feature_importance(importance_type="gain")
                for j, name in enumerate(entry["features"]):
                    importance.append(
                        {
                            "objective": objective,
                            "feature": name,
                            "family": family[name],
                            "gain": float(gain[j]),
                            "mean_absolute_shap": float(np.abs(native[:, j]).mean()),
                            "mean_signed_shap": float(native[:, j].mean()),
                        }
                    )
        importance_path = output / "feature_importance.parquet"
        pl.DataFrame(importance).write_parquet(importance_path, compression="zstd")
        engine = FeatureEngine(root / "retrieval")
        selected_names = tuple(dict.fromkeys(n for m in entries for n in m["features"]))
        modes = {
            "broad_catalog": tuple(f.name for f in feature_catalog()),
            "screened": names,
            "selected_models": selected_names,
        }
        benchmark: dict[str, Any] = {}
        with Heartbeat(logger, stage="feature_cost_benchmark", interval_seconds=15):
            for label, schema in modes.items():
                durations = []
                for repeat in range(3):
                    for index in chosen_indices[: min(32, chosen_indices.size)]:
                        prefix = queries.prefix(int(index))
                        clock = time.perf_counter()
                        candidates = engine.candidates(prefix, 400)
                        engine.transform(prefix, candidates, schema)
                        elapsed = time.perf_counter() - clock
                        if repeat:
                            durations.append(elapsed * 1000)
                benchmark[label] = {
                    "features": len(schema),
                    "measured_queries": len(durations),
                    "p50_ms": float(np.median(durations)),
                    "p95_ms": float(np.quantile(durations, 0.95)),
                }
        report = {
            "status": "passed",
            "input_id": input_id,
            "seal_id": seal["seal_id"],
            "selection_sessions": sessions.size,
            "candidate_rows": x.shape[0],
            "sample_score": original,
            "group_permutation": permutations,
            "permutation_method": "Two whole-query block shuffles per family "
            "preserve within-family relationships; "
            "candidate/feature dependencies can still be broken. Diagnostic, not a causal effect.",
            "shap_rows": background.shape[0],
            "shap_additivity_max_abs_error": additivity,
            "shap_method": "Native LightGBM TreeSHAP on deterministic selection candidate rows; "
            "contributions explain ranking scores, not calibrated probabilities.",
            "feature_benchmark": benchmark,
            "benchmark_scope": "Single-process warm candidate generation plus features, "
            "same selection prefixes and 400-item budget; no network or serving latency claim.",
            "hardware": {
                "system": platform.platform(),
                "logical_cpus": psutil.cpu_count(),
                "memory_gib": psutil.virtual_memory().total / 2**30,
                "model_threads": threads,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "files": {importance_path.name: sha256_file(importance_path)},
            "scope": contract["role"],
        }
        atomic_json(report_path, report)
        if publish:
            publish(contract_path)
            publish(importance_path)
            publish(report_path)
        return report
