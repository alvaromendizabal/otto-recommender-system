"""Training-only feature quality checks, grouped pilots and redundancy removal."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.research.dataset import OBJECTIVES, cache_batches, session_hash
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat


def core_features() -> tuple[str, ...]:
    names = [
        f"source_{s}_{v}"
        for s in ("time", "cart", "order", "revisit", "popularity")
        for v in ("score", "rank", "share")
    ]
    names += [
        "source_count",
        "query_events",
        "query_unique",
        "query_duration_hours",
        "query_repeat_share",
        "query_carts_share",
        "query_orders_share",
    ]
    names += [f"repeat_all_n50_{v}" for v in ("count", "last_position", "age_hours")]
    names += [f"repeat_{a}_n50_count" for a in ("clicks", "carts", "orders")]
    return tuple(names)


def quality_screen(
    x: np.ndarray, target: np.ndarray, names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """All checks operate exclusively on the supplied fitting matrix."""
    if (
        x.shape[1] != len(names)
        or target.shape != (x.shape[0], 3)
        or not np.isin(target, [0, 1]).all()
    ):
        raise ValueError("screening features and three binary targets must align")
    families = {f.name: f.family for f in feature_catalog()}
    if not set(names).issubset(families) or len(set(names)) != len(names):
        raise ValueError("screening schema must contain unique catalog features only")
    records = []
    seen: dict[str, str] = {}
    y = target.astype(np.float64)
    y -= y.mean(axis=0)
    yscale = np.maximum(np.sqrt((y * y).sum(axis=0)), 1e-12)
    for j, name in enumerate(names):
        values = np.asarray(x[:, j], dtype=np.float32)
        record: dict[str, Any] = {
            "name": name,
            "family": families.get(name, "unknown"),
            "status": "eligible",
            "dtype": "float32",
            "missing": 0,
        }
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite feature rejected: {name}")
        distinct, counts = np.unique(values, return_counts=True)
        record.update(
            unique_values=int(distinct.size),
            mode_fraction=float(counts.max() / values.size),
            minimum=float(distinct[0]),
            maximum=float(distinct[-1]),
        )
        fingerprint = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        if distinct.size < 2:
            record["status"] = "constant"
        elif counts.max() / values.size >= 0.9999:
            record["status"] = "near_constant"
        elif fingerprint in seen:
            record["status"] = "duplicate"
            record["duplicate_of"] = seen[fingerprint]
        else:
            seen[fingerprint] = name
        centered = values.astype(np.float64) - values.mean(dtype=np.float64)
        denominator = max(float(np.sqrt(centered @ centered)), 1e-12)
        correlations = (centered @ y) / (denominator * yscale)
        record["target_correlations"] = dict(zip(OBJECTIVES, correlations.tolist(), strict=True))
        record["univariate_utility"] = float(np.abs(correlations) @ np.array([0.1, 0.3, 0.6]))
        if np.max(np.abs(correlations)) > 0.999999:
            record["status"] = "target_equivalent"
        records.append(record)
    return records


def screen_cache(
    cache: Path,
    output: Path,
    *,
    max_retained: int,
    seed: int,
    logger: logging.Logger,
    threads: int = 4,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache / "manifest.json").read_text())
    source_contract = json.loads((cache / "contract.json").read_text())
    if manifest["role"] != "fit" or source_contract["role"] != "fit":
        raise ValueError("feature selection accepts fitting sessions only")
    names = tuple(source_contract["features"])
    contract = {
        "cache_id": manifest["input_id"],
        "files": manifest["files"],
        "max_retained": max_retained,
        "seed": seed,
        "folds": 3,
        "pilot_rounds": 100,
        "correlation_threshold": 0.995,
        "code_sha256": sha256_file(Path(__file__)),
        "lightgbm": lgb.__version__,
    }
    input_id = canonical_json_sha256(contract)
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("screening output has a different training contract")
    atomic_json(path, contract)
    report_path = output / "selection.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["input_id"] == input_id:
            return report
    rows = int(manifest["rows"])
    x = np.memmap(
        output / "working_features.f32", mode="w+", dtype=np.float32, shape=(rows, len(names))
    )
    targets = np.empty((rows, 3), dtype=np.int8)
    sessions = np.empty(rows, dtype=np.int64)
    progress: dict[str, Any] = {"phase": "quality_screen", "rows": rows, "features": len(names)}
    start = time.perf_counter()
    with Heartbeat(
        logger, stage="feature_screening", interval_seconds=15, progress_provider=progress.copy
    ):
        position = 0
        for frame in cache_batches(cache, names):
            stop = position + frame.height
            x[position:stop] = frame.select(names).to_numpy()
            targets[position:stop] = frame.select([f"target_{o}" for o in OBJECTIVES]).to_numpy()
            sessions[position:stop] = frame["session"].to_numpy()
            position = stop
        if position != rows:
            raise ValueError("screening row count differs from its manifest")
        x.flush()
        quality_path = output / "quality.json"
        if quality_path.exists():
            quality = json.loads(quality_path.read_text())
            if quality["input_id"] != input_id:
                raise ValueError("screening quality checkpoint is stale")
            records = quality["features"]
        else:
            records = quality_screen(x, targets, names)
            atomic_json(quality_path, {"input_id": input_id, "features": records})
        eligible = [j for j, r in enumerate(records) if r["status"] == "eligible"]
        # Pilots consider every eligible feature; no holdout-driven top-N filter.
        fold = (session_hash(sessions, seed) % 3).astype(np.int8)
        pilot_importance = np.zeros((3, 3, len(eligible)), dtype=np.float64)
        diagnostics = []
        for f in range(3):
            for objective, label in enumerate(OBJECTIVES):
                progress.update(phase="grouped_pilot", fold=f, objective=label)
                model_path = output / f"pilot-{f}-{label}.txt"
                diagnostic_path = output / f"pilot-{f}-{label}.json"
                if model_path.exists() and diagnostic_path.exists():
                    diagnostic = json.loads(diagnostic_path.read_text())
                    if diagnostic["input_id"] != input_id or diagnostic["sha256"] != sha256_file(
                        model_path
                    ):
                        raise ValueError("pilot model checkpoint failed verification")
                    booster = lgb.Booster(model_file=str(model_path))
                else:
                    fit = fold != f
                    valid = ~fit
                    train_x = np.asarray(x[np.ix_(fit, eligible)])
                    data = lgb.Dataset(
                        train_x,
                        label=targets[fit, objective],
                        feature_name=[names[j] for j in eligible],
                        free_raw_data=True,
                    )
                    booster = lgb.train(
                        {
                            "objective": "binary",
                            "metric": "None",
                            "verbosity": -1,
                            "num_threads": threads,
                            "num_leaves": 31,
                            "learning_rate": 0.08,
                            "min_data_in_leaf": 40,
                            "feature_fraction": 0.8,
                            "seed": seed + f,
                            "deterministic": True,
                            "force_col_wise": True,
                        },
                        data,
                        num_boost_round=100,
                    )
                    prediction = np.clip(
                        booster.predict(x[np.ix_(valid, eligible)], num_threads=threads),
                        1e-7,
                        1 - 1e-7,
                    )
                    truth = targets[valid, objective]
                    loss = float(
                        -np.mean(truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction))
                    )
                    temporary = model_path.with_suffix(".txt.tmp")
                    booster.save_model(str(temporary))
                    temporary.replace(model_path)
                    diagnostic = {
                        "input_id": input_id,
                        "fold": f,
                        "objective": label,
                        "fit_rows": int(fit.sum()),
                        "valid_rows": int(valid.sum()),
                        "validation_binary_logloss": loss,
                        "sha256": sha256_file(model_path),
                        "scope": "grouped fitting-only screening; sampled negatives",
                    }
                    atomic_json(diagnostic_path, diagnostic)
                    del data, train_x
                gain = booster.feature_importance(importance_type="gain")
                pilot_importance[f, objective] = gain / max(gain.sum(), 1e-12)
                diagnostics.append(diagnostic)
        importance = (pilot_importance.mean(axis=0) * np.array([0.1, 0.3, 0.6])[:, None]).sum(
            axis=0
        )
        stability = (pilot_importance.sum(axis=1) > 0).mean(axis=0)
        for j, utility, stable in zip(eligible, importance, stability, strict=True):
            records[j].update(
                model_utility=float(utility), positive_gain_fold_fraction=float(stable)
            )
        mandatory = set(core_features())
        ordering = sorted(
            eligible,
            key=lambda j: (names[j] not in mandatory, -records[j]["model_utility"], names[j]),
        )
        sample = np.random.default_rng(seed).choice(rows, size=min(rows, 10000), replace=False)
        z = np.asarray(x[np.ix_(sample, eligible)], dtype=np.float64)
        z -= z.mean(axis=0)
        z /= np.maximum(np.sqrt((z * z).sum(axis=0)), 1e-12)
        local = {j: k for k, j in enumerate(eligible)}
        retained: list[int] = []
        for j in ordering:
            record = records[j]
            if retained:
                correlations = np.abs(
                    z[:, local[j]] @ z[:, np.array([local[k] for k in retained], dtype=np.int64)]
                )
                maximum = int(np.argmax(correlations))
                if correlations[maximum] >= 0.995 and names[j] not in mandatory:
                    record.update(
                        status="redundant",
                        redundant_with=names[retained[maximum]],
                        correlation=float(correlations[maximum]),
                    )
                    continue
            if len(retained) >= max_retained:
                record["status"] = "capacity"
            elif record["model_utility"] <= 0 and names[j] not in mandatory:
                record["status"] = "no_pilot_gain"
            else:
                retained.append(j)
                record["status"] = "retained"
        report = {
            "status": "passed",
            "input_id": input_id,
            "fitting_sessions": int(np.unique(sessions).size),
            "fitting_rows": rows,
            "candidate_features": len(names),
            "retained_count": len(retained),
            "retained": [names[j] for j in retained],
            "rejections": dict(Counter(r["status"] for r in records)),
            "retained_families": dict(Counter(records[j]["family"] for j in retained)),
            "selection_scope": "fitting sessions only; three grouped folds; "
            "no selection/evaluation labels",
            "features": records,
            "pilots": diagnostics,
            "elapsed_seconds": time.perf_counter() - start,
        }
        atomic_json(report_path, report)
    del x
    (output / "working_features.f32").unlink()
    return report
