"""Bounded fitting-only OOF baseline-vs-shared screen on frozen scale artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.research.dataset import OBJECTIVES, sampled_rows
from otto_recsys.research.fit_screen_contracts import advancement, fold_assignment
from otto_recsys.research.metrics import official_score, paired_bootstrap, ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat

CANDIDATE_BUDGET = 400


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def validate_scale(
    repo: Path, scale: Path, fit_contract: Path, config: dict[str, Any]
) -> tuple[pl.DataFrame, pl.DataFrame, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    report = json.loads(
        (repo / "reports/research/early_shared_feature_scale.json").read_text()
    )
    if (
        report["status"] != "EARLY_SHARED_FEATURE_SCALE_PASSED"
        or report["protocol"]["selection_access"] is not False
        or report["protocol"]["sessions"] != config["sessions"]
        or report["protocol"]["candidate_rows"] != config["sessions"] * CANDIDATE_BUDGET
    ):
        raise ValueError("scale evidence differs from the preregistered fitting-only screen")

    shared_config = json.loads(
        (repo / "configs/shared_feature_validation.json").read_text()
    )
    baseline = tuple(shared_config["baseline_features"])
    added = tuple(shared_config["added_features"])
    names = (*baseline, *added)
    if len(baseline) != 102 or len(added) != 32 or len(set(names)) != 134:
        raise ValueError("frozen screen schema must remain 102 + 32 unique features")

    frames = []
    for receipt in report["outputs"]["parts"]:
        path = scale / receipt["part"]
        if (
            receipt["sessions"] != 256
            or receipt["rows"] != 102400
            or sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"scale partition differs: {receipt['part']}")
        frames.append(pl.read_parquet(path))
    frame = pl.concat(frames).sort("session", "candidate_position")
    expected_columns = {
        *names,
        "session",
        "aid",
        "candidate_position",
        *[f"target_{objective}" for objective in OBJECTIVES],
    }
    if set(frame.columns) != expected_columns or frame.height != config["sessions"] * 400:
        raise ValueError("scale frame schema/row count differs")

    sessions = frame["session"].to_numpy()
    ids, starts, groups = np.unique(sessions, return_index=True, return_counts=True)
    positions = frame["candidate_position"].to_numpy()
    aids = frame["aid"].to_numpy()
    if (
        ids.size != config["sessions"]
        or not np.all(groups == CANDIDATE_BUDGET)
        or (np.diff(sessions) < 0).any()
    ):
        raise ValueError("screen requires sorted complete 400-candidate sessions")
    for start in starts:
        stop = start + CANDIDATE_BUDGET
        if (
            not np.array_equal(
                positions[start:stop], np.arange(CANDIDATE_BUDGET, dtype=positions.dtype)
            )
            or np.unique(aids[start:stop]).size != CANDIDATE_BUDGET
        ):
            raise ValueError("candidate identity/order differs within a complete session")

    ledger_path = scale / "fit_query_ledger.parquet"
    ledger_receipt = report["outputs"]["query_ledger"]
    if sha256_file(ledger_path) != ledger_receipt["sha256"]:
        raise ValueError("fitting denominator ledger checksum differs")
    ledger = pl.read_parquet(ledger_path).sort("session")
    if ledger.height != config["sessions"] or not np.array_equal(
        ledger["session"].to_numpy(), ids
    ):
        raise ValueError("fitting denominator ledger membership differs")

    source = json.loads(fit_contract.read_text())
    if (
        source["role"] != "fit"
        or source["candidate_budget"] != CANDIDATE_BUDGET
        or source["negative_budget"] != config["negative_budget"]
        or source["seed"] != config["candidate_seed"]
    ):
        raise ValueError("source fitting-negative contract differs")
    return frame, ledger, baseline, names, source


def sampled_training_indices(
    frame: pl.DataFrame,
    *,
    session_ids: np.ndarray,
    negative_budget: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sessions = frame["session"].to_numpy()
    aids = frame["aid"].to_numpy()
    target = frame.select(
        [f"target_{objective}" for objective in OBJECTIVES]
    ).to_numpy().astype(np.int8)
    chosen_global: list[np.ndarray] = []
    groups = []
    for session in session_ids:
        left = int(np.searchsorted(sessions, session, side="left"))
        right = int(np.searchsorted(sessions, session, side="right"))
        if right - left != CANDIDATE_BUDGET:
            raise ValueError("training session is not a complete candidate query")
        chosen = sampled_rows(
            target[left:right],
            aids[left:right],
            int(session),
            negative_budget,
            seed,
        )
        chosen_global.append(left + chosen)
        groups.append(chosen.size)
    return np.concatenate(chosen_global), np.asarray(groups, dtype=np.int32)


def fit_fixed(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    names: tuple[str, ...],
    *,
    seed: int,
    rounds: int,
    threads: int,
) -> lgb.Booster:
    if np.unique(y).size != 2 or groups.sum() != y.size:
        raise ValueError("fitting fold requires both classes and aligned groups")
    parameters = {
        "objective": "lambdarank",
        "metric": "None",
        "verbosity": -1,
        "num_threads": threads,
        "seed": seed,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 100,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "lambdarank_truncation_level": 25,
    }
    return lgb.train(
        parameters,
        lgb.Dataset(x, label=y, group=groups, feature_name=list(names)),
        num_boost_round=rounds,
    )


def run(
    repo: Path,
    scale: Path,
    fit_contract: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads((repo / "configs/shared_feature_fit_screen.json").read_text())
    frame, ledger, baseline_names, shared_names, source = validate_scale(
        repo, scale, fit_contract, config
    )
    ids = ledger["session"].to_numpy().astype(np.int64)
    fold = fold_assignment(ids, seed=config["fold_seed"], folds=config["folds"])
    all_sessions = frame["session"].to_numpy().astype(np.int64)
    all_aids = frame["aid"].to_numpy().astype(np.int64)
    target = frame.select(
        [f"target_{objective}" for objective in OBJECTIVES]
    ).to_numpy().astype(np.int8)
    full_x = frame.select(shared_names).to_numpy().astype(np.float32)
    denominators = ledger.select(
        [f"denominator_{objective}" for objective in OBJECTIVES]
    ).to_numpy().astype(np.int64)
    if not np.isfinite(full_x).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("screen inputs contain nonfinite features or invalid targets")

    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fit_screen")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    arms = {"baseline": baseline_names, "shared": shared_names}
    oof_hits = {
        arm: np.zeros((ids.size, len(OBJECTIVES)), dtype=np.int32) for arm in arms
    }
    models: dict[str, Any] = {arm: {} for arm in arms}
    fold_metrics: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    feature_index = {name: index for index, name in enumerate(shared_names)}
    model_fits = 0

    for f in range(config["folds"]):
        valid_session_mask = fold == f
        train_ids, valid_ids = ids[~valid_session_mask], ids[valid_session_mask]
        if np.intersect1d(train_ids, valid_ids).size:
            raise ValueError("whole-session folds overlap")
        train_rows, train_groups = sampled_training_indices(
            frame,
            session_ids=train_ids,
            negative_budget=config["negative_budget"],
            seed=config["candidate_seed"],
        )
        valid_row_mask = np.isin(all_sessions, valid_ids)
        valid_rows = np.flatnonzero(valid_row_mask)
        valid_groups = np.full(valid_ids.size, CANDIDATE_BUDGET, dtype=np.int32)
        if valid_rows.size != valid_ids.size * CANDIDATE_BUDGET:
            raise ValueError("OOF validation rows do not cover complete sessions")
        valid_slot = np.flatnonzero(valid_session_mask)

        for arm, names in arms.items():
            columns = np.asarray([feature_index[name] for name in names], dtype=np.int64)
            x_train = np.ascontiguousarray(full_x[train_rows][:, columns])
            x_valid = np.ascontiguousarray(full_x[valid_rows][:, columns])
            fold_hits = np.zeros((valid_ids.size, len(OBJECTIVES)), dtype=np.int32)
            for j, objective in enumerate(OBJECTIVES):
                with Heartbeat(
                    logger,
                    stage=f"fit_screen_{arm}_fold{f}_{objective}",
                    interval_seconds=15,
                ):
                    model = fit_fixed(
                        x_train,
                        target[train_rows, j],
                        train_groups,
                        names,
                        seed=config["model_seed"] + f,
                        rounds=config["rounds"],
                        threads=config["threads"],
                    )
                directory = output / "models" / arm / f"fold-{f}"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{objective}.txt"
                model.save_model(str(path))
                prediction = np.asarray(
                    model.predict(x_valid, num_threads=config["threads"])
                )
                fold_hits[:, j] = ranked_hits(
                    prediction,
                    all_aids[valid_rows],
                    target[valid_rows, j],
                    valid_groups,
                )
                models[arm][f"{f}:{objective}"] = {
                    "sha256": sha256_file(path),
                    "rounds": config["rounds"],
                    "features": len(names),
                }
                model_fits += 1
            oof_hits[arm][valid_slot] = fold_hits
            fold_metrics[arm].append(
                {
                    "fold": f,
                    "train_sessions": int(train_ids.size),
                    "validation_sessions": int(valid_ids.size),
                    **official_score(fold_hits, denominators[valid_slot]),
                }
            )

    summaries = {
        arm: {
            **official_score(hits, denominators),
            "folds": fold_metrics[arm],
        }
        for arm, hits in oof_hits.items()
    }
    fold_gains = [
        fold_metrics["shared"][i]["weighted_recall_at_20"]
        - fold_metrics["baseline"][i]["weighted_recall_at_20"]
        for i in range(config["folds"])
    ]
    decision = advancement(
        summaries["baseline"],
        summaries["shared"],
        fold_gains,
        minimum_gain=config["minimum_weighted_gain"],
        minimum_nonnegative_folds=config["minimum_nonnegative_folds"],
    )
    comparison = paired_bootstrap(
        oof_hits["shared"],
        oof_hits["baseline"],
        denominators,
        seed=config["bootstrap_seed"],
        replicates=config["bootstrap_replicates"],
    )
    stats = pl.DataFrame(
        {
            "session": ids,
            "fold": fold,
            **{
                f"denominator_{objective}": denominators[:, j]
                for j, objective in enumerate(OBJECTIVES)
            },
            **{
                f"baseline_hits_{objective}": oof_hits["baseline"][:, j]
                for j, objective in enumerate(OBJECTIVES)
            },
            **{
                f"shared_hits_{objective}": oof_hits["shared"][:, j]
                for j, objective in enumerate(OBJECTIVES)
            },
        }
    )
    stats_path = output / "oof_statistics.parquet"
    stats.write_parquet(stats_path, compression="zstd")
    result = {
        "status": "FITTING_ONLY_SHARED_SCREEN_PASSED",
        "scope": "Fitting-role OOF only; selection/evaluation inaccessible.",
        "source_scale_report": "reports/research/early_shared_feature_scale.json",
        "source_candidate_contract": {
            "candidate_budget": source["candidate_budget"],
            "negative_budget": source["negative_budget"],
            "seed": source["seed"],
        },
        "sessions": int(ids.size),
        "folds": config["folds"],
        "rounds": config["rounds"],
        "model_fits": model_fits,
        "selection_access": False,
        "arms": summaries,
        "paired_shared_vs_baseline": comparison,
        "decision": decision,
        "models": models,
        "statistics_sha256": sha256_file(stats_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "feature_retention_decisions": 0,
        "next_gate": (
            "If decision passes, run preregistered fitting-only leave-one-block-out "
            "ablations on the same folds/artifacts. Selection remains frozen."
        ),
    }
    atomic_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline": summaries["baseline"]["weighted_recall_at_20"],
                "shared": summaries["shared"]["weighted_recall_at_20"],
                "decision": decision,
                "model_fits": model_fits,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--fit-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.scale, args.fit_contract, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
