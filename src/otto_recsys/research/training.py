"""Task-specific rankers selected on complete chronological selection queries."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.metrics import ranked_hits
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat


def fit_model(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_groups: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_aids: np.ndarray,
    valid_groups: np.ndarray,
    denominator: int,
    *,
    names: tuple[str, ...],
    directory: Path,
    lineage: dict[str, Any],
    config: dict[str, Any],
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Atomic per-iteration recovery; first maximum wins metric ties."""
    if (
        any(int(config[key]) < 1 for key in ("threads", "rounds", "patience"))
        or int(config.get("evaluation_interval", 5)) < 1
    ):
        raise ValueError("ranker resource and stopping settings must be positive")
    if (
        fit_x.shape != (fit_y.size, len(names))
        or valid_x.shape != (valid_y.size, len(names))
        or fit_groups.sum() != fit_y.size
        or valid_groups.sum() != valid_y.size
        or denominator < 1
        or np.unique(fit_y).size < 2
        or np.unique(valid_y).size < 2
    ):
        raise ValueError(
            "ranker inputs must align with complete query groups and both target classes"
        )
    if lineage.get("fit_role") != "fit" or lineage.get("selection_role") != "selection":
        raise ValueError("ranker fitting and selection must use their declared temporal roles")
    parameters = {
        "objective": "lambdarank",
        "metric": "None",
        "verbosity": -1,
        "num_threads": int(config["threads"]),
        "seed": int(config["seed"]),
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 100,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "lambdarank_truncation_level": 25,
    }
    contract = {
        "lineage": lineage,
        "config": config,
        "parameters": parameters,
        "features": list(names),
        "lightgbm": lgb.__version__,
        "code_sha256": sha256_file(Path(__file__)),
        "selection_metric": "complete-query official Recall@20; first maximum wins ties",
    }
    input_id = canonical_json_sha256(contract)
    directory.mkdir(parents=True, exist_ok=True)
    with workspace_lock(directory):
        contract_path = directory / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("ranker checkpoint has a different experiment contract")
        atomic_json(contract_path, contract)
        if publish:
            publish(contract_path)
        checkpoint = directory / "checkpoints"
        checkpoint.mkdir(exist_ok=True)
        initial = None
        state: dict[str, Any] = {
            "input_id": input_id,
            "iteration": 0,
            "best_iteration": 0,
            "best_score": -1.0,
            "complete": False,
            "retained_fit_seconds": 0.0,
        }
        for path in sorted(checkpoint.glob("*.json"), reverse=True):
            try:
                saved = json.loads(path.read_text())
                model_path = path.with_suffix(".txt")
                if saved["input_id"] != input_id or saved["sha256"] != sha256_file(model_path):
                    continue
                candidate = lgb.Booster(model_file=str(model_path))
                if candidate.current_iteration() != saved["iteration"]:
                    continue
                state, initial = saved, candidate
                break
            except (OSError, ValueError, KeyError, TypeError, lgb.basic.LightGBMError):
                continue
        if state["complete"]:
            model_path = directory / "model.txt"
            if initial is None:
                raise ValueError("completed ranker receipt has no checkpoint model")
            initial.save_model(str(model_path), num_iteration=int(state["best_iteration"]))
            state["model_sha256"] = sha256_file(model_path)
            state["features"] = list(names)
            state["gain_importance"] = dict(
                zip(
                    names,
                    initial.feature_importance(
                        importance_type="gain", iteration=int(state["best_iteration"])
                    ).tolist(),
                    strict=True,
                )
            )
            atomic_json(directory / "manifest.json", state)
            if publish:
                publish(model_path)
                publish(directory / "manifest.json")
            return state
        start = time.perf_counter()
        previous_seconds = float(state["retained_fit_seconds"])
        train = lgb.Dataset(fit_x, label=fit_y, group=fit_groups, feature_name=list(names))
        valid = lgb.Dataset(
            valid_x, label=valid_y, group=valid_groups, reference=train, feature_name=list(names)
        )

        def metric(prediction: Any, _dataset: Any) -> tuple[str, float, bool]:
            iteration = int(state["iteration"]) + 1
            interval = int(config.get("evaluation_interval", 5))
            if iteration != 1 and iteration % interval and "last_evaluated_score" in state:
                return "official_recall_at_20", float(state["last_evaluated_score"]), True
            hits = ranked_hits(prediction, valid_aids, valid_y, valid_groups)
            score = float(hits.sum() / denominator)
            state["last_evaluation_iteration"] = iteration
            state["last_evaluated_score"] = score
            return "official_recall_at_20", score, True

        def save(booster: lgb.Booster) -> None:
            path = checkpoint / f"{state['iteration']:06d}.txt"
            temporary = path.with_suffix(".txt.tmp")
            booster.save_model(str(temporary), num_iteration=-1)
            temporary.replace(path)
            state["retained_fit_seconds"] = previous_seconds + time.perf_counter() - start
            state["sha256"] = sha256_file(path)
            atomic_json(path.with_suffix(".json"), state)
            if publish:
                publish(path)
                publish(path.with_suffix(".json"))
            logger.info(
                "research_ranker_checkpoint",
                extra={
                    "iteration": state["iteration"],
                    "best_iteration": state["best_iteration"],
                    "best_score": state["best_score"],
                    "elapsed_seconds": state["retained_fit_seconds"],
                },
            )

        def progress(environment: Any) -> None:
            score = float(environment.evaluation_result_list[0][2])
            state["iteration"] = environment.model.current_iteration()
            if score > state["best_score"]:
                state["best_score"] = score
                state["best_iteration"] = state["iteration"]
            exhausted = state["iteration"] - state["best_iteration"] >= config["patience"]
            state["complete"] = bool(exhausted or state["iteration"] >= config["rounds"])
            if state["iteration"] % 25 == 0 or state["complete"]:
                save(environment.model)
            if exhausted:
                raise lgb.callback.EarlyStopException(
                    int(state["best_iteration"]) - 1, environment.evaluation_result_list
                )

        with Heartbeat(
            logger, stage="research_ranker", interval_seconds=15, progress_provider=state.copy
        ):
            booster = lgb.train(
                parameters,
                train,
                num_boost_round=config["rounds"] - state["iteration"],
                valid_sets=[valid],
                feval=metric,
                init_model=initial,
                keep_training_booster=True,
                callbacks=[progress],
            )
        state["complete"] = True
        save(booster)
        model_path = directory / "model.txt"
        booster.save_model(str(model_path), num_iteration=int(state["best_iteration"]))
        state["model_sha256"] = sha256_file(model_path)
        state["features"] = list(names)
        state["gain_importance"] = dict(
            zip(
                names,
                booster.feature_importance(
                    importance_type="gain", iteration=int(state["best_iteration"])
                ).tolist(),
                strict=True,
            )
        )
        atomic_json(directory / "manifest.json", state)
        if publish:
            publish(model_path)
            publish(directory / "manifest.json")
        return state
