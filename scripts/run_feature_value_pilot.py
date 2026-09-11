"""Purged fitting-only feature-value pilot; frozen candidates and native checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

OBJECTIVES = ("clicks", "carts", "orders")
SOURCES = (
    "hist_clicks_h6", "hist_carts_h72", "hist_orders_h72",
    "hist_carts_h168", "hist_all_h336", "hist_all_h1",
)
RELATIVE_NAMES = tuple(f"relative_{s}_{v}" for s in SOURCES for v in ("percentile", "mass"))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def preserve(path: Path, payload: bytes) -> None:
    """Atomic create only. Identical evidence is reusable; conflicts are preserved."""
    if path.is_symlink():
        raise ValueError("checkpoint symlink is forbidden")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"checkpoint conflict preserved: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_demand(counts: np.ndarray) -> np.ndarray:
    """Label-blind ranks/shares within each COMPLETE candidate pool, before sampling."""
    x = np.asarray(counts, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(SOURCES) or x.shape[0] < 2:
        raise ValueError("relative-demand input shape")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("historical counts must be finite and nonnegative")
    result = np.zeros((x.shape[0], 2 * x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        values = x[:, j]
        ordered = np.sort(values)
        middle = (np.searchsorted(ordered, values, side="left")
                  + np.searchsorted(ordered, values, side="right") - 1) / 2
        result[:, 2 * j] = np.where(values > 0, middle / (len(values) - 1), 0)
        total = values.sum()
        if total > 0:
            result[:, 2 * j + 1] = values / total
    return result


def purged_folds(
    query_ts: np.ndarray, embargo_ms: int = 6 * 60 * 60 * 1000,
) -> list[dict[str, Any]]:
    """Forward session folds with an embargo; labels are separately censored at cutoff."""
    q = np.asarray(query_ts)
    if q.shape != (1024,) or not np.isfinite(q).all() or embargo_ms < 0:
        raise ValueError("need 1024 finite query timestamps and a nonnegative embargo")
    order = np.argsort(q, kind="stable")
    folds = []
    for begin, stop in ((512, 768), (768, 1024)):
        valid = order[begin:stop]
        cutoff = int(q[valid].min())
        earlier = order[:begin]
        train = earlier[q[earlier] < cutoff - embargo_ms]
        if train.size < 128 or np.intersect1d(train, valid).size:
            raise ValueError("insufficient embargoed training support or overlapping fold")
        folds.append({"train": train, "valid": valid, "cutoff": cutoff,
                      "purged": begin - int(train.size)})
    return folds


def targets_asof(
    ids: np.ndarray, aids: np.ndarray, query_ts: np.ndarray,
    labels: dict[int, list[tuple[int, int, int]]], cutoff: int,
) -> np.ndarray:
    """Censor training targets BEFORE sampling negatives; never use later outcomes.

    Saved labels retain first future occurrence per cart/order item and the next
    click. These sufficient statistics exactly recover their positive sets before
    the exclusive cutoff, but do not certify that sessions have finished.
    """
    if aids.shape != (len(ids), 400) or query_ts.shape != ids.shape:
        raise ValueError("training target shape mismatch")
    if (query_ts >= cutoff).any():
        raise ValueError("training queries reach the validation cutoff")
    target = np.zeros((*aids.shape, 3), dtype=np.int8)
    for i, session in enumerate(ids):
        lookup = {int(aid): index for index, aid in enumerate(aids[i])}
        for aid, objective, ts in labels.get(int(session), []):
            if objective not in (0, 1, 2) or ts < query_ts[i]:
                raise ValueError("invalid timestamped fitting target")
            if ts < cutoff and aid in lookup:
                target[i, lookup[aid], objective] = 1
    return target


def session_hits(scores: np.ndarray, aids: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if scores.shape != aids.shape or scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError("prediction/target shape mismatch")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("invalid scores or binary targets")
    if any(np.unique(row).size != row.size for row in aids):
        raise ValueError("duplicate candidate identifiers")
    result = []
    for prediction, items, truth in zip(scores, aids, targets, strict=True):
        top = np.lexsort((items, -prediction))[:20]
        result.append(int(truth[top].sum()))
    return np.asarray(result, dtype=np.int64)


def pooled(hits: np.ndarray, denominators: np.ndarray) -> dict[str, Any]:
    h, d = np.asarray(hits), np.asarray(denominators)
    if h.shape != d.shape or h.ndim != 2 or h.shape[1] != 3:
        raise ValueError("pooled metric shape mismatch")
    if not np.isfinite(h).all() or not np.isfinite(d).all():
        raise ValueError("non-finite pooled metric")
    if (h < 0).any() or (h > d).any() or (d < 0).any() or (d > 20).any():
        raise ValueError("invalid pooled numerator/denominator")
    total = d.sum(axis=0)
    if (total <= 0).any():
        raise ValueError("missing objective support; no weighted score reported")
    recall = h.sum(axis=0) / total
    return {"weighted_recall_at_20": float(recall @ np.array([0.1, 0.3, 0.6])),
            "recall": dict(zip(OBJECTIVES, map(float, recall), strict=True)),
            "hits": h.sum(axis=0).tolist(), "denominators": total.tolist()}


def arms(config: dict[str, Any]) -> dict[str, list[str]]:
    base, added = config["baseline_features"], config["added_features"]
    full = base + added
    blocks = config["ablation_blocks"]
    covered = [name for block in blocks.values() for name in block]
    if len(base) != 102 or len(added) != 32 or len(set(full)) != 134:
        raise ValueError("frozen 102/134 schema differs")
    if len(covered) != 32 or set(covered) != set(added):
        raise ValueError("ablation blocks do not partition additions")
    result = {"baseline": base, "shared": full,
              "baseline_relative": base + list(RELATIVE_NAMES),
              "shared_relative": full + list(RELATIVE_NAMES)}
    for name, members in blocks.items():
        result[f"without_{name}"] = [column for column in full if column not in members]
    return result


def fit_or_reuse(
    directory: Path, contract: dict[str, Any], x: np.ndarray, y: np.ndarray,
    groups: list[int], valid: np.ndarray, params: dict[str, Any], rounds: int,
    feature_names: list[str],
) -> tuple[np.ndarray, int]:
    lgb = importlib.import_module("lightgbm")
    model_path, receipt = directory / "model.txt", directory / "receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["contract"] != contract or sha(model_path) != saved["model_sha256"]:
            raise ValueError("model checkpoint identity differs")
        model = lgb.Booster(model_file=str(model_path))
        fits = 0
    else:
        if model_path.exists():
            raise ValueError("orphan model preserved; audit before fitting")
        if sum(groups) != len(y) or len(y) != x.shape[0] or int(y.sum()) < 5:
            raise ValueError("invalid group sizes or insufficient training positives")
        train = lgb.Dataset(x, label=y, group=groups, feature_name=feature_names)
        model = lgb.train(params, train, num_boost_round=rounds)
        preserve(model_path, model.model_to_string().encode())
        preserve(receipt, encode({"contract": contract, "model_sha256": sha(model_path)}))
        fits = 1
    if model.feature_name() != feature_names:
        raise ValueError("native model feature order differs")
    prediction = np.asarray(model.predict(valid, num_threads=4), dtype=np.float64)
    loaded = lgb.Booster(model_file=str(model_path))
    replay = np.asarray(loaded.predict(valid, num_threads=4), dtype=np.float64)
    if not np.array_equal(prediction, replay):
        raise ValueError("saved native model prediction replay differs")
    return prediction, fits


def run(repo: Path, scale: Path, corpus: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    pl = importlib.import_module("polars")
    dataset = importlib.import_module("otto_recsys.research.dataset")
    config = json.loads((repo / "configs/shared_feature_validation.json").read_text())
    protocol = json.loads((repo / "configs/feature_value_pilot.json").read_text())
    if (protocol["role"] != "fit" or protocol["selection_access"] is not False
            or protocol["evaluation_access"] is not False):
        raise ValueError("pilot must remain fitting-only")
    source_result = json.loads((scale / "result.json").read_text())
    if source_result["status"] != "EARLY_SHARED_FEATURE_SCALE_PASSED":
        raise ValueError("scale prerequisite is not passed")
    if source_result["history_end"] != 1660687200000:
        raise ValueError("wrong historical cutoff")
    source_hashes = {"result.json": sha(scale / "result.json")}
    partitions = []
    for item in source_result["parts"]:
        path = scale / item["part"]
        if path.parent.resolve() != scale.resolve() or sha(path) != item["sha256"]:
            raise ValueError("feature partition checksum/path mismatch")
        partitions.append(pl.read_parquet(path))
        source_hashes[path.name] = item["sha256"]
    frame = pl.concat(partitions).sort("session", "candidate_position")
    ids = frame["session"].unique().sort().to_numpy()
    if ids.size != 1024 or frame.height != 409600:
        raise ValueError("complete cohort shape mismatch")
    if not np.array_equal(frame["session"].to_numpy(), np.repeat(ids, 400)):
        raise ValueError("incomplete session groups")
    if not np.array_equal(frame["candidate_position"].to_numpy(), np.tile(np.arange(400), 1024)):
        raise ValueError("candidate order differs")
    names = config["baseline_features"] + config["added_features"]
    full = frame.select(names).to_numpy().astype(np.float32).reshape(1024, 400, 134)
    if not np.isfinite(full).all():
        raise ValueError("non-finite saved feature matrix")
    aids = frame["aid"].to_numpy().reshape(1024, 400)
    target = frame.select([f"target_{o}" for o in OBJECTIVES]).to_numpy().reshape(1024, 400, 3)
    if not np.isin(target, [0, 1]).all() or any(np.unique(a).size != 400 for a in aids):
        raise ValueError("invalid targets or duplicate candidates")
    ledger_path = scale / "fit_query_ledger.parquet"
    if sha(ledger_path) != source_result["query_ledger"]["sha256"]:
        raise ValueError("denominator ledger checksum mismatch")
    ledger = pl.read_parquet(ledger_path).sort("session")
    if not np.array_equal(ledger["session"].to_numpy(), ids):
        raise ValueError("denominator cohort mismatch")
    denom = ledger.select([f"denominator_{o}" for o in OBJECTIVES]).to_numpy()
    source_hashes[ledger_path.name] = sha(ledger_path)
    manifest = json.loads((corpus / "manifest.json").read_text())
    if manifest["protocol"]["history_end"] != 1660687200000:
        raise ValueError("timestamp source cutoff differs")
    times: dict[int, int] = {}
    timestamped: dict[int, list[tuple[int, int, int]]] = {}
    for filename in ("queries.parquet", "labels.parquet"):
        if sha(corpus / filename) != manifest["files"][filename]:
            raise ValueError("timestamp source checksum mismatch")
        source_hashes[f"corpus/{filename}"] = manifest["files"][filename]
        table = (pl.scan_parquet(corpus / filename)
                 .filter((pl.col("split_role") == "fit") & pl.col("session").is_in(ids.tolist()))
                 .collect())
        if filename == "queries.parquet":
            pairs = table.group_by("session").agg(pl.col("query_ts").max()).iter_rows()
            times = {int(k): int(v) for k, v in pairs}
        else:
            for session, aid, objective, ts in table.select(
                "session", "aid", "objective", "label_ts"
            ).iter_rows():
                timestamped.setdefault(int(session), []).append(
                    (int(aid), OBJECTIVES.index(objective), int(ts))
                )
    q = np.asarray([times[int(s)] for s in ids], dtype=np.int64)
    folds = purged_folds(q, int(protocol["embargo_hours"]) * 60 * 60 * 1000)
    historical_indices = [names.index(s) for s in SOURCES]
    relative = np.stack([relative_demand(x[:, historical_indices]) for x in full])
    data = np.concatenate((full, relative), axis=2)
    universe = names + list(RELATIVE_NAMES)
    schemas = arms(config)
    candidate_contract_path = corpus.parent / "fit_cache/contract.json"
    candidate_contract = json.loads(candidate_contract_path.read_text())
    if candidate_contract["role"] != "fit" or candidate_contract["candidate_budget"] != 400:
        raise ValueError("frozen negative-sampling contract differs")
    source_hashes["candidate_contract"] = sha(candidate_contract_path)
    identity = {"sources": source_hashes, "protocol": protocol,
                "runner_sha256": sha(Path(__file__)),
                "schema_sha256": sha(repo / "configs/shared_feature_validation.json")}
    preserve(output / "contract.json", encode(identity))
    training = protocol["lightgbm"]
    params = {**training, "objective": "lambdarank", "metric": "None",
              "verbosity": -1, "deterministic": True, "force_col_wise": True,
              "num_threads": 4, "seed": 20260911, "feature_fraction": 1.0,
              "bagging_fraction": 1.0, "bagging_freq": 0}
    fits = 0
    records = []
    per_arm: dict[str, list[np.ndarray]] = {name: [] for name in schemas}
    all_denoms = []
    folds_report = []
    for fold_index, fold in enumerate(folds):
        train_ids, valid_ids = fold["train"], fold["valid"]
        all_denoms.append(denom[valid_ids])
        train_target = targets_asof(
            ids[train_ids], aids[train_ids], q[train_ids], timestamped, fold["cutoff"]
        )
        selected = [dataset.sampled_rows(train_target[k], aids[i], int(ids[i]),
                    int(candidate_contract["negative_budget"]), int(candidate_contract["seed"]))
                    for k, i in enumerate(train_ids)]
        groups = [len(values) for values in selected]
        xtrain = np.concatenate([
            data[i, chosen] for i, chosen in zip(train_ids, selected, strict=True)
        ])
        ytrain = np.concatenate([train_target[k, chosen] for k, chosen in enumerate(selected)])
        validation = data[valid_ids].reshape(-1, len(universe))
        folds_report.append({"fold": fold_index, "train_sessions": len(train_ids),
                             "valid_sessions": len(valid_ids), "purged": fold["purged"],
                             "cutoff_ms": fold["cutoff"],
                             "latest_training_query_ms": int(q[train_ids].max()),
                             "training_label_horizon_exclusive_ms": fold["cutoff"],
                             "training_targets_censored_before_sampling": True,
                             "full_horizon_positive_rows_removed": int(
                                 target[train_ids].sum() - train_target.sum()
                             ),
                             "positive_training_rows": ytrain.sum(axis=0).tolist(),
                             "validation_denominators": denom[valid_ids].sum(axis=0).tolist()})
        preserve(output / f"fold-{fold_index}.json", encode({
            **folds_report[-1], "train_ids": ids[train_ids].tolist(),
            "valid_ids": ids[valid_ids].tolist()}))
        for arm, feature_names in schemas.items():
            if time.monotonic() - started > protocol["work_seconds"] - 30:
                raise TimeoutError("pilot deadline; completed model checkpoints preserved")
            arm_start = time.monotonic()
            columns = [universe.index(name) for name in feature_names]
            hits = np.zeros((len(valid_ids), 3), dtype=np.int64)
            for j, objective in enumerate(OBJECTIVES):
                contract = {"study": identity, "fold": fold_index, "arm": arm,
                            "objective": objective, "features": feature_names,
                            "train_ids": ids[train_ids].tolist(),
                            "valid_ids": ids[valid_ids].tolist()}
                scores, count = fit_or_reuse(
                    output / "models" / f"fold-{fold_index}" / arm / objective,
                    contract, xtrain[:, columns], ytrain[:, j], groups,
                    validation[:, columns], params, int(protocol["rounds"]), feature_names)
                fits += count
                hits[:, j] = session_hits(
                    scores.reshape(-1, 400), aids[valid_ids], target[valid_ids, :, j]
                )
            metrics = pooled(hits, denom[valid_ids])
            per_arm[arm].append(hits)
            record = {"fold": fold_index, "arm": arm, "features": len(feature_names),
                      **metrics, "elapsed_seconds": time.monotonic() - arm_start}
            records.append(record)
            preserve(output / f"fold-{fold_index}-{arm}.json", encode({
                "metrics": metrics, "hits_by_session": hits.tolist(),
                "session_ids": ids[valid_ids].tolist()}))
            print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "stage": "feature_arm_complete", **record}), flush=True)
    den = np.concatenate(all_denoms)
    hit_arrays = {name: np.concatenate(values) for name, values in per_arm.items()}
    aggregates = {name: pooled(value, den) for name, value in hit_arrays.items()}
    comparisons = {}
    pairs = [("shared", "baseline"), ("baseline_relative", "baseline"),
             ("shared_relative", "shared")]
    pairs += [("shared", f"without_{name}") for name in config["ablation_blocks"]]
    rng = np.random.default_rng(20260911)
    bootstrap = [rng.integers(0, den.shape[0], den.shape[0]) for _ in range(1000)]
    for left, right in pairs:
        differences = []
        for idx in bootstrap:
            if (den[idx].sum(axis=0) == 0).any():
                continue
            delta = ((hit_arrays[left][idx] - hit_arrays[right][idx]).sum(axis=0)
                     / den[idx].sum(axis=0)) @ np.array([0.1, 0.3, 0.6])
            differences.append(float(delta))
        comparisons[f"{left}_minus_{right}"] = {
            "difference": aggregates[left]["weighted_recall_at_20"]
            - aggregates[right]["weighted_recall_at_20"],
            "descriptive_95_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
            "bootstrap_replicates": len(differences)}
    evaluation_indices = np.concatenate([fold["valid"] for fold in folds])
    oracle = pooled(np.minimum(target[evaluation_indices].sum(axis=1), 20), den)
    model_inventory = {
        str(path.relative_to(output)): sha(path)
        for path in sorted((output / "models").rglob("model.txt"))
    }
    if len(model_inventory) != protocol["maximum_models"]:
        raise ValueError("native model inventory differs from preregistration")
    result = {"status": "FITTING_FEATURE_VALUE_PILOT_PASSED", "role": "fit",
              "selection_access": False, "evaluation_access": False,
              "new_model_fits": fits, "completed_models": len(schemas) * 6,
              "folds": folds_report, "fold_results": records,
              "arms": aggregates, "comparisons": comparisons,
              "candidate_oracle_recall_at_20": oracle,
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "source_hashes": source_hashes, "model_native_replay": True,
              "model_inventory": model_inventory,
              "new_features": list(RELATIVE_NAMES),
              "feature_retention_decisions": 0,
              "limitations": ["Small, reused fitting cohort; not Kaggle or selection score.",
                              "Two embargoed forward folds; training targets censored at cutoff.",
                              "Retrospective within-fit prefixes; not an online backtest.",
                              "One model seed; exploratory intervals on dependent folds.",
                              "Fixed Aug-16 historical features, not rolling as-of snapshots.",
                              "Family/relative-demand comparisons do not authorize promotion."]}
    result_path = output / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["arms"] != aggregates or previous["source_hashes"] != source_hashes:
            raise ValueError("completed result replay mismatch")
        if fits != 0:
            raise ValueError("completed replay unexpectedly fitted a model")
        preserve(output / "replay.json", encode({"model_fits": 0, "identical_metrics": True}))
    else:
        preserve(result_path, encode(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "scale", "corpus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo, args.scale, args.corpus, args.output)
    print(json.dumps({"status": result["status"], "new_model_fits": result["new_model_fits"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
