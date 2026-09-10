"""Observed-prefix support and descriptive complete-session error slices.

Diagnostics do not select models or alter the registered comparisons. Session slices
describe query context; they are not item-level repeat/new or rare-item recall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import sha256_file

OBJECTIVES = ("clicks", "carts", "orders")


def prefix_table(corpus: Path) -> pl.DataFrame:
    manifest = json.loads((corpus / "manifest.json").read_text())
    observed = corpus / "observed.parquet"
    if sha256_file(observed) != manifest["files"]["observed.parquet"]:
        raise ValueError("prefix diagnostics require the verified observed corpus")
    events = (
        pl.read_parquet(observed)
        .filter(pl.col("split_role").is_in(["fit", "selection"]))
        .sort("split_role", "session", "event_index")
    )
    keys = ["split_role", "session"]
    progress = (
        events.group_by([*keys, "aid"])
        .agg(pl.col("event_type").n_unique().alias("actions"))
        .group_by(keys)
        .agg((pl.col("actions") > 1).any().alias("multiple_actions_same_item"))
    )
    return (
        events.group_by(keys)
        .agg(
            pl.len().alias("events"),
            pl.col("aid").n_unique().alias("unique_items"),
            (pl.col("event_type") == 1).any().alias("observed_cart"),
            (pl.col("event_type") == 2).any().alias("observed_order"),
            pl.col("ts").diff().max().fill_null(0).alias("maximum_gap_ms"),
            (pl.col("ts").diff() == 0).sum().alias("tied_timestamps"),
        )
        .join(progress, on=keys, how="left", validate="1:1")
        .sort(keys)
    )


def slice_masks(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    length = frame["events"].to_numpy()
    return {
        "prefix_1": length == 1,
        "prefix_2_to_5": (length >= 2) & (length <= 5),
        "prefix_6_to_20": (length >= 6) & (length <= 20),
        "prefix_21_plus": length >= 21,
        "has_repeat_item": length > frame["unique_items"].to_numpy(),
        "all_items_distinct": length == frame["unique_items"].to_numpy(),
        "observed_cart_or_order": frame["observed_cart"].to_numpy()
        | frame["observed_order"].to_numpy(),
        "clicks_only": ~(frame["observed_cart"].to_numpy() | frame["observed_order"].to_numpy()),
        "gap_over_30_minutes": frame["maximum_gap_ms"].to_numpy() > 1_800_000,
        "gap_at_most_30_minutes": frame["maximum_gap_ms"].to_numpy() <= 1_800_000,
    }


def profile(corpus: Path) -> dict[str, Any]:
    frame = prefix_table(corpus)
    roles = {}
    for role in ("fit", "selection"):
        subset = frame.filter(pl.col("split_role") == role)
        length = subset["events"].to_numpy()
        roles[role] = {
            "sessions": subset.height,
            "events": int(length.sum()),
            "prefix_length_quantiles": {
                str(q): float(np.quantile(length, q)) for q in (0, 0.5, 0.9, 0.95, 0.99, 1)
            },
            "slice_sessions": {name: int(mask.sum()) for name, mask in slice_masks(subset).items()},
            "multiple_actions_same_item_sessions": int(subset["multiple_actions_same_item"].sum()),
            "observed_cart_sessions": int(subset["observed_cart"].sum()),
            "observed_order_sessions": int(subset["observed_order"].sum()),
            "gap_over_two_hours_sessions": int((subset["maximum_gap_ms"] > 7_200_000).sum()),
            "adjacent_timestamp_ties": int(subset["tied_timestamps"].sum()),
        }
    return {
        "scope": "Verified observed development prefixes only; no labels or model selection",
        "observed_sha256": sha256_file(corpus / "observed.parquet"),
        "code_sha256": sha256_file(Path(__file__)),
        "roles": roles,
    }


def error_slices(corpus: Path, study: Path) -> dict[str, Any]:
    frame = prefix_table(corpus).filter(pl.col("split_role") == "selection")
    sessions = frame["session"].to_numpy()
    manifest = json.loads((corpus / "manifest.json").read_text())
    if sha256_file(corpus / "queries.parquet") != manifest["files"]["queries.parquet"]:
        raise ValueError("diagnostic denominator ledger checksum differs")
    ledger = pl.read_parquet(corpus / "queries.parquet").filter(pl.col("split_role") == "selection")
    denominators = []
    for objective in OBJECTIVES:
        rows = ledger.filter(pl.col("objective") == objective).sort("session")
        if not np.array_equal(rows["session"].to_numpy(), sessions):
            raise ValueError("diagnostic query ledger differs")
        denominators.append(rows["recall_denominator"].to_numpy())
    den = np.column_stack(denominators)
    result = json.loads((study / "results.json").read_text())
    summaries: dict[str, Any] = {}
    files: dict[str, str] = {}
    for arm in result["arms"]:
        path = study / f"models/{arm}/selection_statistics.parquet"
        files[arm] = sha256_file(path)
        statistics = pl.read_parquet(path)
        if not np.array_equal(statistics["session"].to_numpy(), sessions):
            raise ValueError("diagnostic model session coverage differs")
        hits = statistics.select([f"hits_{o}" for o in OBJECTIVES]).to_numpy()
        if (hits < 0).any() or (hits > den).any():
            raise ValueError("diagnostic hits exceed the full capped targets")
        summaries[arm] = {}
        for name, mask in slice_masks(frame).items():
            h, d = hits[mask].sum(axis=0), den[mask].sum(axis=0)
            summaries[arm][name] = {
                "sessions": int(mask.sum()),
                "weighted_recall_at_20": float((h / d) @ [0.1, 0.3, 0.6])
                if (d > 0).all()
                else None,
                "objectives": {
                    o: {
                        "hits": int(h[j]),
                        "denominator": int(d[j]),
                        "recall_at_20": float(h[j] / d[j]) if d[j] else None,
                    }
                    for j, o in enumerate(OBJECTIVES)
                },
            }
    return {
        "study_id": result["study_id"],
        "scope": (
            "Descriptive query-context slices, selected before results were available; "
            "overlapping slice families; no adjusted intervals or promotion"
        ),
        "limits": (
            "These are whole-session context slices, not target-item repeat/new "
            "or rare-item recall."
        ),
        "code_sha256": sha256_file(Path(__file__)),
        "files": files,
        "arms": summaries,
    }
