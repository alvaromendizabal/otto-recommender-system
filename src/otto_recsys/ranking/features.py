"""Observed-prefix features and label-blind candidate joins for learned ranking."""

from __future__ import annotations

import polars as pl

OBJECTIVES = ("clicks", "carts", "orders")
SOURCES = ("revisit", "time", "type", "buy", "item2vec", "two_tower")
SESSION_FEATURES = (
    "session_events",
    "session_unique_items",
    "session_duration_ms",
    "session_clicks",
    "session_carts",
    "session_orders",
)
ITEM_FEATURES = (
    "item_events",
    "item_clicks",
    "item_carts",
    "item_orders",
    "item_age_ms",
    "item_last_type",
    "item_event_share",
)
FEATURE_COLUMNS = (
    *(f"{source}_{kind}" for source in SOURCES for kind in ("present", "rank", "score")),
    "source_count",
    "reciprocal_rank_sum",
    *SESSION_FEATURES,
    *ITEM_FEATURES,
)
FEATURE_SPEC = {
    "columns": list(FEATURE_COLUMNS),
    "timestamps": "UTC Unix milliseconds; durations and ages in milliseconds",
    "source_missing": "presence=0; score/rank=null; reciprocal-rank contribution=0",
    "unobserved_item": "counts/share=0; age/last_type=null",
    "inputs": "observed session prefix and label-blind retriever outputs only",
    "target": "binary membership; attached only after candidate generation",
    "query": "session * 3 + objective index (clicks=0, carts=1, orders=2)",
    "excluded_from_model": [
        "session",
        "aid",
        "objective",
        "query_id",
        "fold",
        "bucket",
        "inner_partition",
        "target",
        "true_items",
        "recall_denominator",
    ],
    "positive_insertion": False,
    "negative_sampling": False,
}


def require_unique(frame: pl.DataFrame, keys: list[str], name: str) -> None:
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"{name} contains null keys")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name} contains duplicate keys")


def observed_features(
    events: pl.DataFrame,
    examples: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate only observed events; verify boundaries against the frozen cache."""
    require_unique(examples, ["session"], "examples")
    require_unique(events, ["session", "event_index"], "events")
    if events.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("events contains null values")
    if events.filter(~pl.col("event_type").is_in([0, 1, 2]) | (pl.col("aid") < 0)).height:
        raise ValueError("invalid observed event")
    ordered = events.sort(["session", "event_index"])
    if ordered.filter(pl.col("ts").diff().over("session") < 0).height:
        raise ValueError("observed events are not chronological")
    if events.join(
        examples.select("session", "fold", "bucket"), on=["session", "fold", "bucket"], how="anti"
    ).height:
        raise ValueError("event session/fold/bucket mismatch")
    sessions = events.group_by("session").agg(
        pl.len().alias("session_events"),
        pl.col("aid").n_unique().alias("session_unique_items"),
        pl.col("ts").min().alias("first_ts"),
        pl.col("ts").max().alias("last_ts"),
        pl.col("event_index").min().alias("first_index"),
        pl.col("event_index").max().alias("last_index"),
        *(
            (pl.col("event_type") == i).sum().alias(f"session_{name}")
            for i, name in enumerate(OBJECTIVES)
        ),
    )
    expected = examples.select(
        "session",
        "observed_events",
        "observed_unique_items",
        pl.col("first_ts").alias("expected_first"),
        pl.col("last_ts").alias("expected_last"),
    )
    check = sessions.join(expected, on="session", how="full", coalesce=True)
    if check.filter(
        pl.any_horizontal(pl.all().is_null())
        | (pl.col("session_events") != pl.col("observed_events"))
        | (pl.col("session_unique_items") != pl.col("observed_unique_items"))
        | (pl.col("first_ts") != pl.col("expected_first"))
        | (pl.col("last_ts") != pl.col("expected_last"))
        | (pl.col("first_index") != 0)
        | (pl.col("last_index") != pl.col("session_events") - 1)
    ).height:
        raise ValueError("observed prefix disagrees with frozen session boundaries")
    sessions = (
        sessions.drop("first_index", "last_index")
        .with_columns((pl.col("last_ts") - pl.col("first_ts")).alias("session_duration_ms"))
        .join(examples.select("session", "fold", "bucket", "inner_partition"), on="session")
    )
    items = (
        ordered.group_by(["session", "aid"])
        .agg(
            pl.len().alias("item_events"),
            pl.col("ts").max().alias("item_last_ts"),
            pl.col("event_type").last().alias("item_last_type"),
            *(
                (pl.col("event_type") == i).sum().alias(f"item_{name}")
                for i, name in enumerate(OBJECTIVES)
            ),
        )
        .join(sessions.select("session", "last_ts", "session_events"), on="session")
        .with_columns(
            (pl.col("last_ts") - pl.col("item_last_ts")).alias("item_age_ms"),
            (pl.col("item_events") / pl.col("session_events")).alias("item_event_share"),
        )
        .select("session", "aid", *ITEM_FEATURES)
    )
    return sessions.sort("session"), items.sort(["session", "aid"])


def query_ledger(examples: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Retain all queries and full capped denominators, including retrieval misses."""
    require_unique(labels, ["session", "objective", "aid"], "labels")
    if labels.filter(~pl.col("objective").is_in(OBJECTIVES) | (pl.col("aid") < 0)).height:
        raise ValueError("invalid label")
    if labels.join(
        examples.select("session", "fold", "bucket"), on=["session", "fold", "bucket"], how="anti"
    ).height:
        raise ValueError("label session/fold/bucket mismatch")
    objectives = pl.DataFrame({"objective": OBJECTIVES, "objective_id": [0, 1, 2]})
    counts = labels.group_by(["session", "objective"]).agg(pl.len().alias("true_items"))
    return (
        examples.select("session", "fold", "bucket", "inner_partition")
        .join(
            objectives,
            how="cross",
        )
        .join(counts, on=["session", "objective"], how="left")
        .with_columns(
            pl.col("true_items").fill_null(0),
            (pl.col("session").cast(pl.Int64) * 3 + pl.col("objective_id")).alias("query_id"),
        )
        .with_columns(pl.col("true_items").clip(upper_bound=20).alias("recall_denominator"))
        .drop("objective_id")
        .sort(["objective", "session"])
    )


def candidate_features(
    sources: pl.DataFrame,
    sessions: pl.DataFrame,
    items: pl.DataFrame,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    """Deduplicate source candidates, preserve source evidence, then attach targets.

    This transformation does not certify retriever fit provenance. The caller must
    validate each learned source against its outer/inner split before training.
    """
    require_unique(sources, ["source", "objective", "session", "aid"], "sources")
    require_unique(labels, ["session", "objective", "aid"], "labels")
    if sources.filter(
        ~pl.col("source").is_in(SOURCES)
        | ~pl.col("objective").is_in(OBJECTIVES)
        | (pl.col("aid") < 0)
        | (pl.col("source_rank") < 1)
        | (pl.col("source_rank") != pl.col("source_rank").floor())
        | ~pl.col("score").is_finite()
        | pl.col("score").is_null()
        | pl.col("source_rank").is_null()
    ).height:
        raise ValueError("invalid source candidate")
    require_unique(sessions, ["session"], "session features")
    require_unique(items, ["session", "aid"], "item features")
    if sources.join(sessions.select("session"), on="session", how="anti").height:
        raise ValueError("candidate session lacks observed features")
    expressions: list[pl.Expr] = []
    for source in SOURCES:
        present = pl.col("source") == source
        expressions.extend(
            [
                present.any().cast(pl.UInt8).alias(f"{source}_present"),
                pl.col("source_rank").filter(present).min().alias(f"{source}_rank"),
                pl.col("score").filter(present).max().alias(f"{source}_score"),
            ]
        )
    rows = (
        sources.group_by(["objective", "session", "aid"])
        .agg(
            *expressions,
            pl.col("source").n_unique().alias("source_count"),
            (1.0 / pl.col("source_rank")).sum().alias("reciprocal_rank_sum"),
        )
        .join(sessions.select("session", *SESSION_FEATURES), on="session")
        .join(
            items,
            on=["session", "aid"],
            how="left",
        )
        .with_columns(
            pl.col(
                "item_events",
                "item_clicks",
                "item_carts",
                "item_orders",
                "item_event_share",
            ).fill_null(0)
        )
        .with_columns(
            (
                pl.col("session").cast(pl.Int64) * 3
                + pl.col("objective").replace_strict(dict(zip(OBJECTIVES, range(3), strict=True)))
            ).alias("query_id"),
        )
    )
    # Labels cannot alter membership, ordering, or feature values.
    targets = labels.select("session", "objective", "aid").with_columns(pl.lit(1).alias("target"))
    return (
        rows.join(targets, on=["session", "objective", "aid"], how="left")
        .with_columns(
            pl.col("target").fill_null(0).cast(pl.UInt8),
        )
        .select("session", "objective", "aid", "query_id", *FEATURE_COLUMNS, "target")
        .sort(["objective", "session", "aid"])
    )
