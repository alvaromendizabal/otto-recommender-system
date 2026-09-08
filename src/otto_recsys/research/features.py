"""Label-blind candidate generation and a catalog of temporal graph features.

Only requested columns are materialized. The broad catalog is screened on
fitting queries; later stages compute the retained columns in bounded batches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.retrievers import CHANNELS, HISTORY_HOURS

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
ACTIONS = ("all", "clicks", "carts", "orders")
LENGTHS = (1, 3, 5, 10, 20, 50)
WEIGHTS = ("uniform", "position", "hour", "six_hours")
AGGREGATES = ("sum", "max", "mean", "std")


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    definition: str
    availability: str
    dtype: str = "float32"


def feature_catalog() -> list[Feature]:
    """Domain-motivated formulas, without identifiers, labels or future values."""
    result = []

    def add(name: str, family: str, definition: str, historical: bool = False) -> None:
        availability = (
            "fixed historical cutoff and observed prefix" if historical else "observed prefix"
        )
        result.append(Feature(name, family, definition, availability))

    for source in (*CHANNELS, "revisit", "popularity"):
        for value in ("score", "rank", "present", "share"):
            add(f"source_{source}_{value}", "source", f"{source} evidence: {value}", True)
    for name in (
        "source_count",
        "source_rank_best",
        "source_rank_mean",
        "source_rank_std",
        "source_score_max",
        "source_score_mean",
        "source_score_std",
    ):
        add(name, "source", name.replace("_", " "), True)
    for left, right in (("time", "cart"), ("time", "order"), ("cart", "order")):
        for transform in ("product", "difference", "ratio"):
            add(
                f"source_{left}_{right}_{transform}",
                "source",
                f"{transform} of query-normalized {left}/{right} evidence",
                True,
            )
    for name in (
        "query_events",
        "query_unique",
        "query_duration_hours",
        "query_repeat_share",
        "query_hour_sin",
        "query_hour_cos",
        "query_weekday_sin",
        "query_weekday_cos",
        "query_gap_mean",
        "query_gap_std",
        "query_gap_max",
        "query_gap_median",
    ):
        add(name, "context", name.replace("_", " "))
    for action in ACTIONS:
        for name in ("count", "share", "last_age_hours"):
            add(f"query_{action}_{name}", "context", f"Observed {action}: {name}")
        for horizon in (0, *HISTORY_HOURS):
            label = f"h{horizon}" if horizon else "all"
            add(
                f"hist_{action}_{label}", "history", f"Historical {action} count over {label}", True
            )
            if horizon:
                add(
                    f"hist_{action}_{label}_share",
                    "history",
                    f"Share of historical {action} activity in last {horizon} hours",
                    True,
                )
        for short, long in ((1, 6), (6, 24), (24, 72), (72, 168), (168, 336)):
            add(
                f"hist_{action}_trend_h{short}_h{long}",
                "history",
                f"Per-hour {action} rate over {short}h divided by {long}h rate",
                True,
            )
        for length in LENGTHS:
            for name in ("count", "share", "last_position", "first_position", "span", "age_hours"):
                add(
                    f"repeat_{action}_n{length}_{name}",
                    "repeat",
                    f"Candidate {action} recurrence within last {length} observed events: {name}",
                )
        for horizon in (1, 6, 24, 72):
            add(
                f"repeat_{action}_h{horizon}_decay",
                "repeat",
                f"Sum of exponential candidate {action} weights with {horizon}h time constant",
            )
    for action in ("carts", "orders"):
        for horizon in (0, *HISTORY_HOURS):
            label = f"h{horizon}" if horizon else "all"
            add(
                f"hist_{action}_per_click_{label}",
                "history",
                f"Historical {action}/(1+clicks) over {label}",
                True,
            )
    for name in ("history_known", "history_last_age_hours", "history_lifetime_hours"):
        add(name, "history", name.replace("_", " "), True)
    for channel in CHANNELS:
        for action in ACTIONS:
            for length in LENGTHS:
                for weight in WEIGHTS:
                    for aggregate in AGGREGATES:
                        add(
                            f"graph_{channel}_{action}_n{length}_{weight}_{aggregate}",
                            "graph",
                            f"{aggregate} of {channel} graph affinity from last {length} observed "
                            f"events restricted to {action}, with {weight} weights; "
                            "missing edges zero",
                            True,
                        )
    for channel in CHANNELS:
        for action in ("clicks", "carts", "orders"):
            add(
                f"intent_{channel}_{action}",
                "interaction",
                f"Normalized {channel} evidence multiplied by observed {action} share",
                True,
            )
    for action in ("carts", "orders"):
        add(
            f"intent_repeat_{action}",
            "interaction",
            f"Candidate repeat count multiplied by historical {action}/click ratio",
            True,
        )
    return result


@dataclass(frozen=True)
class Prefix:
    session: int
    aid: IntArray
    ts: IntArray
    kind: IntArray

    def validate(self) -> None:
        if (
            not self.aid.size
            or self.aid.shape != self.ts.shape
            or self.aid.shape != self.kind.shape
        ):
            raise ValueError("observed prefix arrays must be nonempty and aligned")
        if (
            (self.aid < 0).any()
            or (np.diff(self.ts) < 0).any()
            or not np.isin(self.kind, [0, 1, 2]).all()
        ):
            raise ValueError(
                "observed prefix contains invalid identifiers, ordering or event types"
            )


@dataclass(frozen=True)
class Candidates:
    aid: IntArray
    source_scores: FloatArray
    source_ranks: FloatArray
    graph: FloatArray  # most recent observed event first, candidate, channel


def ranks(scores: FloatArray, aids: IntArray) -> FloatArray:
    order = np.lexsort((aids, -scores))
    result = np.zeros(scores.size, dtype=np.float32)
    positive = order[scores[order] > 0]
    result[positive] = np.arange(1, positive.size + 1, dtype=np.float32)
    return result


class FeatureEngine:
    """Shared sparse graph and historical counts; no labels are accepted."""

    def __init__(self, directory: Path) -> None:
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if (name == "history_statistics.parquet" or name.startswith("parts/")) and (
                sha256_file(directory / name) != digest
            ):
                raise ValueError(f"retrieval checksum mismatch: {name}")
        self.manifest = manifest
        self.cutoff = int(manifest["history_end"])
        frame = pl.read_parquet(directory / "history_statistics.parquet").sort("aid")
        self.stat_names = [n for n in frame.columns if n.startswith("hist_")]
        graph = pl.read_parquet(directory / "parts/part-*.parquet").sort("source_aid", "target_aid")
        high = (
            max(
                int(frame["aid"].to_numpy().max(initial=0)),
                int(graph["source_aid"].to_numpy().max(initial=0)),
                int(graph["target_aid"].to_numpy().max(initial=0)),
            )
            + 1
        )
        item_ids = frame["aid"].to_numpy().astype(np.int64)
        self.known = np.zeros(high, dtype=np.float32)
        self.known[item_ids] = 1
        self.statistics = np.zeros((high, len(self.stat_names)), dtype=np.float32)
        self.statistics[item_ids] = frame.select(self.stat_names).to_numpy()
        self.first_ts = np.full(high, self.cutoff, dtype=np.int64)
        self.last_ts = np.full(high, self.cutoff, dtype=np.int64)
        self.first_ts[item_ids] = frame["history_first_ts"].to_numpy()
        self.last_ts[item_ids] = frame["history_last_ts"].to_numpy()
        source = graph["source_aid"].to_numpy()
        target = graph["target_aid"].to_numpy()
        self.graphs = [
            csr_matrix((graph[f"{channel}_score"].to_numpy(), (source, target)), shape=(high, high))
            for channel in CHANNELS
        ]
        self.popular = []
        for action in ("clicks", "carts", "orders"):
            values = self.statistics[:, self.stat_names.index(f"hist_{action}_h24")]
            order = np.lexsort((np.arange(high), -values))[:400]
            self.popular.append(order[values[order] > 0])
        self.names = tuple(f.name for f in feature_catalog())

    def candidates(self, prefix: Prefix, budget: int = 400) -> Candidates:
        prefix.validate()
        if not 1 <= budget <= 400:
            raise ValueError("candidate budget must be in [1, 400]")
        if prefix.ts[0] < self.cutoff:
            raise ValueError("query precedes the fitted-history cutoff")
        seeds = prefix.aid[-50:][::-1]
        valid = seeds < self.known.size
        safe = np.where(valid, seeds, 0)
        rows = [graph[safe] for graph in self.graphs]
        if not valid.all():
            rows = [row.multiply(valid[:, None]).tocsr() for row in rows]
            for row in rows:
                row.eliminate_zeros()
        discovery = np.unique(
            np.concatenate([prefix.aid, *self.popular, *[r.indices for r in rows]])
        )
        inside = discovery < self.known.size
        graph_values = np.zeros((seeds.size, discovery.size, 3), dtype=np.float32)
        for i, row in enumerate(rows):
            graph_values[:, inside, i] = row[:, discovery[inside]].toarray()
        graph_values[~valid] = 0
        positions = np.arange(1, seeds.size + 1)
        recency = (1 / np.sqrt(positions)).astype(np.float32)
        scores = np.zeros((discovery.size, 5), dtype=np.float32)
        scores[:, :3] = np.einsum("scg,s->cg", graph_values, recency, optimize=False)
        # Revisit score counts all observed events, with deterministic recency weights.
        np.add.at(
            scores[:, 3],
            np.searchsorted(discovery, prefix.aid[::-1]),
            1 / np.sqrt(np.arange(1, prefix.aid.size + 1)),
        )
        for weight, popular in zip((0.1, 0.3, 0.6), self.popular, strict=True):
            scores[np.searchsorted(discovery, popular), 4] += (
                weight / np.arange(1, popular.size + 1)
            ).astype(np.float32)
        source_ranks = np.column_stack([ranks(scores[:, i], discovery) for i in range(5)])
        fusion = np.sum(
            np.where(source_ranks > 0, np.array([0.4, 0.3, 0.6, 1, 0.01]) / (20 + source_ranks), 0),
            axis=1,
        )
        # Preserve most-recent unique revisits, then fill from the common fusion policy.
        _, first = np.unique(prefix.aid[::-1], return_index=True)
        revisits = prefix.aid[::-1][np.sort(first)][:budget]
        revisit_index = np.searchsorted(discovery, revisits)
        order = np.lexsort((discovery, -fusion))
        selected = np.concatenate([revisit_index, order[~np.isin(order, revisit_index)]])[:budget]
        return Candidates(
            discovery[selected], scores[selected], source_ranks[selected], graph_values[:, selected]
        )

    def transform(
        self, prefix: Prefix, candidates: Candidates, names: tuple[str, ...] | None = None
    ) -> FloatArray:
        requested = self.names if names is None else names
        if len(set(requested)) != len(requested) or not set(requested).issubset(self.names):
            raise ValueError("feature schema is duplicated or unknown")
        wanted = set(requested)
        result: dict[str, Any] = {}
        n = candidates.aid.size

        def put(name: str, values: Any) -> None:
            if name in wanted:
                result[name] = np.broadcast_to(np.asarray(values, dtype=np.float32), (n,))

        scores = candidates.source_scores
        sranks = candidates.source_ranks
        shares = scores / np.maximum(scores.max(axis=0, initial=0), 1e-12)
        for j, source in enumerate((*CHANNELS, "revisit", "popularity")):
            for label, values in (
                ("score", scores[:, j]),
                ("rank", sranks[:, j]),
                ("present", scores[:, j] > 0),
                ("share", shares[:, j]),
            ):
                put(f"source_{source}_{label}", values)
        present = sranks > 0
        count = present.sum(axis=1)
        mean_rank = sranks.sum(axis=1) / np.maximum(count, 1)
        put("source_count", count)
        put("source_rank_best", np.where(present, sranks, 1e6).min(axis=1))
        put("source_rank_mean", mean_rank)
        put(
            "source_rank_std",
            np.sqrt(np.maximum(0, (sranks**2).sum(axis=1) / np.maximum(count, 1) - mean_rank**2)),
        )
        put("source_score_max", shares.max(axis=1))
        put("source_score_mean", shares.mean(axis=1))
        put("source_score_std", shares.std(axis=1))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            name = f"source_{CHANNELS[left]}_{CHANNELS[right]}"
            put(f"{name}_product", shares[:, left] * shares[:, right])
            put(f"{name}_difference", shares[:, left] - shares[:, right])
            put(f"{name}_ratio", shares[:, left] / (0.01 + shares[:, right]))
        age = (prefix.ts[-1] - prefix.ts[::-1]) / 3_600_000
        gaps = np.diff(prefix.ts) / 3_600_000
        reverse = prefix.aid[::-1]
        kinds = prefix.kind[::-1]
        put("query_events", reverse.size)
        put("query_unique", np.unique(reverse).size)
        put("query_duration_hours", age[-1])
        put("query_repeat_share", 1 - np.unique(reverse).size / reverse.size)
        hour = (prefix.ts[-1] / 3_600_000) % 24
        weekday = ((prefix.ts[-1] / 86_400_000) + 3) % 7
        for period, value, length in (("hour", hour, 24), ("weekday", weekday, 7)):
            put(f"query_{period}_sin", np.sin(value / length * 2 * np.pi))
            put(f"query_{period}_cos", np.cos(value / length * 2 * np.pi))
        for label, function in (
            ("mean", np.mean),
            ("std", np.std),
            ("max", np.max),
            ("median", np.median),
        ):
            put(f"query_gap_{label}", function(gaps) if gaps.size else 0)
        known = candidates.aid < self.known.size
        safe = np.where(known, candidates.aid, 0)
        history = self.statistics[safe].copy()
        history[~known] = 0
        hist = dict(zip(self.stat_names, history.T, strict=True))
        for name, values in hist.items():
            put(name, values)
        put("history_known", np.where(known, self.known[safe], 0))
        put(
            "history_last_age_hours",
            np.where(known, (prefix.ts[-1] - self.last_ts[safe]) / 3_600_000, -1),
        )
        put(
            "history_lifetime_hours",
            np.where(known, (self.last_ts[safe] - self.first_ts[safe]) / 3_600_000, 0),
        )
        equality = reverse[:, None] == candidates.aid[None, :]
        for ai, action in enumerate(ACTIONS):
            mask = np.ones(reverse.size, dtype=bool) if ai == 0 else kinds == ai - 1
            put(f"query_{action}_count", mask.sum())
            put(f"query_{action}_share", mask.mean())
            put(f"query_{action}_last_age_hours", age[mask][0] if mask.any() else -1)
            for horizon in HISTORY_HOURS:
                put(
                    f"hist_{action}_h{horizon}_share",
                    hist[f"hist_{action}_h{horizon}"] / (1 + hist[f"hist_{action}_all"]),
                )
            for short, long in ((1, 6), (6, 24), (24, 72), (72, 168), (168, 336)):
                put(
                    f"hist_{action}_trend_h{short}_h{long}",
                    hist[f"hist_{action}_h{short}"]
                    / (1 + hist[f"hist_{action}_h{long}"])
                    * (long / short),
                )
            matches = equality & mask[:, None]
            for length in LENGTHS:
                block = matches[:length]
                positions = np.arange(1, block.shape[0] + 1)[:, None]
                count = block.sum(axis=0)
                first = np.where(block, positions, 1e6).min(axis=0)
                last = np.where(block, positions, 0).max(axis=0)
                put(f"repeat_{action}_n{length}_count", count)
                put(f"repeat_{action}_n{length}_share", count / max(1, mask[:length].sum()))
                put(f"repeat_{action}_n{length}_last_position", np.where(count, first, -1))
                put(f"repeat_{action}_n{length}_first_position", np.where(count, last, -1))
                put(f"repeat_{action}_n{length}_span", np.where(count, last - first, 0))
                put(
                    f"repeat_{action}_n{length}_age_hours",
                    np.where(count, np.where(block, age[:length, None], 1e6).min(axis=0), -1),
                )
            for horizon in (1, 6, 24, 72):
                put(f"repeat_{action}_h{horizon}_decay", np.exp(-age / horizon) @ matches)
        for action in ("carts", "orders"):
            for horizon in (0, *HISTORY_HOURS):
                label = f"h{horizon}" if horizon else "all"
                put(
                    f"hist_{action}_per_click_{label}",
                    hist[f"hist_{action}_{label}"] / (1 + hist[f"hist_clicks_{label}"]),
                )
        for ci, channel in enumerate(CHANNELS):
            for ai, action in enumerate(("clicks", "carts", "orders")):
                put(f"intent_{channel}_{action}", shares[:, ci] * (prefix.kind == ai).mean())
        for action in ("carts", "orders"):
            put(
                f"intent_repeat_{action}",
                equality.sum(axis=0) * hist[f"hist_{action}_all"] / (1 + hist["hist_clicks_all"]),
            )
        graph_requested = [name for name in requested if name.startswith("graph_")]
        if graph_requested:
            block = candidates.graph
            weights = {
                "uniform": np.ones(block.shape[0], dtype=np.float32),
                "position": (1 / np.sqrt(np.arange(1, block.shape[0] + 1))).astype(np.float32),
                "hour": np.exp(-age[: block.shape[0]]).astype(np.float32),
                "six_hours": np.exp(-age[: block.shape[0]] / 6).astype(np.float32),
            }
            groups: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
            for name in graph_requested:
                _, channel, action, graph_length, *rest = name.split("_")
                weight = "_".join(rest[:-1])
                groups.setdefault((action, graph_length, weight), []).append(
                    (name, channel, rest[-1])
                )
            for (action, graph_length, weight), columns in groups.items():
                stop = min(int(graph_length[1:]), block.shape[0])
                mask = (
                    np.ones(stop, dtype=bool)
                    if action == "all"
                    else kinds[:stop] == ACTIONS.index(action) - 1
                )
                w = weights[weight][:stop] * mask
                mass = max(float(w.sum()), 1e-12)
                values = block[:stop]
                total = np.einsum("scg,s->cg", values, w, optimize=False)
                mean = total / mass
                maximum = (values * w[:, None, None]).max(axis=0)
                variance = np.maximum(
                    0, np.einsum("scg,s->cg", values**2, w, optimize=False) / mass - mean**2
                )
                derived = {"sum": total, "mean": mean, "max": maximum, "std": np.sqrt(variance)}
                for name, channel, aggregate in columns:
                    put(name, derived[aggregate][:, CHANNELS.index(channel)])
        matrix = np.column_stack([result[name] for name in requested]).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("feature matrix contains non-finite values")
        return matrix

    def catalog(self) -> list[dict[str, str]]:
        return [asdict(f) for f in feature_catalog()]
