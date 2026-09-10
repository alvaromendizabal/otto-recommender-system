"""Observed shopping progression, gap-defined episodes and normalized graph evidence.

These transforms accept only an observed prefix, candidate identifiers and certified
historical graphs. Cart flags describe recorded actions, never inferred abandonment.
Graph normalization uses the retained sparse graph, not unobserved full-graph counts.
"""

from __future__ import annotations

import numpy as np

from otto_recsys.research.features import Feature, Prefix
from otto_recsys.research.graph_signals import CHANNELS, FAMILIES, GraphSignals

DOMAIN_FAMILIES = ("funnel", "episode", "raw_graph", "normalized_graph")
EPISODE_GAPS = (1800, 7200)
POOLS = ("last", "recent_unique", "purchase_unique")
FUNNEL_FIELDS = (
    "seen",
    "full_count_share",
    "distinct_actions",
    "last_unique_reciprocal_rank",
    "last_age_log_hours",
    "events_since_last_log",
    "observed_span_fraction",
    "cart_without_later_order",
    "click_after_cart",
    "click_after_order",
    "clicks_since_cart_log",
    "carts_since_order_log",
    "trailing_run_share",
    "prefix_item_entropy",
    "prefix_item_concentration",
    "prefix_switch_share",
    *(f"last_action_{kind}" for kind in range(3)),
    *(f"previous_action_{kind}" for kind in range(3)),
    *(
        f"transition_{left}_{right}_{stat}"
        for left in range(3)
        for right in range(3)
        for stat in ("count_log", "mean_gap_log_hours")
    ),
)
EPISODE_FIELDS = (
    "seen",
    "count_share",
    "distinct_actions",
    "cart_without_later_order",
    "previous_episode_seen",
    "earlier_episodes_log",
    "last_age_log_hours",
    "query_events_log",
    "query_unique_log",
    "query_duration_log_hours",
    "query_episode_count_log",
    "query_boundary_gap_log_hours",
)


def feature_names(family: str) -> tuple[str, ...]:
    if family == "funnel":
        return tuple(f"domain_funnel_{name}" for name in FUNNEL_FIELDS)
    if family == "episode":
        return tuple(
            f"domain_episode_s{gap}_{name}" for gap in EPISODE_GAPS for name in EPISODE_FIELDS
        )
    if family in ("raw_graph", "normalized_graph"):
        modes = ("raw",) if family == "raw_graph" else ("row", "degree")
        return tuple(
            f"domain_norm_{graph}_{channel}_{normalization}_{pool}_{reduction}"
            for graph in FAMILIES
            for channel in CHANNELS
            for normalization in modes
            for pool in POOLS
            for reduction in ("mean", "max")
        )
    raise ValueError("unknown domain feature family")


def feature_catalog() -> list[Feature]:
    result = []
    for family in DOMAIN_FAMILIES:
        for name in feature_names(family):
            result.append(
                Feature(
                    name,
                    family,
                    name.removeprefix("domain_").replace("_", " "),
                    "fixed historical graph cutoff and observed prefix"
                    if family in ("raw_graph", "normalized_graph")
                    else "observed prefix only",
                )
            )
    return result


def validate_candidates(prefix: Prefix, aids: np.ndarray) -> None:
    prefix.validate()
    if (
        aids.ndim != 1
        or not np.issubdtype(aids.dtype, np.integer)
        or (aids < 0).any()
        or np.unique(aids).size != aids.size
    ):
        raise ValueError("candidate identifiers must be unique nonnegative integers")


def _pending(kinds: np.ndarray) -> bool:
    carts, orders = np.flatnonzero(kinds == 1), np.flatnonzero(kinds == 2)
    return bool(carts.size and (not orders.size or carts[-1] > orders[-1]))


def funnel_features(prefix: Prefix, aids: np.ndarray) -> np.ndarray:
    """Full-prefix item subsequences retain action order and repeated-item support."""
    validate_candidates(prefix, aids)
    result = np.zeros((aids.size, len(FUNNEL_FIELDS)), dtype=np.float32)
    fields = {name: index for index, name in enumerate(FUNNEL_FIELDS)}
    _, counts = np.unique(prefix.aid, return_counts=True)
    probability = counts / counts.sum()
    context = {
        "prefix_item_entropy": -np.sum(probability * np.log(probability)),
        "prefix_item_concentration": np.sum(probability**2),
        "prefix_switch_share": np.mean(prefix.aid[1:] != prefix.aid[:-1])
        if prefix.aid.size > 1
        else 0.0,
    }
    for name, value in context.items():
        result[:, fields[name]] = value
    recency = {int(aid): i + 1 for i, aid in enumerate(dict.fromkeys(prefix.aid[::-1]))}
    candidate_rows = {int(aid): row for row, aid in enumerate(aids)}
    for aid in np.unique(prefix.aid):
        if int(aid) not in candidate_rows:
            continue
        row = candidate_rows[int(aid)]
        positions = np.flatnonzero(prefix.aid == aid)
        if not positions.size:
            continue
        kinds, times = prefix.kind[positions], prefix.ts[positions]
        cart, order, click = (np.flatnonzero(kinds == kind) for kind in (1, 2, 0))
        duration = max(1, int(prefix.ts[-1] - prefix.ts[0]))
        trailing = 0
        for item in prefix.aid[::-1]:
            if item != aid:
                break
            trailing += 1
        values = {
            "seen": 1.0,
            "full_count_share": positions.size / prefix.aid.size,
            "distinct_actions": np.unique(kinds).size,
            "last_unique_reciprocal_rank": 1 / recency[int(aid)],
            "last_age_log_hours": np.log1p((prefix.ts[-1] - times[-1]) / 3_600_000),
            "events_since_last_log": np.log1p(prefix.aid.size - 1 - positions[-1]),
            "observed_span_fraction": (times[-1] - times[0]) / duration,
            "cart_without_later_order": float(_pending(kinds)),
            "click_after_cart": float(bool(click.size and cart.size and click[-1] > cart[-1])),
            "click_after_order": float(bool(click.size and order.size and click[-1] > order[-1])),
            "clicks_since_cart_log": np.log1p(np.sum(click > cart[-1])) if cart.size else 0,
            "carts_since_order_log": np.log1p(np.sum(cart > order[-1])) if order.size else 0,
            "trailing_run_share": trailing / prefix.aid.size,
            f"last_action_{kinds[-1]}": 1.0,
        }
        if kinds.size > 1:
            values[f"previous_action_{kinds[-2]}"] = 1.0
        for left in range(3):
            for right in range(3):
                mask = (kinds[:-1] == left) & (kinds[1:] == right)
                values[f"transition_{left}_{right}_count_log"] = np.log1p(mask.sum())
                values[f"transition_{left}_{right}_mean_gap_log_hours"] = (
                    np.log1p(np.diff(times)[mask].mean() / 3_600_000) if mask.any() else 0.0
                )
        for name, value in values.items():
            result[row, fields[name]] = value
    return result


def episode_features(prefix: Prefix, aids: np.ndarray) -> np.ndarray:
    """Event timestamps are milliseconds; gap thresholds are fixed in seconds."""
    validate_candidates(prefix, aids)
    blocks = []
    for threshold in EPISODE_GAPS:
        episode = np.r_[0, np.cumsum(np.diff(prefix.ts) > threshold * 1000)]
        last = int(episode[-1])
        current = np.flatnonzero(episode == last)
        block = np.zeros((aids.size, len(EPISODE_FIELDS)), dtype=np.float32)
        context = (
            np.log1p(current.size),
            np.log1p(np.unique(prefix.aid[current]).size),
            np.log1p((prefix.ts[-1] - prefix.ts[current[0]]) / 3_600_000),
            np.log1p(last + 1),
            np.log1p((prefix.ts[current[0]] - prefix.ts[current[0] - 1]) / 3_600_000)
            if current[0]
            else 0.0,
        )
        block[:, 7:] = context
        candidate_rows = {int(aid): row for row, aid in enumerate(aids)}
        for aid in np.unique(prefix.aid):
            if int(aid) not in candidate_rows:
                continue
            row = candidate_rows[int(aid)]
            all_positions = np.flatnonzero(prefix.aid == aid)
            positions = all_positions[episode[all_positions] == last]
            block[row, 4] = float(bool(last and (episode[all_positions] == last - 1).any()))
            block[row, 5] = np.log1p(
                np.unique(episode[all_positions][episode[all_positions] < last]).size
            )
            if positions.size:
                block[row, :4] = (
                    1,
                    positions.size / current.size,
                    np.unique(prefix.kind[positions]).size,
                    float(_pending(prefix.kind[positions])),
                )
                block[row, 6] = np.log1p((prefix.ts[-1] - prefix.ts[positions[-1]]) / 3_600_000)
        blocks.append(block)
    return np.column_stack(blocks)


class NormalizedGraphSignals:
    """Normalize certified retained-edge mass with fixed additive support of one."""

    def __init__(self, graphs: dict[str, GraphSignals]) -> None:
        if set(graphs) != set(FAMILIES) or len({g.cutoff for g in graphs.values()}) != 1:
            raise ValueError("both graph families must share one historical cutoff")
        self.graphs = graphs
        self.mass = {
            family: [
                (
                    np.asarray(g.sum(axis=1)).ravel().astype(np.float64),
                    np.asarray(g.sum(axis=0)).ravel().astype(np.float64),
                )
                for g in graphs[family].graphs
            ]
            for family in FAMILIES
        }

    def transform(self, prefix: Prefix, aids: np.ndarray) -> dict[str, np.ndarray]:
        validate_candidates(prefix, aids)
        pools = (
            prefix.aid[-1:],
            np.asarray(list(dict.fromkeys(prefix.aid[::-1]))[:20], dtype=np.int64),
            np.asarray(list(dict.fromkeys(prefix.aid[prefix.kind > 0][::-1]))[:20], dtype=np.int64),
        )
        result: dict[str, list[np.ndarray]] = {"raw_graph": [], "normalized_graph": []}
        for family in FAMILIES:
            source = self.graphs[family]
            if prefix.ts[0] < source.cutoff:
                raise ValueError("observed prefix precedes the historical graph cutoff")
            for graph, (row_mass, column_mass) in zip(
                source.graphs, self.mass[family], strict=True
            ):
                matrices = []
                for seeds in pools:
                    valid_seed, valid_aid = seeds < source.high, aids < source.high
                    raw = np.zeros((seeds.size, aids.size), dtype=np.float64)
                    source_mass, target_mass = np.zeros(seeds.size), np.zeros(aids.size)
                    if valid_seed.any() and valid_aid.any():
                        raw[np.ix_(valid_seed, valid_aid)] = graph[seeds[valid_seed]][
                            :, aids[valid_aid]
                        ].toarray()
                    source_mass[valid_seed] = row_mass[seeds[valid_seed]]
                    target_mass[valid_aid] = column_mass[aids[valid_aid]]
                    matrices.append((raw, source_mass, target_mass))
                for normalization in ("raw", "row", "degree"):
                    for raw, source_mass, target_mass in matrices:
                        denominator = source_mass[:, None] + 1
                        if normalization == "degree":
                            denominator = np.sqrt(denominator * (target_mass[None, :] + 1))
                        normalized = raw if normalization == "raw" else raw / denominator
                        group = "raw_graph" if normalization == "raw" else "normalized_graph"
                        result[group].extend(
                            (
                                normalized.mean(axis=0) if raw.shape[0] else np.zeros(aids.size),
                                normalized.max(axis=0, initial=0),
                            )
                        )
        output = {
            group: np.column_stack(values).astype(np.float32) for group, values in result.items()
        }
        if any(not np.isfinite(values).all() for values in output.values()):
            raise ValueError("domain graph features must be finite")
        return output
