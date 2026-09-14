"""Forty-eight observed-prefix/certified-graph features. No targets are accepted.

These are proposed explicit summaries, not measured winner features. Cart state
means 'cart observed after latest observed order', not actual inventory/abandonment.
"""
from __future__ import annotations
import numpy as np

FAMILIES = ('symmetric', 'forward')
CHANNELS = ('time', 'cart', 'order')
POOLS = ('recent_unique', 'pending_cart')
STATS = ('coverage', 'entropy', 'mean_age_log_hours', 'strongest_age_log_hours')
NAMES = tuple(f'intent_support_{f}_{c}_{p}_{s}' for f in FAMILIES
              for c in CHANNELS for p in POOLS for s in STATS)
SUPPORT = tuple(i for i, n in enumerate(NAMES) if n.endswith(('_coverage', '_entropy')))
TIMING = tuple(i for i in range(48) if i not in SUPPORT)
HOUR_MS = 3_600_000


def ids_array(a, unique=False):
    a = np.asarray(a)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or (a.size and
            (int(a.min()) < 0 or int(a.max()) > np.iinfo(np.int64).max)):
        raise ValueError('Expected nonnegative one-dimensional int64-representable IDs')
    if unique and len(np.unique(a)) != len(a):
        raise ValueError('Duplicate candidate IDs')
    return a.astype(np.int64, copy=False)


def seed_pools(aids, ts, kinds, query_ts):
    """Return at most 20 distinct seeds per pool, ordered by last observed occurrence."""
    aids = ids_array(aids)
    ts, kinds = ids_array(ts), ids_array(kinds)
    if (not len(aids) or ts.shape != aids.shape or kinds.shape != aids.shape
            or ts.dtype.kind not in 'iu' or kinds.dtype.kind not in 'iu'
            or (np.diff(ts) < 0).any() or not np.isin(kinds, (0, 1, 2)).all()
            or int(ts[-1]) != int(query_ts)):
        raise ValueError('Malformed prefix, future event, or wrong query cutoff')
    latest, carts, orders = {}, {}, {}
    for i, (aid, kind) in enumerate(zip(aids, kinds, strict=True)):
        item = int(aid)
        latest[item] = i
        if kind == 1:
            carts[item] = i
        if kind == 2:
            orders[item] = i
    recent = sorted(latest, key=lambda a: latest[a], reverse=True)
    pending = [a for a in recent if carts.get(a, -1) > orders.get(a, -1)]
    result = {}
    for pool, items in zip(POOLS, (recent[:20], pending[:20]), strict=True):
        result[pool] = (np.asarray(items, dtype=np.int64),
                        np.asarray([query_ts - int(ts[latest[a]]) for a in items],
                                   dtype=np.float64) / HOUR_MS)
    return result


def summarize_support(raw, seeds, candidates, ages):
    """Summarize other-item evidence, including valid zero/missing support.

    Denominator = number of distinct pool seeds other than this candidate.
    Unknown catalog seeds have zero edges but remain in this denominator.
    Entropy is normalized by log(max(2, number of eligible other-item seeds)).
    The strongest-support age ties go to the most recent seed (pool order).
    """
    seeds, candidates = ids_array(seeds, True), ids_array(candidates, True)
    x, age = np.asarray(raw, dtype=np.float64).copy(), np.asarray(ages, dtype=np.float64)
    if (x.shape != (len(seeds), len(candidates)) or age.shape != seeds.shape
            or not np.isfinite(x).all() or (x < 0).any()
            or not np.isfinite(age).all() or (age < 0).any()):
        raise ValueError('Invalid graph affinity or age')
    out = np.zeros((len(candidates), 4), dtype=np.float64)
    if not len(seeds):
        return out.astype(np.float32)
    other = seeds[:, None] != candidates[None, :]
    x[~other] = 0
    eligible = other.sum(axis=0)
    out[:, 0] = np.divide((x > 0).sum(axis=0), eligible,
                           out=np.zeros(len(candidates)), where=eligible > 0)
    mass = x.sum(axis=0)
    p = np.divide(x, mass[None, :], out=np.zeros_like(x), where=mass[None, :] > 0)
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    out[:, 1] = -(p * logs).sum(axis=0) / np.log(np.maximum(2, eligible))
    age_log = np.log1p(age)
    out[:, 2] = (p * age_log[:, None]).sum(axis=0)
    strongest = x.argmax(axis=0)
    out[:, 3] = np.where(mass > 0, age_log[strongest], 0)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite derived feature')
    return out.astype(np.float32)


def transform(aids, ts, kinds, candidates, graphs, cutoff_ms):
    """graphs maps family to three certified CSR matrices; unknown IDs get zero edges."""
    candidates = ids_array(candidates, True)
    ts = ids_array(ts)
    if not len(ts):
        raise ValueError('Empty observed timestamp prefix')
    if set(graphs) != set(FAMILIES) or int(ts[0]) < int(cutoff_ms):
        raise ValueError('Wrong graph families or observed prefix precedes history cutoff')
    pools = seed_pools(aids, ts, kinds, int(ts[-1]))
    blocks = []
    for family in FAMILIES:
        if len(graphs[family]) != 3:
            raise ValueError('Three graph channels required')
        for graph in graphs[family]:
            if graph.shape[0] != graph.shape[1]:
                raise ValueError('Graph is not square')
            for pool in POOLS:
                seeds, age = pools[pool]
                raw = np.zeros((len(seeds), len(candidates)), dtype=np.float64)
                a, b = seeds < graph.shape[0], candidates < graph.shape[1]
                if a.any() and b.any():
                    raw[np.ix_(a, b)] = graph[seeds[a]][:, candidates[b]].toarray()
                blocks.append(summarize_support(raw, seeds, candidates, age))
    out = np.column_stack(blocks)
    if out.shape != (len(candidates), 48):
        raise ValueError('Feature schema mismatch')
    return out


def catalog():
    result = []
    for i, name in enumerate(NAMES):
        result.append({'name': name, 'group': 'support' if i in SUPPORT else 'timing',
                       'availability': 'certified Aug-16 history + observed prefix only',
                       'self_edges': 'excluded', 'candidate_policy': 'unchanged 400',
                       'empty_support': 'zero; coverage indicates whether evidence exists'})
    return result
