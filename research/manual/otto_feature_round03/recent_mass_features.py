"""Candidate affinity mass by observed-seed age, independent of target labels.

Frozen historical graphs are supplied by the caller. A value of -1 means there
is no positive other-item evidence; 0 means evidence exists but is all older.
"""
from __future__ import annotations
import numpy as np
from intent_features import FAMILIES, CHANNELS, POOLS, ids_array, seed_pools

WINDOWS_MINUTES = (5, 30, 120)
NAMES = tuple(f'recent_mass_{f}_{c}_{p}_m{m}' for f in FAMILIES
              for c in CHANNELS for p in POOLS for m in WINDOWS_MINUTES)


def summarize_mass(raw, seeds, candidates, ages_hours):
    seeds, candidates = ids_array(seeds, True), ids_array(candidates, True)
    x = np.asarray(raw, dtype=np.float64).copy()
    age = np.asarray(ages_hours, dtype=np.float64)
    if (x.shape != (len(seeds), len(candidates)) or age.shape != seeds.shape
            or not np.isfinite(x).all() or (x < 0).any()
            or not np.isfinite(age).all() or (age < 0).any()):
        raise ValueError('Invalid aligned affinity or age')
    x[seeds[:, None] == candidates[None, :]] = 0
    mass = x.sum(axis=0)
    if not np.isfinite(mass).all():
        raise ValueError('Affinity sum overflow')
    out = np.full((len(candidates), 3), -1., dtype=np.float64)
    for j, minutes in enumerate(WINDOWS_MINUTES):
        recent = x[age <= minutes / 60.].sum(axis=0)
        np.divide(recent, mass, out=out[:, j], where=mass > 0)
    if not np.isfinite(out).all() or (out < -1).any() or (out > 1 + 1e-12).any():
        raise ValueError('Invalid recent mass ratio')
    out = np.minimum(out, 1).astype(np.float32)
    validate_values(out, windows=3)
    return out


def validate_values(x, windows=36):
    x = np.asarray(x)
    if x.ndim < 2 or x.shape[-1] != windows or windows % 3:
        raise ValueError('Recent mass schema mismatch')
    if not np.isfinite(x).all() or not (((x >= 0) & (x <= 1)) | (x == -1)).all():
        raise ValueError('Recent mass must be -1 or within [0, 1]')
    grouped = x.reshape(-1, windows // 3, 3)
    missing = grouped == -1
    if (missing.any(-1) != missing.all(-1)).any() or (np.diff(grouped, axis=-1) < -1e-7).any():
        raise ValueError('Window nesting or missingness inconsistent')


def zero_ablation(x):
    validate_values(x)
    return np.where(np.asarray(x) == -1, 0., x).astype(np.float32)


def transform(aids, timestamps, kinds, candidates, graphs, cutoff_ms):
    candidates, t = ids_array(candidates, True), ids_array(timestamps)
    if not len(t) or int(t[0]) < int(cutoff_ms) or set(graphs) != set(FAMILIES):
        raise ValueError('Temporal graph contract or empty prefix')
    pools = seed_pools(aids, t, kinds, int(t[-1]))
    blocks = []
    for family in FAMILIES:
        if len(graphs[family]) != 3:
            raise ValueError('Three graph channels required')
        for graph in graphs[family]:
            if graph.shape[0] != graph.shape[1]:
                raise ValueError('Graph must be square')
            for pool in POOLS:
                seeds, age = pools[pool]
                raw = np.zeros((len(seeds), len(candidates)), dtype=np.float64)
                si, ci = seeds < graph.shape[0], candidates < graph.shape[1]
                if si.any() and ci.any():
                    raw[np.ix_(si, ci)] = graph[seeds[si]][:, candidates[ci]].toarray()
                blocks.append(summarize_mass(raw, seeds, candidates, age))
    output = np.column_stack(blocks)
    validate_values(output)
    return output


def catalog():
    return [{'name':n, 'family':'recent_affinity_mass',
             'definition':'Positive graph weight from eligible observed seeds aged <= window / all eligible positive graph weight',
             'availability':'Observed prefix and graph built strictly before Aug-16 cutoff',
             'missing':-1, 'zero':'Support exists, but all supporting seeds are older than window',
             'self_edges':'Excluded', 'seed_rule':'Last occurrence of up to 20 distinct items; pending cart is observed cart-after-order state only',
             'screening':'Full candidate pools; diagnostics on fold training IDs only; no automatic selection'}
            for n in NAMES]
