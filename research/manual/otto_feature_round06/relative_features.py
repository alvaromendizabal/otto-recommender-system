"""Label-blind graph-affinity context inside repeat/discovery candidate strata.

No learned parameters, labels, I/O, or additional historical data. Group membership
uses the candidate's presence anywhere in its complete observed session prefix.
"""
from __future__ import annotations
import numpy as np

SIGNALS = (
    'graph_time_all_n1_uniform_sum',
    'graph_cart_all_n5_uniform_max',
    'domain_norm_symmetric_time_row_recent_unique_mean',
    'domain_norm_forward_time_row_recent_unique_mean',
    'domain_norm_symmetric_order_row_purchase_unique_mean',
    'domain_norm_symmetric_order_row_last_mean',
)
SEEN_SIGNAL = 'domain_funnel_full_count_share'
NAMES = tuple('repeat_context_' + s + '_' + t for s in SIGNALS
              for t in ('positive_midrank', 'affinity_share'))
GLOBAL_NAMES = tuple('global_context_' + s + '_' + t for s in SIGNALS
                    for t in ('positive_midrank', 'affinity_share'))


def extract(base: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base)
    if base.ndim != 2 or base.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Aligned unique base schema required')
    if not set(SIGNALS + (SEEN_SIGNAL,)).issubset(names):
        raise ValueError('A frozen graph-affinity or complete-prefix presence input is missing')
    seen_share = base[:, names.index(SEEN_SIGNAL)]
    if not np.isfinite(seen_share).all() or (seen_share < 0).any() or (seen_share > 1).any():
        raise ValueError('Full-prefix count share must be finite in [0,1]')
    return base[:, [names.index(s) for s in SIGNALS]].astype(np.float64), seen_share > 0


def _rank_and_share(v: np.ndarray) -> np.ndarray:
    """Positive-aware ascending midrank and mass share within the given group."""
    result = np.zeros((len(v), 2), np.float64)
    if not len(v):
        return result
    _, inverse, count = np.unique(v, return_inverse=True, return_counts=True)
    left = np.cumsum(count) - count
    mid = left + (count + 1) / 2
    result[:, 0] = np.where(v > 0, mid[inverse] / len(v), 0)
    peak = float(v.max(initial=0))
    scaled = v / peak if peak > 0 else np.zeros_like(v)
    total = float(np.sort(scaled).sum())
    result[:, 1] = scaled / total if total > 0 else 0
    return result


def context_transforms(values: np.ndarray, seen: np.ndarray, aids: np.ndarray,
                       expected_candidates: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Twelve repeat-stratum and twelve global context features, before sampling.

    Repeat group is the set of all candidates observed anywhere in the prefix;
    discovery group is its complement. For each group and each nonnegative score,
    positive values receive their average 1-based ascending rank / group size,
    zeros map to zero; share is score / group total (zero if there is no mass).
    The ablation uses the identical operations without partitioning by seen status.
    An empty group contributes nothing; an all-seen/unseen query matches ablation.
    """
    x, s, a = np.asarray(values), np.asarray(seen), np.asarray(aids)
    if (expected_candidates < 2 or x.shape != (expected_candidates, len(SIGNALS))
            or a.shape != (expected_candidates,) or a.dtype.kind not in 'iu'
            or (a < 0).any() or np.unique(a).size != expected_candidates
            or x.dtype.kind not in 'fiu' or not np.isfinite(x).all() or (x < 0).any()
            or s.shape != a.shape or s.dtype.kind != 'b'):
        raise ValueError('Complete unique candidate pool, boolean prefix membership, finite nonnegative affinities required')
    x = x.astype(np.float64, copy=False)
    primary = np.empty((len(a), len(NAMES)), np.float64)
    ablation = np.empty_like(primary)
    for j in range(len(SIGNALS)):
        ablation[:, 2*j:2*j+2] = _rank_and_share(x[:, j])
        for mask in (s, ~s):
            primary[mask, 2*j:2*j+2] = _rank_and_share(x[mask, j])
    answer = primary.astype(np.float32), ablation.astype(np.float32)
    if any(not np.isfinite(t).all() for t in answer):
        raise ValueError('Context transform produced non-finite values')
    return answer


def transform(base: np.ndarray, names: list[str], aids: np.ndarray,
              expected_candidates: int = 400) -> tuple[np.ndarray, np.ndarray]:
    values, seen = extract(base, names)
    return context_transforms(values, seen, aids, expected_candidates)


def coverage_statistics(truth: np.ndarray, denominator: np.ndarray,
                        achieved_hits: np.ndarray) -> dict:
    """Official-metric capped candidate ceiling, not uncapped recall@400.

    Includes targets absent from candidates via the unchanged complete denominator.
    Keeps the ceil@20 cap, even when a session has more than twenty true targets.
    """
    from core import pooled
    y, d, h = map(np.asarray, (truth, denominator, achieved_hits))
    if y.ndim != 3 or y.shape[2] != 3 or d.shape != (len(y), 3) or h.shape != d.shape:
        raise ValueError('Misaligned candidate-coverage arrays')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Binary candidate relevance required')
    oracle = np.minimum(y.sum(axis=1), 20).astype(np.int64)
    if (oracle > d).any() or (h > oracle).any():
        raise ValueError('Achieved hits or candidate ceiling violates complete denominator')
    o, c = pooled(oracle, d), pooled(h, d)
    return {'candidate_oracle_at20': o, 'achieved_control': c,
            'weighted_missing_candidate_gap': 1 - o['weighted_recall_at_20'],
            'weighted_within_candidate_ranking_gap': o['weighted_recall_at_20'] - c['weighted_recall_at_20'],
            'metric_capped_not_uncapped_recall400': True,
            'causal_attribution': False, 'new_model_fits': 0}
