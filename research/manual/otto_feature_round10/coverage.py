import numpy as np
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
