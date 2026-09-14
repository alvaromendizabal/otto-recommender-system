"""Pure latent-affinity feature functions. No labels, file I/O or model fitting.

Eight summary statistics x three destination actions = 24 columns. Missing
embeddings and candidate-self anchors are explicitly masked. Signed cosine
values are retained; no arbitrary clipping to positive similarities.
"""
from __future__ import annotations
import numpy as np

KINDS = ('clicks', 'carts', 'orders')
MEASURES = ('latest_cosine', 'maximum_cosine', 'mean_cosine', 'cosine_std',
            'recency_weighted_cosine', 'centroid_cosine',
            'latest_minus_older_cosine', 'available_anchor_fraction')
PREFIXES = {11: ('idf_session_latent', 'unweighted_session_latent'),
            12: ('forward_latent', 'unsigned_latent')}


def names(round_no: int, ablation: bool = False) -> tuple[str, ...]:
    prefix = PREFIXES[round_no][int(ablation)]
    return tuple(f'{prefix}_{kind}_{measure}' for kind in KINDS for measure in MEASURES)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite two-dimensional vectors required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12).astype(np.float32)


def positions(vocabulary: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vocabulary, values = np.asarray(vocabulary), np.asarray(values)
    if vocabulary.ndim != 1 or values.ndim != 1 or vocabulary.dtype.kind not in 'iu' or values.dtype.kind not in 'iu':
        raise ValueError('Integer item vectors required')
    if not len(vocabulary) or (np.diff(vocabulary) <= 0).any() or (vocabulary < 0).any() or (values < 0).any():
        raise ValueError('Sorted distinct nonnegative vocabulary required')
    ix = np.searchsorted(vocabulary, values)
    safe = np.minimum(ix, len(vocabulary) - 1)
    return safe, (ix < len(vocabulary)) & (vocabulary[safe] == values)


def affinity_summaries(anchors, candidates, vocabulary, query_vectors, target_vectors):
    """Aggregate in one shared latent basis, removing candidate-self anchors.

    query_vectors/target_vectors are (3, vocabulary, latent dimension). Candidate
    order is never a signal. Anchors are distinct items, most recent first.
    """
    anchors, candidates = np.asarray(anchors), np.asarray(candidates)
    if not 1 <= len(anchors) <= 4 or len(np.unique(anchors)) != len(anchors):
        raise ValueError('One to four distinct ordered anchors required')
    if len(candidates) == 0 or len(np.unique(candidates)) != len(candidates):
        raise ValueError('Distinct candidate pool required')
    q, t = np.asarray(query_vectors), np.asarray(target_vectors)
    if q.ndim != 3 or q.shape != t.shape or q.shape[0] != 3 or q.shape[1] != len(vocabulary):
        raise ValueError('Aligned three-action latent spaces required')
    qi, qfound = positions(vocabulary, anchors)
    ci, cfound = positions(vocabulary, candidates)
    recency = 1.0 / (np.arange(len(anchors)) + 1.0)
    nonself = candidates[:, None] != anchors[None, :]
    out = np.zeros((len(candidates), 24), np.float32)
    for kind in range(3):
        qv = q[kind, qi].astype(np.float64)
        cv = t[kind, ci].astype(np.float64)
        # The loader validates the whole basis once. Per-query checks inspect only used rows.
        if not np.isfinite(qv).all() or not np.isfinite(cv).all():
            raise ValueError('Nonfinite selected representation')
        qa = qfound & (np.linalg.norm(qv, axis=1) > 1e-12)
        ca = cfound & (np.linalg.norm(cv, axis=1) > 1e-12)
        valid = nonself & qa[None, :] & ca[:, None]
        score = np.clip(cv @ qv.T, -1.0, 1.0)
        count = valid.sum(1)
        denom = np.maximum(count, 1)
        sums = np.where(valid, score, 0).sum(1)
        mean = sums / denom
        maximum = np.where(count > 0, np.where(valid, score, -2.).max(1), 0)
        std = np.sqrt(np.maximum(0., np.where(valid, (score - mean[:, None]) ** 2, 0).sum(1) / denom))
        latest = np.where(valid[:, 0], score[:, 0], 0.)
        weights = valid * recency[None, :]
        weighted = (weights * score).sum(1) / np.maximum(weights.sum(1), 1e-12)
        centroid = weights @ qv
        cn = np.linalg.norm(centroid, axis=1)
        cosine = np.divide((centroid * cv).sum(1), cn, out=np.zeros(len(candidates)), where=cn > 1e-12)
        old_count = valid[:, 1:].sum(1)
        old_mean = np.where(valid[:, 1:], score[:, 1:], 0).sum(1) / np.maximum(old_count, 1)
        difference = np.where(valid[:, 0] & (old_count > 0), latest - old_mean, 0.)
        coverage = count / np.maximum(nonself.sum(1), 1)
        out[:, kind*8:(kind+1)*8] = np.column_stack((latest, maximum, mean, std, weighted,
                                                   np.clip(cosine, -1, 1), difference, coverage))
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite latent feature')
    return out
