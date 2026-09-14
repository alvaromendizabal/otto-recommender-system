"""Historical adjacent-item transitions, typed by source and destination actions.

Generation uses query prefixes and pre-cutoff histories only. Scores are descriptive
session-unique transition support, not purchase probabilities or causal effects.
"""
from __future__ import annotations
import numpy as np

ACTIONS = ('clicks', 'carts', 'orders')
STATS = ('log_support', 'outgoing_share', 'direction_balance')
NAMES = tuple(f'transition_{a}_to_{b}_{s}' for a in ACTIONS for b in ACTIONS for s in STATS)
COLLAPSED_NAMES = tuple(f'transition_untyped_from_query_{a}_{s}' for a in ACTIONS for s in STATS)
SMOOTHING = 20.0


def ids_array(value, *, unique=False):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in 'iu':
        raise ValueError('One-dimensional integer IDs required')
    if a.size and (int(a.min()) < 0 or int(a.max()) >= 2**31):
        raise ValueError('IDs must fit the declared nonnegative 31-bit catalogue')
    if unique and np.unique(a).size != a.size:
        raise ValueError('Duplicate item IDs')
    return a.astype(np.int64, copy=False)


def pair_keys(source, target):
    s, t = ids_array(source), ids_array(target)
    if s.shape != t.shape:
        raise ValueError('Pair arrays are misaligned')
    return (s.astype(np.uint64) << np.uint64(32)) | t.astype(np.uint64)


def split_keys(keys):
    k = np.asarray(keys)
    if k.ndim != 1 or k.dtype != np.uint64:
        raise ValueError('Expected uint64 pair keys')
    return (k >> np.uint64(32)).astype(np.int64), (k & np.uint64(2**32-1)).astype(np.int64)


def anchors(aids, kinds):
    a = ids_array(aids); k = np.asarray(kinds)
    if k.shape != a.shape or k.dtype.kind not in 'iu' or not np.isin(k, [0,1,2]).all():
        raise ValueError('Observed action arrays are invalid')
    out = np.full(3, -1, dtype=np.int64)
    for action in range(3):
        indices = np.flatnonzero(k == action)
        if len(indices): out[action] = a[indices[-1]]
    return out


def request_keys(anchor_matrix, candidate_matrix):
    aa, cc = np.asarray(anchor_matrix), np.asarray(candidate_matrix)
    if aa.shape != (len(cc), 3) or cc.ndim != 2 or aa.dtype.kind != 'i':
        raise ValueError('Anchor/candidate request shape is invalid')
    chunks = []
    for ax, candidates in zip(aa, cc, strict=True):
        c = ids_array(candidates, unique=True)
        if (ax < -1).any(): raise ValueError('Invalid missing-anchor marker')
        for s in ax[ax >= 0]:
            other = c[c != s]
            source = np.full(other.size, s, np.int64)
            chunks.extend([pair_keys(source, other), pair_keys(other, source)])
    if not chunks: return np.empty(0, np.uint64)
    return np.unique(np.concatenate(chunks))


class TransitionIndex:
    def __init__(self, keys, typed, collapsed, source_ids, outgoing, pooled_outgoing):
        self.keys = np.asarray(keys)
        self.typed = np.asarray(typed)
        self.collapsed = np.asarray(collapsed)
        self.source_ids = ids_array(source_ids, unique=True)
        self.outgoing = np.asarray(outgoing)
        self.pooled_outgoing = np.asarray(pooled_outgoing)
        n, m = len(self.keys), len(self.source_ids)
        if (self.keys.dtype != np.uint64 or self.keys.ndim != 1
                or (self.keys[1:] <= self.keys[:-1]).any()
                or (self.source_ids[1:] <= self.source_ids[:-1]).any()
                or self.typed.shape != (n, 9) or self.collapsed.shape != (n,)
                or self.outgoing.shape != (m, 3) or self.pooled_outgoing.shape != (m,)):
            raise ValueError('Invalid transition-index shapes or ordering')
        for x in (self.typed, self.collapsed, self.outgoing, self.pooled_outgoing):
            if x.dtype.kind not in 'iu' or (x < 0).any():
                raise ValueError('Transition counts must be nonnegative integers')
        s, _ = split_keys(self.keys)
        ix = np.searchsorted(self.source_ids, s)
        if n and ((ix >= m).any() or not np.array_equal(self.source_ids[ix], s)):
            raise ValueError('Pair source is missing from denominator ledger')
        if n:
            for action in range(3):
                if (self.typed[:, action*3:(action+1)*3].sum(1) > self.outgoing[ix, action]).any():
                    raise ValueError('Typed pair support exceeds full outgoing support')
            if (self.collapsed > self.pooled_outgoing[ix]).any():
                raise ValueError('Collapsed pair support exceeds full outgoing support')
            if ((self.typed > self.collapsed[:, None]).any()
                    or (self.collapsed > self.typed.sum(1)).any()):
                raise ValueError('Typed and action-collapsed session supports disagree')

    def lookup(self, sources, targets):
        keys = pair_keys(sources, targets)
        pos = np.searchsorted(self.keys, keys)
        valid = pos < len(self.keys)
        if len(self.keys): valid &= self.keys[np.minimum(pos, len(self.keys)-1)] == keys
        typed = np.zeros((len(keys),9),np.int64); pooled = np.zeros(len(keys),np.int64)
        typed[valid] = self.typed[pos[valid]]; pooled[valid] = self.collapsed[pos[valid]]
        return typed, pooled

    def denominators(self, source):
        ix = np.searchsorted(self.source_ids, source)
        if ix == len(self.source_ids) or self.source_ids[ix] != source:
            return np.zeros(3, np.int64), 0
        return self.outgoing[ix], int(self.pooled_outgoing[ix])


def statistics(forward, reverse, total):
    f, r = np.asarray(forward,dtype=np.float64), np.asarray(reverse,dtype=np.float64)
    if f.shape != r.shape or not np.isfinite(f).all() or not np.isfinite(r).all() or (f < 0).any() or (r < 0).any() or total < 0 or (f > total).any():
        raise ValueError('Invalid directional support or source denominator')
    return np.column_stack((np.log1p(f), f/(float(total)+SMOOTHING), (f-r)/(f+r+SMOOTHING))).astype(np.float32)


def transform(index, anchor_ids, candidate_ids):
    aa = np.asarray(anchor_ids)
    if aa.shape != (3,) or aa.dtype.kind != 'i' or (aa < -1).any():
        raise ValueError('Three query-action anchors required')
    aids = ids_array(candidate_ids, unique=True)
    typed = np.zeros((len(aids),len(NAMES)),np.float32)
    collapsed = np.zeros((len(aids),len(COLLAPSED_NAMES)),np.float32)
    for sk, s in enumerate(aa):
        if s < 0: continue
        sources = np.full(len(aids),s,np.int64)
        ff, pp = index.lookup(sources,aids); rr, reverse_pooled = index.lookup(aids,sources)
        totals, pooled_total = index.denominators(int(s))
        for tk in range(3):
            typed[:,(sk*3+tk)*3:(sk*3+tk+1)*3] = statistics(ff[:,sk*3+tk],rr[:,tk*3+sk],totals[sk])
        collapsed[:,sk*3:(sk+1)*3] = statistics(pp,reverse_pooled,pooled_total)
        # Cross-item representation; own-item evidence is deliberately absent.
        typed[aids==s,sk*9:(sk+1)*9]=0;collapsed[aids==s,sk*3:(sk+1)*3]=0
    if not np.isfinite(typed).all() or not np.isfinite(collapsed).all():
        raise ValueError('Transition features are not finite')
    return typed, collapsed
