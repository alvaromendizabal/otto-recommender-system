"""Pure, testable contracts for a fixed timing-feature replication. No cloud operations."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np

OBJECTIVES = ('clicks', 'carts', 'orders')
WEIGHTS = np.array([.1, .3, .6])


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def array_digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Symlink output rejected')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Conflicting evidence preserved: {path.name}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json(path, value):
    write_once(path, encode(value))


def save_arrays(path, **arrays):
    """Stable NPZ metadata permits byte-identical replay independent of wall time."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.hasobject or '/' in name or '..' in name:
                raise ValueError('Object arrays or unsafe names forbidden')
            binary = io.BytesIO(); np.save(binary, value, allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, binary.getvalue())
    write_once(path, buffer.getvalue())
    return sha(path)


def checked(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f'Missing or changed certified file: {path}')


def choose_cohort(all_fit_ids, excluded_ids, count=4096, seed=20260911):
    """Selection uses IDs only; no timestamps, targets, outcomes or feature values."""
    ids, excluded = np.asarray(all_fit_ids), np.asarray(excluded_ids)
    for a in (ids, excluded):
        if a.ndim != 1 or a.dtype.kind not in 'iu' or (a < 0).any() or len(set(a.tolist())) != len(a):
            raise ValueError('Cohort IDs must be distinct nonnegative integers')
    if not set(excluded.tolist()).issubset(set(ids.tolist())):
        raise ValueError('Prior cohort is not contained in fitting pool')
    ids = np.sort(ids.astype(np.int64))
    eligible = ids[~np.isin(ids, excluded)]
    if type(count) is not int or count < 1 or len(eligible) < count:
        raise ValueError('Insufficient unused fitting sessions')
    with np.errstate(over='ignore'):
        h = (eligible.astype(np.uint64) ^ np.uint64(seed)) * np.uint64(11400714819323198485)
    selected = eligible[np.argsort(h, kind='stable')[:count]]
    return np.sort(selected)


def forward_folds(ids, timestamps, embargo_ms=6*3600*1000):
    ids, q = np.asarray(ids), np.asarray(timestamps)
    if (ids.shape != q.shape or ids.ndim != 1 or len(ids) % 4 or len(ids) < 16
            or q.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
            or np.unique(ids).size != ids.size or embargo_ms < 0):
        raise ValueError('Invalid forward-fold inputs')
    order = np.lexsort((ids, q))
    n = len(ids); result = []
    for begin, stop in ((n//2, 3*n//4), (3*n//4, n)):
        valid = order[begin:stop]
        cutoff = int(q[valid].min())
        prior = order[:begin]
        train = prior[q[prior] < cutoff - embargo_ms]
        if len(train) < max(4, n//16) or np.intersect1d(train, valid).size:
            raise ValueError('Insufficient strictly embargoed training support')
        result.append({'train':train, 'valid':valid, 'cutoff':cutoff, 'purged':begin-len(train)})
    return result


def validate_prefix(aids, ts, kinds, indices, ledger, cutoff, end):
    a, t, k, ix = map(np.asarray, (aids, ts, kinds, indices))
    if (a.ndim != 1 or not a.size or any(v.shape != a.shape for v in (t, k, ix))
            or any(v.dtype.kind not in 'iu' for v in (a, t, k, ix))
            or (a < 0).any() or not np.isin(k, (0,1,2)).all()
            or not np.array_equal(ix, np.arange(len(ix))) or (np.diff(t) < 0).any()):
        raise ValueError('Malformed complete observed prefix')
    if not cutoff <= int(t[0]) <= int(t[-1]) < end:
        raise ValueError('Observed prefix outside historical role')
    if any(ledger[name] != value for name,value in (
            ('first_ts',int(t[0])), ('query_ts',int(t[-1])),
            ('observed_events',len(a)), ('observed_last_index',len(a)-1), ('period_end',end))):
        raise ValueError('Observed prefix differs from query ledger')


def target_arrays(ids, aids, q, last_index, true_counts, labels, period_end, cutoff=None):
    """Validate full targets before optional exclusive-time censoring for training."""
    ids, aids, q = np.asarray(ids), np.asarray(aids), np.asarray(q)
    if (ids.ndim!=1 or q.shape!=ids.shape or aids.ndim!=2 or len(aids)!=len(ids)
            or np.asarray(true_counts).shape!=(len(ids),3)
            or np.asarray(last_index).shape!=ids.shape):
        raise ValueError('Target arrays misaligned')
    if cutoff is not None and (q >= cutoff).any():
        raise ValueError('Training queries reach validation cutoff')
    y = np.zeros((*aids.shape,3), dtype=np.int8)
    for i,s in enumerate(ids):
        if np.unique(aids[i]).size != aids.shape[1]:
            raise ValueError('Duplicate candidate identities')
        rows=labels.get(int(s), [])
        seen=set(); truth=[[],[],[]]
        lookup={int(a):p for p,a in enumerate(aids[i])}
        for item,objective,t,index in rows:
            if objective not in (0,1,2) or item<0 or (objective,item) in seen:
                raise ValueError('Invalid or duplicate distinct target')
            seen.add((objective,item)); truth[objective].append(item)
            if not int(q[i]) <= t < period_end or index <= last_index[i]:
                raise ValueError('Target timestamp/index violates prefix boundaries')
            if (cutoff is None or t < cutoff) and item in lookup:
                y[i,lookup[item],objective]=1
        if len(truth[0])>1 or [len(t) for t in truth] != list(true_counts[i]):
            raise ValueError('Complete target counts disagree with query ledger')
    return y


def hits_at20(scores, aids, target):
    scores,aids,target=map(np.asarray,(scores,aids,target))
    if (scores.shape!=aids.shape or target.shape!=scores.shape or scores.ndim!=2
            or not np.isfinite(scores).all() or not np.isin(target,(0,1)).all()):
        raise ValueError('Invalid aligned predictions/targets')
    hits=[]
    for p,a,t in zip(scores,aids,target,strict=True):
        if np.unique(a).size!=len(a): raise ValueError('Duplicate candidate identities')
        hits.append(int(t[np.lexsort((a,-p))[:20]].sum()))
    return np.asarray(hits,dtype=np.int64)


def pooled(hits, denominator):
    h,d=map(np.asarray,(hits,denominator))
    if (h.shape!=d.shape or h.ndim!=2 or h.shape[1]!=3 or h.dtype.kind not in 'iu'
            or d.dtype.kind not in 'iu' or (h<0).any() or (d<0).any() or (d>20).any()
            or (h>d).any() or (d[:,0]>1).any()):
        raise ValueError('Invalid complete integer metric ledger')
    sums=d.sum(axis=0)
    if (sums<=0).any(): raise ValueError('Missing objective support')
    r=h.sum(axis=0)/sums
    return {'weighted_recall_at_20':float(r@WEIGHTS),
            'recall':dict(zip(OBJECTIVES,map(float,r),strict=True)),
            'hits':h.sum(axis=0).tolist(),'denominators':sums.tolist()}


def paired_interval(delta,den,fold_ids,replicates=2000,seed=20260911):
    delta,den,fold_ids=map(np.asarray,(delta,den,fold_ids))
    if (delta.shape!=den.shape or delta.ndim!=2 or delta.shape[1]!=3
            or fold_ids.shape!=(len(den),) or not np.isfinite(delta).all()
            or (abs(delta)>den).any()):
        raise ValueError('Invalid paired differences')
    groups=[np.flatnonzero(fold_ids==k) for k in np.unique(fold_ids)]
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(replicates):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        d=den[ix].sum(axis=0)
        if (d==0).any(): continue
        estimates.append(float((delta[ix].sum(axis=0)/d)@WEIGHTS))
    if len(estimates)<replicates*.9: raise ValueError('Insufficient bootstrap objective support')
    return {'descriptive_95_interval':np.quantile(estimates,[.025,.975]).tolist(),
            'replicates':len(estimates),'method':'paired session bootstrap within time fold; descriptive'}


def decision(gain,fold_gains,order_gain,interval):
    if gain<=0: return 'DO_NOT_RETAIN_NO_REPLICATION_GAIN'
    if min(fold_gains)<0 or order_gain<0: return 'UNSTABLE_DO_NOT_PROMOTE'
    if gain<.003: return 'SMALL_GAIN_CONFIRM_COST_BEFORE_MORE_COMPUTE'
    if interval[0]<=0: return 'POINT_ESTIMATE_REPLICATED_UNCERTAINTY_REMAINS'
    return 'ADVANCE_TO_SEPARATE_TEMPORAL_CONFIRMATION_NOT_PROMOTION'
