"""Seventy-five strict as-of demand features; no target or graph inputs.

Matches the repository demand feature definitions while making full-catalogue
prior totals explicit. Candidate-only storage must not redefine the global prior.
"""
from __future__ import annotations
import numpy as np

HOUR = 3_600_000
ACTIONS = ('clicks', 'carts', 'orders')
WINDOWS = (1, 6, 24, 72, 168)
PAIRS = ((1, 24), (6, 24), (24, 168))

def names():
    result=[]
    for a in ACTIONS:
        for h in WINDOWS:
            result += [f'asof_{a}_h{h}_{s}' for s in ('log_count','candidate_percentile','candidate_mass_share')]
        result += [f'asof_{a}_last_seen',f'asof_{a}_last_age_log_hours']
        for short,long in PAIRS:
            result += [f'asof_{a}_trend_h{short}_h{long}',f'asof_{a}_percentile_delta_h{short}_h{long}']
    result += [f'asof_action_mix_{a}_h{h}' for h in (24,168) for a in ACTIONS]
    return tuple(result)
NAMES=names()

def percentile(x):
    if not len(x):return np.empty(0,dtype=np.float32)
    if len(x)==1:return np.full(1,.5,dtype=np.float32)
    v=np.sort(x)
    return ((np.searchsorted(v,x,'left')+np.searchsorted(v,x,'right')-1)/2/(len(x)-1)).astype(np.float32)

def transform(counts,last,global_counts,*,cutoff,query_ts,prior_strength=20.0):
    """Complete candidate pool only. Windows are [cutoff-h, cutoff), in milliseconds."""
    c,l,g=map(np.asarray,(counts,last,global_counts))
    if c.ndim!=3 or c.shape[1:]!=(5,3) or l.shape!=(len(c),3) or g.shape!=(5,3):
        raise ValueError('Demand snapshot shape mismatch')
    if c.dtype.kind not in 'iu' or l.dtype.kind not in 'iu' or g.dtype.kind not in 'iu':
        raise ValueError('Exact integer counts and timestamps required')
    # Convert only after checking representability; unsigned differences must not wrap.
    for arr in (c, l, g):
        if arr.size and int(arr.max()) > np.iinfo(np.int64).max:
            raise ValueError('Snapshot integers exceed supported range')
    c, l, g = (arr.astype(np.int64, copy=False) for arr in (c, l, g))
    if type(cutoff) not in (int, np.int64) or type(query_ts) not in (int, np.int64):
        raise ValueError('Integer snapshot and query timestamps required')
    if cutoff<=0 or query_ts<cutoff or not np.isfinite(prior_strength) or prior_strength<=0:
        raise ValueError('Invalid snapshot cutoff/query/prior')
    if (c<0).any() or (g<0).any() or (np.diff(c,axis=1)<0).any() or (np.diff(g,axis=0)<0).any():
        raise ValueError('Nested nonnegative count windows required')
    if (c.sum(axis=0)>g).any():raise ValueError('Full-catalogue totals cannot be candidate-only totals')
    if (l < -1).any() or (l[l>=0]>=cutoff).any():raise ValueError('Last occurrence reaches the snapshot cutoff')
    for j,h in enumerate(WINDOWS):
        present=c[:,j,:]>0
        if np.any(present & (l<cutoff-h*HOUR)):
            raise ValueError('Positive counts disagree with last occurrence')
        if np.any((~present)&(l>=cutoff-h*HOUR)):
            raise ValueError('Last occurrence disagrees with zero window counts')
    # The original transform uses float32 item counts, float64 catalogue totals.
    x=c.astype(np.float32);columns={};pr={}
    for ai,a in enumerate(ACTIONS):
        for wi,h in enumerate(WINDOWS):
            values=x[:,wi,ai];stem=f'asof_{a}_h{h}';total=float(values.sum())
            pr[(ai,h)]=percentile(values)
            columns[stem+'_log_count']=np.log1p(values).astype(np.float32)
            columns[stem+'_candidate_percentile']=pr[(ai,h)]
            columns[stem+'_candidate_mass_share']=(values/total if total else np.zeros_like(values))
        seen=l[:,ai]>=0;age=np.zeros(len(c),np.float32)
        age[seen]=((query_ts-l[seen,ai])/HOUR).astype(np.float32)
        columns[f'asof_{a}_last_seen']=seen.astype(np.float32)
        columns[f'asof_{a}_last_age_log_hours']=np.where(seen,np.log1p(age),0).astype(np.float32)
        for short,long in PAIRS:
            columns[f'asof_{a}_trend_h{short}_h{long}']=(np.log1p(x[:,WINDOWS.index(short),ai]/short)-np.log1p(x[:,WINDOWS.index(long),ai]/long))
            columns[f'asof_{a}_percentile_delta_h{short}_h{long}']=pr[(ai,short)]-pr[(ai,long)]
    for h in (24,168):
        wi=WINDOWS.index(h);s=g[wi].astype(np.float64);total=float(s.sum())
        prior=s/total if total else np.full(3,1/3)
        denominator=x[:,wi,:].sum(1)+prior_strength
        for ai,a in enumerate(ACTIONS):
            columns[f'asof_action_mix_{a}_h{h}']=((x[:,wi,ai]+prior_strength*prior[ai])/denominator).astype(np.float32)
    result=np.column_stack([columns[n] for n in NAMES]).astype(np.float32)
    if not np.isfinite(result).all() or result.shape!=(len(c),75):raise ValueError('Invalid demand feature matrix')
    return result
