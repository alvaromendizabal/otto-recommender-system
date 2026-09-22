"""Prediction-time two-hop affinity features on the already frozen retrieval policy.

Ranks and maxima use the complete path-reachable universe, not validation targets
or a truncated 400-item pool. No trained transform, target, or outcome is accepted.
"""
from __future__ import annotations
import numpy as np
from .contracts import require
from .retrieval import PATHS, weighted_step, best
TAGS=('time_time','time_cart','cart_order')
NAMES=tuple(f'hop_{tag}_{kind}' for tag in TAGS for kind in ('mass','max_share','reciprocal_rank','present'))+(
    'hop_combined_mass','hop_combined_logmass','hop_combined_reciprocal_rank',
    'hop_path_support','hop_in_original_pool','hop_original_reciprocal_rank')

def aligned(ids,values,chosen):
    out=np.zeros(len(chosen),np.float64)
    if len(ids):
        p=np.searchsorted(ids,chosen);good=p<len(ids)
        good[good]=ids[p[good]]==chosen[good]
        out[good]=values[p[good]]
    return out

def reciprocal(ids,score):
    rr=np.zeros(len(ids),np.float64)
    pos=np.flatnonzero(score>0)
    order=pos[np.lexsort((ids[pos],-score[pos]))]
    rr[order]=1/np.arange(1,len(order)+1)
    return rr

def transform(graphs,prefix,chosen,original):
    chosen=np.asarray(chosen,np.int64);original=np.asarray(original,np.int64)
    require(len(graphs)==3 and chosen.ndim==1 and len(np.unique(chosen))==len(chosen),'PATH_FEATURE_INPUT')
    require(len(prefix.aid)>0 and np.isin(prefix.kind,[0,1,2]).all(),'PATH_PREFIX')
    seeds=np.asarray(prefix.aid[-20:][::-1],np.int64);k=np.asarray(prefix.kind[-20:][::-1],np.int64)
    w=np.array([1.,3.,6.])[k]/np.sqrt(np.arange(1,len(seeds)+1));w=w/w.sum()
    blocks=[];path_ids=[];path_mass=[]
    for left,right in PATHS:
        ids,mass=weighted_step(graphs[left],seeds,w);bridge,bmass=best(ids,mass,20)
        ids,mass=weighted_step(graphs[right],bridge,bmass)
        path_ids.append(ids);path_mass.append(mass)
        a=aligned(ids,mass,chosen);rr=aligned(ids,reciprocal(ids,mass),chosen)
        blocks.extend([a,a/max(float(mass.max(initial=0)),1e-30),rr,(a>0).astype(float)])
    if any(len(x) for x in path_ids):
        ids=np.concatenate(path_ids);mass=np.concatenate([v*w for v,w in zip(path_mass,(.1,.3,.6),strict=True)])
        unique,ix=np.unique(ids,return_inverse=True);total=np.zeros(len(unique));np.add.at(total,ix,mass)
    else:unique=np.empty(0,np.int64);total=np.empty(0,np.float64)
    a=aligned(unique,total,chosen);rr=aligned(unique,reciprocal(unique,total),chosen)
    rank={int(s):1/(i+1) for i,s in enumerate(original)}
    blocks.extend([a,np.log1p(a*1e6),rr,sum(blocks[j] for j in (3,7,11)),
                   np.isin(chosen,original).astype(float),np.asarray([rank.get(int(s),0.) for s in chosen])])
    x=np.column_stack(blocks).astype(np.float32)
    require(x.shape==(len(chosen),18) and np.isfinite(x).all(),'PATH_FEATURE_SHAPE')
    return x
