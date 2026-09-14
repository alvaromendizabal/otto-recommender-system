"""Historical feature dispatch and shared neighbor selection; no targets, fits or I/O.

R08: reciprocal-square-root historical anchor-frequency weighting versus equal
rarity weights. R09: strictly forward original-event continuation versus the
same unsigned local window. R10: same-product ordered versus unordered action
progression in funnel_features.py. All supports count distinct historical sessions.
"""
from __future__ import annotations
from collections import defaultdict
import math
from typing import Mapping
import numpy as np
from neighbors import HistoricalSession

KINDS=('clicks','carts','orders')
R08_MEASURES=('log_session_support','weighted_vote_share','max_similarity',
 'reciprocal_neighbor_rank_share','multi_anchor_vote_share','after_anchor_vote_share',
 'log_effective_support','anchor_coverage_vote_share')
R09_MEASURES=('log_session_support','weighted_vote_share','max_similarity',
 'inverse_event_gap_vote_share','time_decay_vote_share','immediate_event_vote_share',
 'multi_anchor_vote_share','reciprocal_neighbor_rank_share')

def names(round_no:int, ablation:bool=False)->tuple[str,...]:
    if round_no==10:
        from funnel_features import feature_names
        return feature_names(ablation)
    if round_no not in (8,9):raise ValueError('Round must be 8, 9 or 10')
    prefix=({8:'rarity_neighbor',9:'forward_continuation'} if not ablation else
            {8:'equal_weight_neighbor',9:'unsigned_window'})[round_no]
    measures=R08_MEASURES if round_no==8 else R09_MEASURES
    return tuple(f'{prefix}_{k}_{m}' for k in KINDS for m in measures)

def inputs(anchors,candidates,source_ids,history,frequencies):
    aa=np.asarray(anchors);cc=np.asarray(candidates)
    for a in (aa,cc):
        if a.ndim!=1 or a.dtype.kind not in 'iu' or (a<0).any() or len(np.unique(a))!=len(a):
            raise ValueError('Nonnegative unique integer anchors/candidates required')
    if not 1<=len(aa)<=4 or not len(cc):raise ValueError('Empty/oversized query')
    if len(set(source_ids))!=len(source_ids):raise ValueError('Duplicate historical source IDs')
    for sid in source_ids:
        if sid not in history or history[sid].session!=sid:raise ValueError('Missing/misaligned historical source')
    for a in aa:
        df=frequencies.get(int(a),0)
        if type(df) not in (int,np.int64,np.int32) or df<0:raise ValueError('Invalid uncapped anchor frequency')
    return aa,cc

def select(anchors,source_ids,history,frequencies,rarity=False,require_last=False):
    """Weighted-query/binary-history cosine. Deterministic ties, cap exactly 64.

    Frequencies count eligible sessions BEFORE the per-anchor postings cap.
    This is inverse-square-root support, NOT TF-IDF or calibrated probability.
    """
    weights={int(a):1.0/(j+1)/(math.sqrt(max(1,int(frequencies.get(int(a),0)))) if rarity else 1.)
             for j,a in enumerate(anchors)}
    norm=math.sqrt(math.fsum(w*w for w in weights.values()));result=[]
    for sid in sorted(source_ids):
        h=history[sid];items=set(map(int,h.aids));common=sorted(items & weights.keys())
        if not common or (require_last and int(anchors[0]) not in items):continue
        if any(frequencies.get(a,0)<=0 for a in common):raise ValueError('History exists but anchor frequency is zero')
        sim=math.fsum(weights[a] for a in common)/(norm*math.sqrt(len(items)))
        if not 0<sim<=1+1e-12:raise ValueError('Cosine outside contract')
        result.append((h,min(sim,1.),len(common)))
    result.sort(key=lambda r:(-r[1],-int(r[0].timestamps[-1]),r[0].session))
    return result[:64]

def rarity_votes(anchors,nn):
    total=math.fsum(s for _,s,_ in nn);ranks=math.fsum(1/(i+1) for i in range(len(nn)))
    sums=defaultdict(lambda:np.zeros(9,dtype=np.float64))
    for rank,(h,sim,common) in enumerate(nn,1):
        matched=np.isin(h.aids,anchors);pivot=int(h.indices[matched].max())
        distinct=set(zip(map(int,h.aids),map(int,h.kinds)))
        after=set(zip(map(int,h.aids[h.indices>pivot]),map(int,h.kinds[h.indices>pivot])))
        for key in sorted(distinct):
            v=sums[key];v[0]+=1;v[1]+=sim;v[2]=max(v[2],sim);v[3]+=1/rank
            v[4]+=sim*(common>=2);v[5]+=sim*(key in after);v[6]+=sim*sim
            v[7]+=sim*common/len(anchors)
    out={}
    for key,v in sums.items():
        effective=v[1]*v[1]/v[6] if v[6] else 0.
        out[key]=np.array([math.log1p(v[0]),v[1]/total,v[2],v[3]/ranks,
                         v[4]/total,v[5]/total,math.log1p(effective),v[7]/total])
    return out

def continuation_votes(anchors,nn,forward=True):
    """Same selected neighbors/denominators in both arms; change direction only.

    Pivot = final retained occurrence of query's most recent distinct product.
    Gap uses ORIGINAL event_index, never compressed retained row positions.
    A product/action contributes at most once per neighbor. Self-anchor excluded
    symmetrically; old revisit features remain in the untouched control.
    """
    total=math.fsum(s for _,s,_ in nn);ranks=math.fsum(1/(i+1) for i in range(len(nn)))
    sums=defaultdict(lambda:np.zeros(8,dtype=np.float64))
    for rank,(h,sim,common) in enumerate(nn,1):
        matched=np.flatnonzero(h.aids==anchors[0])
        if not len(matched):raise ValueError('Selected continuation neighbor lacks last anchor')
        pivot=int(matched[-1]);gap=h.indices-int(h.indices[pivot]);dt=h.timestamps-int(h.timestamps[pivot])
        use=(gap>0) if forward else (gap!=0)
        use &= (np.abs(gap)<=5)&(np.abs(dt)<=1800_000)&(h.aids!=anchors[0])
        positions=defaultdict(list)
        for ix in np.flatnonzero(use):positions[(int(h.aids[ix]),int(h.kinds[ix]))].append(int(ix))
        for key,ii in sorted(positions.items()):
            # Per-measure maxima avoid inflation by repeated events of one item/action.
            g=int(np.min(np.abs(gap[ii])));t=int(np.min(np.abs(dt[ii])))
            v=sums[key];v[0]+=1;v[1]+=sim;v[2]=max(v[2],sim)
            v[3]+=sim/g;v[4]+=sim*math.exp(-t/300_000);v[5]+=sim*(g==1)
            v[6]+=sim*(common>=2);v[7]+=1/rank
    out={}
    for key,v in sums.items():
        out[key]=np.array([math.log1p(v[0]),v[1]/total,v[2],v[3]/total,
                         v[4]/total,v[5]/total,v[6]/total,v[7]/ranks])
    return out

def project(votes:Mapping,candidates:np.ndarray)->np.ndarray:
    out=np.zeros((len(candidates),24),np.float32)
    for i,a in enumerate(candidates):
        for k in range(3):
            value=votes.get((int(a),k))
            if value is not None:out[i,k*8:k*8+8]=value
    if not np.isfinite(out).all() or (out<0).any():raise ValueError('Invalid feature values')
    return out

def build(round_no,anchors,candidates,source_ids,history,frequencies,*,cache=None):
    if round_no==10:
        from funnel_features import build_funnel
        return build_funnel(anchors,candidates,source_ids,history,frequencies,cache=cache)
    anchors,candidates=inputs(anchors,candidates,source_ids,history,frequencies)
    if round_no==8:
        na=select(anchors,source_ids,history,frequencies,rarity=True)
        nb=select(anchors,source_ids,history,frequencies,rarity=False)
        a=rarity_votes(anchors,na);b=rarity_votes(anchors,nb)
    elif round_no==9:
        na=nb=select(anchors,source_ids,history,frequencies,rarity=False,require_last=True)
        a=continuation_votes(anchors,na,True);b=continuation_votes(anchors,nb,False)
    else:raise ValueError('Unknown feature round')
    return project(a,candidates),project(b,candidates),{
       'neighbor_counts':[len(na),len(nb)],'multi_anchor_counts':[sum(r[2]>=2 for r in n) for n in (na,nb)],
       'neighbor_identity_overlap':len({r[0].session for r in na}&{r[0].session for r in nb}),
       'primary_supported_pairs':len(a),'ablation_supported_pairs':len(b)}
