"""Same-product action progression in a fixed historical-session neighborhood.

No targets, prediction cutoffs inferred from labels, I/O, model fitting or
candidate changes occur here. The caller verifies historical availability and
study-session exclusions. Unobserved behaviors are NOT negatives; local ratios
are selected-history evidence, not calibrated conversion probabilities.
"""
from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np
from neighbors import HistoricalSession

PAIRS=((0,1),(1,2),(0,2))
PAIR_NAMES=('click_to_cart','cart_to_order','click_to_order')
MEASURES=('log_pair_session_support','weighted_pair_share',
          'shrunk_pair_given_source','weighted_pair_given_source',
          'max_supporting_similarity','reciprocal_neighbor_rank_share',
          'lag_decayed_pair_share','log_effective_pair_support')
MAX_ELAPSED_MS=1800000
DECAY_MS=300000
SHRINKAGE=20


def feature_names(ablation:bool=False)->tuple[str,...]:
    prefix='unordered_funnel' if ablation else 'ordered_funnel'
    return tuple(f'{prefix}_{pair}_{measure}' for pair in PAIR_NAMES for measure in MEASURES)


def nearest_lag(source:list[tuple[int,int]], destination:list[tuple[int,int]], ordered:bool)->int|None:
    """Minimum eligible gap without a quadratic all-event cross join.

    Positions are ORIGINAL event indices, timestamps are nondecreasing. Strict
    index order resolves tied timestamps. No position-adjacency requirement:
    intermediate shopping events may separate two actions on the same product.
    """
    if not source or not destination:return None
    best=None
    if ordered:
        j=0;latest=None
        for di,dt in destination:
            while j<len(source) and source[j][0]<di:
                latest=source[j][1];j+=1
            if latest is not None:
                lag=dt-latest
                if lag<0:raise ValueError('Historical timestamps violate original event order')
                if lag<=MAX_ELAPSED_MS:best=lag if best is None else min(best,lag)
    else:
        i=j=0
        while i<len(source) and j<len(destination):
            lag=abs(source[i][1]-destination[j][1])
            if lag<=MAX_ELAPSED_MS:best=lag if best is None else min(best,lag)
            if source[i][1]<=destination[j][1]:i+=1
            else:j+=1
    return best


@dataclass(frozen=True)
class SessionSummary:
    click_items:frozenset[int]
    cart_items:frozenset[int]
    # (item, pair-index) -> (minimum ordered lag, minimum unsigned lag)
    pairs:Mapping[tuple[int,int],tuple[int|None,int|None]]


def summarize_session(h:HistoricalSession)->SessionSummary:
    grouped={}
    for aid,kind,ix,ts in zip(h.aids,h.kinds,h.indices,h.timestamps,strict=True):
        rows=grouped.setdefault(int(aid),[[],[],[]])
        rows[int(kind)].append((int(ix),int(ts)))
    pairs={};clicks=set();carts=set()
    for aid,actions in grouped.items():
        if actions[0]:clicks.add(aid)
        if actions[1]:carts.add(aid)
        for k,(src,dst) in enumerate(PAIRS):
            if actions[src] and actions[dst]:
                ordered=nearest_lag(actions[src],actions[dst],True)
                unsigned=nearest_lag(actions[src],actions[dst],False)
                if unsigned is not None:pairs[(aid,k)]=(ordered,unsigned)
    return SessionSummary(frozenset(clicks),frozenset(carts),pairs)


class SummaryCache:
    """Bounded in-process memoization. It is discarded between feature stages.

    The history mapping is bound by identity to prevent summaries leaking across
    different sources that happen to reuse a session ID. No files are created.
    """
    def __init__(self,history:Mapping[int,HistoricalSession],max_entries:int=32768):
        if type(max_entries) is not int or max_entries<1:raise ValueError('Positive cache bound required')
        self.history=history;self.max_entries=max_entries;self._cache=OrderedDict()
    def get(self,sid:int)->SessionSummary:
        if sid in self._cache:
            value=self._cache.pop(sid);self._cache[sid]=value;return value
        value=summarize_session(self.history[sid]);self._cache[sid]=value
        if len(self._cache)>self.max_entries:self._cache.popitem(last=False)
        return value


def aggregate(neighbors,history,cache=None):
    """Two matched 24-feature arms. Every pair votes at most once per session."""
    if cache is not None and cache.history is not history:raise ValueError('Cache/history identity mismatch')
    mass=math.fsum(sim for _,sim,_ in neighbors)
    rank_mass=math.fsum(1/r for r in range(1,len(neighbors)+1))
    exposure=defaultdict(lambda:[0,0.])
    arms=[defaultdict(lambda:np.zeros(7,dtype=np.float64)),defaultdict(lambda:np.zeros(7,dtype=np.float64))]
    for rank,(h,sim,_) in enumerate(neighbors,1):
        summary=cache.get(h.session) if cache is not None else summarize_session(h)
        for kind,items in ((0,summary.click_items),(1,summary.cart_items)):
            for aid in sorted(items):
                x=exposure[(aid,kind)];x[0]+=1;x[1]+=sim
        for key,lags in sorted(summary.pairs.items()):
            for acc,lag in zip(arms,lags,strict=True):
                if lag is None:continue
                v=acc[key];v[0]+=1;v[1]+=sim;v[2]=max(v[2],sim);v[3]+=1/rank
                v[4]+=sim*math.exp(-lag/DECAY_MS);v[5]+=sim*sim
    output=[]
    for acc in arms:
        votes={}
        for (aid,k),v in acc.items():
            n,source_mass=exposure[(aid,PAIRS[k][0])]
            if n<v[0] or source_mass+1e-12<v[1] or mass<=0 or source_mass<=0:
                raise ValueError('Invalid same-product source denominator')
            eff=v[1]**2/v[5]
            votes[(aid,k)]=np.array([math.log1p(v[0]),v[1]/mass,v[0]/(n+SHRINKAGE),
                                    min(1.,v[1]/source_mass),v[2],v[3]/rank_mass,
                                    v[4]/mass,math.log1p(eff)],dtype=np.float64)
        output.append(votes)
    return output


def build_funnel(anchors,candidates,source_ids,history,frequencies,*,cache=None):
    # Local import avoids a circular module import and preserves legacy R09 code.
    from feature_logic import inputs,select,project
    anchors,candidates=inputs(anchors,candidates,source_ids,history,frequencies)
    nn=select(anchors,source_ids,history,frequencies,rarity=False,require_last=False)
    ordered,unsigned=aggregate(nn,history,cache)
    return project(ordered,candidates),project(unsigned,candidates),{
        'neighbor_counts':[len(nn),len(nn)],
        'multi_anchor_counts':[sum(v[2]>=2 for v in nn)]*2,
        'neighbor_identity_overlap':len(nn),
        'primary_supported_pairs':len(ordered),'ablation_supported_pairs':len(unsigned)}
