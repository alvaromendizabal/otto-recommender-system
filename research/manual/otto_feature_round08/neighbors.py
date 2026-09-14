"""Historical session-neighbor evidence. Pure NumPy; no targets or training calls.

This is a bounded, query-dependent representation of whole retained historical
sessions, not an adjacency counter or a repetition-score normalization.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np

KINDS = ('clicks', 'carts', 'orders')
MEASURES = ('log_session_support', 'weighted_vote_share', 'max_similarity', 'after_anchor_vote_share')
NAMES = tuple(f'session_neighbor_{kind}_{measure}' for kind in KINDS for measure in MEASURES)
ABLATION_NAMES = tuple(n.replace('session_neighbor_', 'last_anchor_neighbor_') for n in NAMES)
MAX_ANCHORS=4
POSTINGS_PER_ANCHOR=64
NEIGHBORS=64
PROPOSALS=100


def query_anchors(aids: np.ndarray, max_anchors: int=MAX_ANCHORS) -> np.ndarray:
    """Last distinct observed items, most recent first. Never uses labels."""
    a=np.asarray(aids)
    if a.ndim!=1 or not a.size or a.dtype.kind not in 'iu' or (a<0).any() or not 1<=max_anchors<=4:
        raise ValueError('Invalid observed items/anchor bound')
    result=[];seen=set()
    for item in a[::-1]:
        item=int(item)
        if item not in seen: result.append(item);seen.add(item)
        if len(result)==max_anchors:break
    return np.array(result,dtype=np.int64)


@dataclass(frozen=True)
class HistoricalSession:
    session: int
    aids: np.ndarray
    kinds: np.ndarray
    indices: np.ndarray
    timestamps: np.ndarray

    def validate(self, cutoff: int) -> None:
        a,k,ix,t=map(np.asarray,(self.aids,self.kinds,self.indices,self.timestamps))
        if self.session<0 or a.ndim!=1 or not a.size or any(x.shape!=a.shape or x.dtype.kind not in 'iu' for x in (a,k,ix,t)):
            raise ValueError('Malformed historical session')
        if (a<0).any() or (ix<0).any() or not np.isin(k,(0,1,2)).all() or (np.diff(ix)<=0).any():
            raise ValueError('Invalid product/action or duplicate/nonmonotone original index')
        if (t<0).any() or (t>=cutoff).any() or (np.diff(t)<0).any():
            raise ValueError('Historical source reaches cutoff or has invalid chronology')


def group_history(rows: np.ndarray, cutoff: int, excluded: set[int]) -> dict[int,HistoricalSession]:
    """Rows are (session, aid, ts, kind, original event_index)."""
    x=np.asarray(rows)
    if x.ndim!=2 or x.shape[1]!=5 or x.dtype.kind not in 'iu':raise ValueError('History schema')
    if not len(x):return {}
    x=x[np.lexsort((x[:,4],x[:,0]))];ids,starts,counts=np.unique(x[:,0],return_index=True,return_counts=True)
    result={}
    for sid,s,n in zip(ids,starts,counts,strict=True):
        if int(sid) in excluded:raise ValueError('Study session leaked into historical source')
        a=x[s:s+n];obj=HistoricalSession(int(sid),a[:,1],a[:,3],a[:,4],a[:,2]);obj.validate(cutoff)
        result[int(sid)]=obj
    return result


def select_neighbors(anchors: np.ndarray, source_ids: list[int], history: dict[int,HistoricalSession], last_only: bool=False) -> list[tuple[HistoricalSession,float,int,int]]:
    """Cosine-like item overlap, with reciprocal query recency weights.

    Historical vectors are binary, the query is weighted. The last-anchor arm
    removes the older query anchors but sees the identical frozen source pool.
    Ties use most recent retained timestamp then source session ID.
    """
    anchors=np.asarray(anchors)
    if anchors.ndim!=1 or not 1<=len(anchors)<=4 or len(np.unique(anchors))!=len(anchors) or (anchors<0).any():
        raise ValueError('Distinct nonnegative query anchors required')
    if len(set(source_ids))!=len(source_ids):raise ValueError('Duplicate requested historical session')
    selected=anchors[:1] if last_only else anchors
    weights={int(a):1/(i+1) for i,a in enumerate(selected)}
    qnorm=math.sqrt(sum(w*w for w in weights.values()));neighbors=[]
    for sid in source_ids:
        if sid not in history:raise ValueError('Requested historical session absent')
        h=history[sid];items=set(map(int,h.aids));common=items & weights.keys()
        if not common:continue
        sim=math.fsum(weights[a] for a in sorted(common))/(qnorm*math.sqrt(len(items)))
        if not 0<sim<=1+1e-12:raise ValueError('Invalid session similarity')
        matched=np.isin(h.aids,list(common));pivot=int(h.indices[matched].max())
        neighbors.append((h,min(sim,1.),pivot,len(common)))
    neighbors.sort(key=lambda row:(-row[1],-int(row[0].timestamps[-1]),row[0].session))
    return neighbors[:NEIGHBORS]


def votes(neighbors: list[tuple[HistoricalSession,float,int,int]]) -> dict[tuple[int,int],np.ndarray]:
    """Once-per-session item/action votes, plus original-index follow-up evidence."""
    denom=math.fsum(row[1] for row in neighbors)
    totals=defaultdict(lambda:np.zeros(4,dtype=np.float64))
    for h,sim,pivot,_ in neighbors:
        distinct=set(zip(map(int,h.aids),map(int,h.kinds)))
        after=set(zip(map(int,h.aids[h.indices>pivot]),map(int,h.kinds[h.indices>pivot])))
        for key in sorted(distinct):
            a=totals[key];a[0]+=1;a[1]+=sim;a[2]=max(a[2],sim)
            if key in after:a[3]+=sim
    for a in totals.values():
        a[0]=math.log1p(a[0]);a[1]/=denom;a[3]/=denom
    return dict(totals)


def project(vote_map: dict[tuple[int,int],np.ndarray], candidates: np.ndarray) -> np.ndarray:
    aids=np.asarray(candidates)
    if aids.ndim!=1 or aids.dtype.kind not in 'iu' or (aids<0).any() or len(np.unique(aids))!=len(aids):
        raise ValueError('Distinct candidate identities required')
    out=np.zeros((len(aids),12),np.float32)
    for i,a in enumerate(aids):
        for kind in range(3):
            if (int(a),kind) in vote_map:out[i,4*kind:4*kind+4]=vote_map[(int(a),kind)]
    if not np.isfinite(out).all():raise ValueError('Nonfinite neighbor feature')
    return out


def propose(vote_map: dict[tuple[int,int],np.ndarray], count: int=PROPOSALS) -> tuple[np.ndarray,np.ndarray]:
    if type(count) is not int or not 1<=count<=100:raise ValueError('Invalid proposal cap')
    ids=np.full((3,count),-1,np.int64);scores=np.zeros((3,count),np.float32)
    for k in range(3):
        rows=sorted([(a,float(v[1])) for (a,t),v in vote_map.items() if t==k and v[1]>0],key=lambda x:(-x[1],x[0]))[:count]
        for j,(a,v) in enumerate(rows):ids[k,j]=a;scores[k,j]=v
    return ids,scores


def build_query(anchors: np.ndarray, candidates: np.ndarray, source_ids: list[int], history: dict[int,HistoricalSession]) -> dict:
    result={}
    for arm,last_only in [('session_context',False),('last_anchor',True)]:
        nn=select_neighbors(anchors,source_ids,history,last_only)
        vv=votes(nn);pi,ps=propose(vv)
        result[arm]={'features':project(vv,candidates),'proposal_ids':pi,'proposal_scores':ps,
                     'neighbor_count':len(nn),'multi_overlap_neighbors':sum(r[3]>=2 for r in nn),
                     'neighbor_ids':[r[0].session for r in nn]}
    return result
