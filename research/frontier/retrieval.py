"""Label-blind two-hop retrieval, independently implemented on historical CSR graphs.

This is a bounded adaptation, not a reconstruction of the full winning pipeline.
The fixed paths are time->time, time->cart and cart->order. Each row is normalized
before propagation so a high-degree intermediate cannot win solely by raw mass.
"""
from __future__ import annotations
import numpy as np
from .contracts import require

PATHS=((0,0),(0,1),(1,2))

def weighted_step(graph, seeds, weights):
    ids=[]; mass=[]
    require(len(seeds)==len(weights),'SEED_WEIGHT_ALIGNMENT')
    for source,weight in zip(seeds,weights,strict=True):
        source=int(source)
        require(np.isfinite(weight) and weight>=0,'INVALID_SEED_WEIGHT')
        if not 0<=source<graph.shape[0] or weight==0:continue
        start,end=graph.indptr[source:source+2]
        dst=graph.indices[start:end];v=np.asarray(graph.data[start:end],np.float64)
        require(np.isfinite(v).all() and np.all(v>=0),'INVALID_GRAPH_VALUES')
        total=float(v.sum())
        if not total>0:continue
        positive=v>0
        ids.append(dst[positive]);mass.append(v[positive]*(float(weight)/total))
    if not ids:return np.empty(0,np.int64),np.empty(0,np.float64)
    target=np.concatenate(ids).astype(np.int64);score=np.concatenate(mass)
    unique,inverse=np.unique(target,return_inverse=True)
    summed=np.zeros(len(unique),np.float64);np.add.at(summed,inverse,score)
    return unique,summed

def best(ids, scores, k):
    require(ids.shape==scores.shape and np.isfinite(scores).all(),'BAD_RETRIEVAL_SCORES')
    take=np.lexsort((ids,-scores))[:k]
    return ids[take],scores[take]

def expand(graphs, prefix, base, *, seed_tail=20, intermediate=20, replacements=100):
    """Return exactly the old budget, retaining recent seen items and old head.

No labels, targets, scores from validation, or task outcomes enter this API.
"""
    base=np.asarray(base,dtype=np.int64)
    require(len(graphs)==3 and base.ndim==1 and len(np.unique(base))==len(base),'RETRIEVAL_INPUT')
    require(len(prefix.aid)>0 and len(prefix.kind)==len(prefix.aid),'PREFIX_INPUT')
    require(np.isin(prefix.kind,[0,1,2]).all(),'PREFIX_ACTION')
    seeds=np.asarray(prefix.aid[-seed_tail:][::-1],np.int64)
    kinds=np.asarray(prefix.kind[-seed_tail:][::-1],np.int64)
    weights=np.asarray([1.,3.,6.])[kinds]/np.sqrt(np.arange(1,len(seeds)+1))
    weights=weights/weights.sum()
    all_ids=[];all_scores=[];paths=[]
    for weight,(first,second) in zip((.1,.3,.6),PATHS,strict=True):
        ids,scores=weighted_step(graphs[first],seeds,weights)
        bridge,bridge_score=best(ids,scores,intermediate)
        dst,mass=weighted_step(graphs[second],bridge,bridge_score)
        all_ids.append(dst);all_scores.append(weight*mass)
        paths.append({'path':[first,second],'intermediates':len(bridge),'reachable':len(dst)})
    if all(len(a)==0 for a in all_ids):return base.copy(),{'inserted':0,'paths':paths}
    ids=np.concatenate(all_ids);scores=np.concatenate(all_scores)
    keys,inv=np.unique(ids,return_inverse=True);s=np.zeros(len(keys),np.float64);np.add.at(s,inv,scores)
    novel=(~np.isin(keys,base))&(s>0)
    novel_ids,novel_scores=best(keys[novel],s[novel],replacements)
    # Existing generator places most-recent unique revisits first. Preserve every
    # seen item that it admitted, even for unusually long sessions.
    seen=set(map(int,prefix.aid));seen_positions=np.flatnonzero(np.isin(base,list(seen)))
    keep_at_least=max(len(base)-replacements, int(seen_positions.max()+1) if len(seen_positions) else 0)
    n=min(len(novel_ids),len(base)-keep_at_least)
    out=np.r_[base[:len(base)-n],novel_ids[:n]] if n else base.copy()
    require(len(out)==len(base) and len(np.unique(out))==len(base),'CANDIDATE_BUDGET_OR_DUPLICATE')
    require(set(base[np.isin(base,list(seen))]).issubset(set(out)),'SEEN_CANDIDATE_LOST')
    return out,{'inserted':int(n),'paths':paths,'novel_score_max':float(novel_scores.max(initial=0))}

def candidates_for(engine,prefix,chosen,candidate_type,ranks):
    """Reconstruct original feature semantics for arbitrary selected item IDs.

Ranks are calculated over the original discovery universe plus the new IDs, not
just over a truncated list. Baseline exact replay is required before new scoring.
"""
    prefix.validate();chosen=np.asarray(chosen,np.int64)
    require(len(np.unique(chosen))==len(chosen) and np.all(chosen>=0),'CANDIDATE_IDENTITIES')
    require(prefix.ts[0]>=engine.cutoff,'HISTORICAL_CUTOFF')
    seeds=prefix.aid[-50:][::-1];valid=seeds<engine.known.size;safe=np.where(valid,seeds,0)
    rows=[g[safe] for g in engine.graphs]
    if not valid.all():
        rows=[r.multiply(valid[:,None]).tocsr() for r in rows]
        for row in rows:row.eliminate_zeros()
    discovery=np.unique(np.concatenate([prefix.aid,*engine.popular,*[r.indices for r in rows],chosen]))
    inside=discovery<engine.known.size
    values=np.zeros((len(seeds),len(discovery),3),np.float32)
    for j,row in enumerate(rows):values[:,inside,j]=row[:,discovery[inside]].toarray()
    values[~valid]=0
    recency=(1/np.sqrt(np.arange(1,len(seeds)+1))).astype(np.float32)
    scores=np.zeros((len(discovery),5),np.float32)
    scores[:,:3]=np.einsum('scg,s->cg',values,recency,optimize=False)
    np.add.at(scores[:,3],np.searchsorted(discovery,prefix.aid[::-1]),1/np.sqrt(np.arange(1,len(prefix.aid)+1)))
    for weight,popular in zip((.1,.3,.6),engine.popular,strict=True):
        scores[np.searchsorted(discovery,popular),4]+=(weight/np.arange(1,len(popular)+1)).astype(np.float32)
    rr=np.column_stack([ranks(scores[:,j],discovery) for j in range(5)])
    select=np.searchsorted(discovery,chosen)
    require(np.array_equal(discovery[select],chosen),'CHOSEN_LOOKUP')
    return candidate_type(chosen,scores[select],rr[select],values[:,select])
