"""Training-only, censored candidate audits; never presented as ranker performance."""
from __future__ import annotations
import numpy as np
from neighbors import KINDS


def censored_truth(ids, query_ts, last_index, true_counts, labels, fit_end, cutoff):
    result=[]
    for i,sid in enumerate(ids):
        if int(query_ts[i])>=cutoff:raise ValueError('Pilot query reaches validation time')
        all_sets=[set() for _ in range(3)];keep=[set() for _ in range(3)]
        for aid,kind,ts,index in labels.get(int(sid),[]):
            if kind not in (0,1,2) or aid<0 or aid in all_sets[kind]:raise ValueError('Invalid/distinct label contract')
            if not int(query_ts[i])<=ts<fit_end or index<=last_index[i]:raise ValueError('Label chronology differs')
            all_sets[kind].add(int(aid))
            if ts<cutoff:keep[kind].add(int(aid))
        if [len(s) for s in all_sets]!=list(true_counts[i]) or len(all_sets[0])>1:raise ValueError('Complete label ledger differs')
        result.append(keep)
    return result


def audit_arrays(candidates, primary, ablation, primary_proposals, ablation_proposals, truths):
    n=len(candidates)
    if candidates.shape!=(n,400) or primary.shape!=(n,400,12) or ablation.shape!=primary.shape:
        raise ValueError('Pilot feature shape mismatch')
    if primary_proposals.shape!=(n,3,100) or ablation_proposals.shape!=primary_proposals.shape or len(truths)!=n:
        raise ValueError('Pilot proposal shape mismatch')
    den=np.zeros((n,3),np.int64);baseline=np.zeros_like(den);expanded={a:np.zeros_like(den) for a in ('session_context','last_anchor')}
    supported={a:np.zeros_like(den) for a in expanded};hits_new={a:np.zeros_like(den) for a in expanded}
    proposal_counts={a:np.zeros_like(den) for a in expanded}
    for i in range(n):
        base=set(map(int,candidates[i]))
        if len(base)!=400:raise ValueError('Repeated baseline candidate')
        for k in range(3):
            truth=truths[i][k];den[i,k]=min(len(truth),20);baseline[i,k]=min(len(base&truth),20)
            for arm,xx,pp in [('session_context',primary,primary_proposals),('last_anchor',ablation,ablation_proposals)]:
                selected=pp[i,k][pp[i,k]>=0]
                if len(np.unique(selected))!=len(selected):raise ValueError('Repeated proposal')
                proposed=set(map(int,selected));union=base|proposed
                if len(union)>500:raise ValueError('Expansion budget exceeded')
                expanded[arm][i,k]=min(len(union&truth),20)
                # Raw distinct source hits and metric-capped net gains are separate.
                hits_new[arm][i,k]=len((proposed-base)&truth)
                proposal_counts[arm][i,k]=len(proposed-base)
                positive={int(a) for a in candidates[i,xx[i,:,4*k]>0]}
                supported[arm][i,k]=len(positive&truth)
    totals=den.sum(0)
    def summary(h):
        recall=[float(x/d) if d else None for x,d in zip(h.sum(0),totals,strict=True)]
        return {'hits':h.sum(0).tolist(),'denominators':totals.tolist(),'recall':dict(zip(KINDS,recall)),
                'weighted_oracle_at20':float(np.array(recall)@np.array([.1,.3,.6])) if all(x is not None for x in recall) else None}
    res={'training_only':True,'targets_censored_before_validation':True,'new_model_fits':0,'achieved_ranker_score':None,
         'baseline_candidate_oracle':summary(baseline),'arms':{},'metric_note':'Perfect-ranking candidate oracle at20 on a 256-query training-only pilot; NOT measured ranking accuracy or Kaggle score.'}
    for a in expanded:
        if (expanded[a]<baseline).any():raise ValueError('Preserving base pool cannot reduce oracle coverage')
        res['arms'][a]={'expanded_candidate_oracle':summary(expanded[a]),'net_capped_hits':(expanded[a]-baseline).sum(0).tolist(),
              'new_distinct_target_hits':hits_new[a].sum(0).tolist(),'supported_baseline_positive_items':supported[a].sum(0).tolist(),
              'mean_new_candidates_by_objective':proposal_counts[a].mean(0).tolist()}
    arrays={'denominator':den,'baseline_oracle_hits':baseline,
            **{a+'_expanded_oracle_hits':h for a,h in expanded.items()},
            **{a+'_supported_positive_counts':h for a,h in supported.items()},
            **{a+'_new_distinct_hits':h for a,h in hits_new.items()}}
    return res,arrays
