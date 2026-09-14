"""Exact pooled metrics and descriptive paired uncertainty; no model execution."""
import numpy as np
from core import OBJECTIVES, WEIGHTS, pooled, paired_interval

def compare(left,right,den,folds):
    l,r=pooled(left,den),pooled(right,den)
    gain=l['weighted_recall_at_20']-r['weighted_recall_at_20']
    fg=[pooled(left[folds==f],den[folds==f])['weighted_recall_at_20']-
        pooled(right[folds==f],den[folds==f])['weighted_recall_at_20'] for f in np.unique(folds)]
    order=l['recall']['orders']-r['recall']['orders'];ci=paired_interval(left-right,den,folds)
    if gain<=0:decision='STOP_THIS_CONFIGURATION_NO_GAIN'
    elif min(fg)<0 or order<0:decision='UNSTABLE_DO_NOT_PROMOTE'
    elif gain<.003:decision='SMALL_GAIN_REVIEW_COMPUTE_VALUE'
    else:decision='NEW_COHORT_AND_TIME_WINDOW_CONFIRMATION_REQUIRED'
    delta=left.sum(0)-right.sum(0);total=den.sum(0)
    return {'gain':gain,'fold_gains':fg,'order_gain':order,**ci,'decision':decision,
            'net_extra_hits':dict(zip(OBJECTIVES,map(int,delta),strict=True)),
            'weighted_contributions':dict(zip(OBJECTIVES,map(float,delta/total*WEIGHTS),strict=True)),
            'one_order_hit_weight':float(.6/total[2]),'no_automatic_promotion':True}
