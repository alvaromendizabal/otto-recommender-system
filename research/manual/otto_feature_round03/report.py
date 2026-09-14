"""Read-only metric audit and seven Plotly charts. Never fits or rebuilds features."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from core import checked, pooled, save_json, sha
from analysis_utils import compare

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
ARMS=('control_shared','mass_missing_aware','mass_zero_ablation')
LABELS={'control_shared':'134-feature saved control',
        'mass_missing_aware':'+ 36 recent-mass features (missing = -1)',
        'mass_zero_ablation':'+ same 36 features (missing = 0 ablation)'}
PAIRS=(('mass_missing_aware','control_shared'),('mass_zero_ablation','control_shared'),
       ('mass_missing_aware','mass_zero_ablation'))

def validate_result(out, prior_stats=None):
    r=json.loads((out/'result.json').read_text())
    if r['status']!='ROUND03_SCREEN_COMPLETED' or set(r['arms'])!=set(ARMS):
        raise ValueError('Complete Round03 result required')
    checked(out/'evaluation_statistics.npz',r['statistics_sha256'])
    with np.load(out/'evaluation_statistics.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    if data['denominator'].shape!=(2048,3) or len(np.unique(data['ids']))!=2048:
        raise ValueError('Validation cohort shape/identity invalid')
    if set(data['folds'])!={0,1} or any((data['folds']==f).sum()!=1024 for f in (0,1)):
        raise ValueError('Expected two full chronological folds')
    prior_stats=prior_stats or ROOT/'evidence/ROUND02_CONTROL_STATISTICS.npz'
    with np.load(prior_stats,allow_pickle=False) as old:
        for key in ['ids','folds','denominator','control_shared']:
            if not np.array_equal(data[key],old[key]):raise ValueError('Saved control/validation identity changed')
    for arm in ARMS:
        actual=pooled(data[arm],data['denominator'])
        if actual!=r['arms'][arm]:raise ValueError('Reported pooled metric differs from saved integers')
        for f in (0,1):
            ix=data['folds']==f;actual=pooled(data[arm][ix],data['denominator'][ix])
            rows=[x for x in r['fold_results'] if x['arm']==arm and x['fold']==f]
            if len(rows)!=1 or any(rows[0][k]!=v for k,v in actual.items()):
                raise ValueError('Reported fold metric differs from saved integers')
    for left,right in PAIRS:
        actual=compare(data[left],data[right],data['denominator'],data['folds'])
        if actual!=r['comparisons'][left+'_minus_'+right]:
            raise ValueError('Reported gain, interval or decision differs from replay')
    return r

def create_report(out=OUT, prior_stats=None):
    import plotly.graph_objects as go
    import plotly.io as pio
    r=validate_result(out,prior_stats)
    (out/'plots').mkdir(exist_ok=True)
    figures=[]
    f=go.Figure(go.Bar(x=[LABELS[a] for a in ARMS],y=[r['arms'][a]['weighted_recall_at_20'] for a in ARMS],
                     text=[f"{r['arms'][a]['weighted_recall_at_20']:.6f}" for a in ARMS],textposition='outside'))
    f.update_layout(title='Matched fitting-only Recall@20 — identical 400-candidate pools',yaxis_title='Weighted Recall@20');figures.append(f)
    cmp=r['comparisons'];keys=[l+'_minus_'+rr for l,rr in PAIRS]
    vals=[cmp[k]['gain'] for k in keys]
    ci=[cmp[k]['descriptive_95_interval'] for k in keys]
    f=go.Figure(go.Scatter(x=vals,y=['Missing-aware − control','Zero-ablation − control','Missing-aware − zero-ablation'],mode='markers',
                     error_x=dict(type='data',symmetric=False,array=[max(0,b[1]-v) for v,b in zip(vals,ci)],
                                  arrayminus=[max(0,v-b[0]) for v,b in zip(vals,ci)])))
    f.add_vline(x=0,line_dash='dash');f.update_layout(title='Paired descriptive 95% intervals — not promotion evidence',xaxis_title='Absolute gain');figures.append(f)
    f=go.Figure()
    for arm in ARMS[1:]:f.add_trace(go.Bar(name=LABELS[arm],x=['Earlier fold','Later fold'],y=cmp[arm+'_minus_control_shared']['fold_gains']))
    f.add_hline(y=0,line_dash='dash');f.update_layout(title='Does the gain survive both time folds?',yaxis_title='Gain versus matched control',barmode='group');figures.append(f)
    f=go.Figure()
    for arm in ARMS:f.add_trace(go.Bar(name=LABELS[arm],x=['Clicks','Carts','Orders'],y=[r['arms'][arm]['recall'][o] for o in ('clicks','carts','orders')]))
    f.update_layout(title='Action-specific recall — complete target denominators retained',barmode='group',yaxis_title='Pooled Recall@20');figures.append(f)
    f=go.Figure()
    for arm in ARMS[1:]:
        c=cmp[arm+'_minus_control_shared'];f.add_trace(go.Bar(name=LABELS[arm],x=['Clicks','Carts','Orders'],
            y=list(c['weighted_contributions'].values()),text=[str(c['net_extra_hits'][o])+' net hits' for o in ('clicks','carts','orders')]))
    # Values are selected by explicit action key, never dict insertion order from sorted JSON.
    for tr,arm in zip(f.data,ARMS[1:],strict=True):tr.y=tuple(cmp[arm+'_minus_control_shared']['weighted_contributions'][o] for o in ('clicks','carts','orders'))
    f.update_layout(title='Which extra hits actually change the weighted score?',barmode='group',yaxis_title='Contribution to weighted gain');figures.append(f)
    diag=json.loads((out/'diagnostics.json').read_text());first=[x for x in diag if x['fold']==0]
    f=go.Figure()
    f.add_trace(go.Bar(name='Other-item graph evidence available',x=[x['name'] for x in first],y=[x['support_fraction'] for x in first]))
    f.add_trace(go.Bar(name='Positive recent affinity fraction',x=[x['name'] for x in first],y=[x['recent_positive_fraction'] for x in first]))
    f.update_layout(title='Training-only availability: missing is not the same as zero',barmode='group',yaxis_title='Fraction of candidate rows',height=690,xaxis_tickangle=-70);figures.append(f)
    f=go.Figure(go.Scatter(x=[x['support_fraction'] for x in first],y=[x['max_abs_corr_control'] for x in first],
               mode='markers',text=[x['name'] for x in first],hovertemplate='%{text}<br>Support=%{x:.3f}<br>Max correlation=%{y:.3f}<extra></extra>'))
    f.update_layout(title='Training-only redundancy diagnostic — no automatic feature filtering',xaxis_title='Support fraction',yaxis_title='Maximum |correlation| with 134 controls');figures.append(f)
    sections=[];plot_hashes={}
    for i,fig in enumerate(figures,1):
        fig.update_layout(margin=dict(l=65,r=35,t=80,b=80))
        path=out/'plots'/f'{i:02d}.json';path.write_text(pio.to_json(fig,pretty=False))
        plot_hashes[path.name]=sha(path)
        sections.append(pio.to_html(fig,full_html=False,include_plotlyjs=True if i==1 else False))
    title='OTTO Round 03 · Recent affinity mass'
    primary=cmp['mass_missing_aware_minus_control_shared']
    html='<html><head><meta charset="utf-8"><title>'+title+'</title></head><body><h1>'+title+'</h1>'
    html+='<p>Actual saved experiment; fitting-only exploratory comparison. Not a Kaggle score. Primary decision: '+primary['decision']+'</p>'
    html+=''.join(sections)+'</body></html>'
    (out/'round03_report.html').write_text(html)
    save_json(out/'report_receipt.json',{'status':'ROUND03_REPORT_READY','result_sha256':sha(out/'result.json'),
        'plot_hashes':plot_hashes,'charts':7,'new_model_fits':0,'metrics_replayed':True})
    (out/'MILESTONE_REVIEW.md').write_text('# OTTO Round 03\n\n'+
        f"Control {r['arms']['control_shared']['weighted_recall_at_20']:.6f}; primary gain {primary['gain']:+.6f}.\n\n"+
        'Decision: '+primary['decision']+'. Missing encoding is an ablation, not an established explanation of prior failures. '+
        'Twelve challenger models and six existing control replays; no graph/control rebuild, GitHub/cloud writes or submission. '+
        'Feature engineering remains open. Full source and experiment receipts are in this package.\n')
    print('RESULT: ROUND03_REPORT_READY',flush=True)
    return r

if __name__=='__main__':create_report()
