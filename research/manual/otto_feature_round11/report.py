"""Read-only metric audit and nine Plotly charts. Never fits or rebuilds features."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from core import checked, pooled, save_json, sha
from analysis_utils import compare

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
ARMS=('control_shared','idf_session_latent','unweighted_session_latent')
LABELS={'control_shared':'134-feature saved control',
        'idf_session_latent':'+ 24 primary latent-affinity features',
        'unweighted_session_latent':'+ 24 matched ablation features'}
PAIRS=(('idf_session_latent','control_shared'),('unweighted_session_latent','control_shared'),
       ('idf_session_latent','unweighted_session_latent'))

def validate_result(out, prior_stats=None, expected_sessions=2048):
    r=json.loads((out/'result.json').read_text())
    if r['status']!='ROUND11_SCREEN_COMPLETED' or set(r['arms'])!=set(ARMS):
        raise ValueError('Complete Round11 result required')
    checked(out/'evaluation_statistics.npz',r['statistics_sha256'])
    with np.load(out/'evaluation_statistics.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    if data['denominator'].shape!=(expected_sessions,3) or len(np.unique(data['ids']))!=expected_sessions:
        raise ValueError('Validation cohort shape/identity invalid')
    if set(data['folds'])!={0,1} or any((data['folds']==f).sum()!=expected_sessions//2 for f in (0,1)):
        raise ValueError('Expected two full chronological folds')
    prior_stats=prior_stats or ROOT/'evidence/ROUND06_STATISTICS.npz'
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

def create_report(out=OUT, prior_stats=None, expected_sessions=2048):
    import plotly.graph_objects as go
    import plotly.io as pio
    r=validate_result(out,prior_stats,expected_sessions)
    (out/'plots').mkdir(exist_ok=True)
    figures=[]
    f=go.Figure(go.Bar(x=[LABELS[a] for a in ARMS],y=[r['arms'][a]['weighted_recall_at_20'] for a in ARMS],
                     text=[f"{r['arms'][a]['weighted_recall_at_20']:.6f}" for a in ARMS],textposition='outside'))
    f.update_layout(title='Matched fitting-only Recall@20 — identical 400-candidate pools',yaxis_title='Weighted Recall@20');figures.append(f)
    cmp=r['comparisons'];keys=[l+'_minus_'+rr for l,rr in PAIRS]
    vals=[cmp[k]['gain'] for k in keys]
    ci=[cmp[k]['descriptive_95_interval'] for k in keys]
    f=go.Figure(go.Scatter(x=vals,y=['Primary − control','Ablation − control','Primary − ablation'],mode='markers',
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
    checked(out/'context_diagnostics.json',r['context_diagnostics_sha256'])
    contexts=json.loads((out/'context_diagnostics.json').read_text())
    f=go.Figure()
    for row in contexts:
        f.add_trace(go.Bar(name='Training fold '+str(row['fold']),x=[x['name'] for x in first],y=row['mean_abs_primary_minus_ablation']))
    f.update_layout(title='Do the primary and ablation representations differ?',barmode='group',yaxis_title='Mean absolute primary–ablation difference',height=690,xaxis_tickangle=-70);figures.append(f)
    f=go.Figure(go.Scatter(x=[x['support_fraction'] for x in first],y=[x['max_abs_corr_control'] for x in first],
               mode='markers',text=[x['name'] for x in first],hovertemplate='%{text}<br>Support=%{x:.3f}<br>Max correlation=%{y:.3f}<extra></extra>'))
    f.update_layout(title='Training-only redundancy diagnostic — no automatic feature filtering',xaxis_title='Support fraction',yaxis_title='Maximum |correlation| with 134 controls');figures.append(f)
    checked(out/'candidate_coverage.json',r['candidate_coverage_sha256'])
    checked(out/'positive_support.json',r['positive_support_sha256'])
    coverage=json.loads((out/'candidate_coverage.json').read_text())['pooled']
    f=go.Figure()
    f.add_trace(go.Bar(name='Saved control',x=['Clicks','Carts','Orders'],y=[coverage['achieved_control']['recall'][o] for o in ('clicks','carts','orders')]))
    f.add_trace(go.Bar(name='Candidate oracle at 20',x=['Clicks','Carts','Orders'],y=[coverage['candidate_oracle_at20']['recall'][o] for o in ('clicks','carts','orders')]))
    f.update_layout(title='Candidate ceiling versus achieved ranking — not a new score',barmode='group',yaxis_title='Recall under complete capped denominators');figures.append(f)
    support=json.loads((out/'positive_support.json').read_text());ss=[v for v in support if v['fold']==0]
    f=go.Figure()
    for o in ('clicks','carts','orders'):
        f.add_trace(go.Bar(name=o,x=[v['name'] for v in ss],y=[v['positive_training_candidate_support'][o] for v in ss]))
    f.update_layout(title='Training-positive exposure — censored before validation cutoff',barmode='group',yaxis_title='Nonzero support among candidate positives',xaxis_tickangle=-50,height=700);figures.append(f)
    sections=[];plot_hashes={}
    for i,fig in enumerate(figures,1):
        fig.update_layout(margin=dict(l=65,r=35,t=90,b=150 if i in (6,9) else 80),
                          xaxis=dict(automargin=True),yaxis=dict(automargin=True))
        path=out/'plots'/f'{i:02d}.json';path.write_text(pio.to_json(fig,pretty=False))
        plot_hashes[path.name]=sha(path)
        sections.append(pio.to_html(fig,full_html=False,include_plotlyjs=True if i==1 else False))
    title='OTTO Round 11 · Historical item-session latent affinities'
    primary=cmp['idf_session_latent_minus_control_shared']
    html='<html><head><meta charset="utf-8"><title>'+title+'</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.55}h1{letter-spacing:-.03em}p{max-width:900px}.plotly-graph-div{margin:28px 0}</style></head><body><h1>'+title+'</h1>'
    label = 'SYNTHETIC TEST FIXTURE — not an experiment result' if r.get('fixture_only') else 'Actual saved experiment; fitting-only exploratory comparison'
    html+='<p>'+label+'. Not a Kaggle score. Primary decision: '+primary['decision']+'</p>'
    html+=''.join(sections)+'</body></html>'
    (out/'round11_report.html').write_text(html)
    save_json(out/'report_receipt.json',{'status':'ROUND11_REPORT_READY','result_sha256':sha(out/'result.json'),
        'plot_hashes':plot_hashes,'charts':9,'new_model_fits':0,'metrics_replayed':True})
    (out/'MILESTONE_REVIEW.md').write_text('# OTTO Round 11\n\n'+
        f"Control {r['arms']['control_shared']['weighted_recall_at_20']:.6f}; primary gain {primary['gain']:+.6f}.\n\n"+
        'Decision: '+primary['decision']+'. 24 primary and 24 information-removal ablation features; independent fixed-134-control comparisons; no combined 48-feature arm '+
        'Twelve challenger models and six existing control replays; no raw JSON scan, old-graph rebuild, control refit, GitHub/cloud writes or submission. Existing certified historical data are reused read-only; no new historical extraction. '+
        'Feature engineering remains open. Full source and experiment receipts are in this package.\n')
    print('RESULT: ROUND11_REPORT_READY',flush=True)
    return r

if __name__=='__main__':create_report()
