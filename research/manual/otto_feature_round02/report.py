"""Read and independently recompute saved timing-replication results. Never trains."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from core import OBJECTIVES, WEIGHTS, checked, decision, paired_interval, pooled, save_json, sha, write_once

LABELS={'control_shared':'Matched control · 134 features','plus_timing':'+ frozen timing · 158 features'}


def validate_saved_result(out):
    out=Path(out)
    r=json.loads((out/'result.json').read_text())
    if (r.get('status')!='ROUND02_SCREEN_COMPLETED' or r.get('round01_session_overlap')!=0
            or r.get('sessions')!=4096 or r.get('validation_sessions')!=2048
            or r.get('model_count')!=12 or r.get('selection_access') is not False
            or r.get('evaluation_access') is not False or r.get('competition_test_access') is not False):
        raise ValueError('Report is not the complete fitting-only replication')
    checked(out/'evaluation_statistics.npz',r['statistics_sha256'])
    with np.load(out/'evaluation_statistics.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    d=data['denominator'];f=data['folds'];ids=data['ids']
    if len(ids)!=r['validation_sessions'] or np.unique(ids).size!=len(ids) or set(f.tolist())!={0,1}:
        raise ValueError('Report session/fold identity invalid')
    for name in LABELS:
        value=pooled(data[name],d)
        if value!=r['arms'][name]:raise ValueError('Saved pooled metric differs from session statistics')
        for fold in (0,1):
            row=[x for x in r['fold_results'] if x['fold']==fold and x['arm']==name]
            exact=pooled(data[name][f==fold],d[f==fold])
            if len(row)!=1 or any(row[0][k]!=v for k,v in exact.items()):
                raise ValueError('Saved fold metric differs from statistics')
    a,b=data['plus_timing'],data['control_shared']
    gain=r['arms']['plus_timing']['weighted_recall_at_20']-r['arms']['control_shared']['weighted_recall_at_20']
    folds=[pooled(a[f==k],d[f==k])['weighted_recall_at_20']-
           pooled(b[f==k],d[f==k])['weighted_recall_at_20'] for k in (0,1)]
    ci=paired_interval(a-b,d,f)
    c=r['comparison'];order=r['arms']['plus_timing']['recall']['orders']-r['arms']['control_shared']['recall']['orders']
    if (abs(c['gain']-gain)>1e-12 or not np.allclose(c['fold_gains'],folds,rtol=0,atol=1e-12)
            or abs(c['order_gain']-order)>1e-12
            or not np.allclose(c['descriptive_95_interval'],ci['descriptive_95_interval'],rtol=0,atol=1e-12)
            or c['decision']!=decision(gain,folds,order,ci['descriptive_95_interval'])):
        raise ValueError('Saved gain/uncertainty/decision failed independent replay')
    delta=(a-b).sum(axis=0);contrib=delta/d.sum(axis=0)*WEIGHTS
    for j,o in enumerate(OBJECTIVES):
        if c['net_extra_hits'][o]!=int(delta[j]) or abs(c['weighted_contributions'][o]-contrib[j])>1e-12:
            raise ValueError('Saved objective contribution differs')
    one_order=float(.6/d.sum(axis=0)[2])
    if (abs(c['one_order_hit_weight']-one_order)>1e-12
            or abs(c['gain_less_one_order_hit']-(gain-one_order))>1e-12):
        raise ValueError('Saved order-hit sensitivity differs')
    return r,data


def figures(r,data,diagnostics):
    import plotly.graph_objects as go
    figs=[]
    fig=go.Figure(go.Bar(x=[LABELS[n] for n in LABELS],y=[r['arms'][n]['weighted_recall_at_20'] for n in LABELS],
                        text=[f"{r['arms'][n]['weighted_recall_at_20']:.6f}" for n in LABELS],textposition='outside'))
    fig.update_layout(title='1 · Matched timing replication — new fitting sessions only',yaxis_title='Pooled weighted Recall@20')
    figs.append(fig)
    c=r['comparison'];lo,hi=c['descriptive_95_interval'];g=c['gain']
    fig=go.Figure(go.Scatter(x=[g],y=['Timing − control'],mode='markers',marker={'size':14},
            error_x={'type':'data','symmetric':False,'array':[max(0,hi-g)],'arrayminus':[max(0,g-lo)]}))
    fig.add_vline(x=0,line_dash='dash')
    fig.update_layout(title='2 · Paired descriptive 95% interval — not a leaderboard confidence claim',
                      xaxis_title='Absolute weighted-recall difference')
    figs.append(fig)
    fig=go.Figure(go.Bar(x=['Earlier forward fold','Later forward fold'],y=c['fold_gains'],
                        text=[f'{v:+.6f}' for v in c['fold_gains']],textposition='outside'))
    fig.add_hline(y=0,line_dash='dash');fig.update_layout(title='3 · Does the effect survive both chronological folds?',yaxis_title='Timing − matched control')
    figs.append(fig)
    fig=go.Figure()
    for name in LABELS:fig.add_bar(name=LABELS[name],x=list(OBJECTIVES),y=[r['arms'][name]['recall'][o] for o in OBJECTIVES])
    fig.update_layout(title='4 · Every objective — complete target denominators',barmode='group',yaxis_title='Pooled Recall@20')
    figs.append(fig)
    fig=go.Figure(go.Bar(x=list(OBJECTIVES),y=[c['weighted_contributions'][o] for o in OBJECTIVES],
                        text=[f"{c['net_extra_hits'][o]:+d} net hits" for o in OBJECTIVES],textposition='outside'))
    fig.add_hline(y=0,line_dash='dash');fig.update_layout(title='5 · Which extra hits explain the gain?',yaxis_title='Contribution to weighted difference')
    figs.append(fig)
    fig=go.Figure()
    for fold in (0,1):
        rows=[x for x in diagnostics if x['fold']==fold]
        fig.add_bar(name=f'Training fold {fold}',x=[x['name'].removeprefix('intent_support_') for x in rows],
                    y=[x['nonzero_fraction'] for x in rows])
    fig.update_layout(title='6 · Timing evidence availability — training data only',barmode='group',
                      xaxis={'tickangle':-55},yaxis_title='Fraction of sampled candidate rows with nonzero timing',height=690)
    figs.append(fig)
    fig=go.Figure()
    for fold in (0,1):
        rows=[x for x in diagnostics if x['fold']==fold]
        fig.add_trace(go.Scatter(name=f'Training fold {fold}',mode='markers',
            x=[x['nonzero_fraction'] for x in rows],y=[x['max_abs_corr_control'] for x in rows],
            text=[x['name'] for x in rows],hovertemplate='%{text}<br>nonzero=%{x:.3f}<br>correlation=%{y:.3f}<extra></extra>'))
    fig.update_layout(title='7 · Additional information or duplication? Diagnostic only',
                      xaxis_title='Nonzero fraction on sampled training rows',yaxis_title='Largest absolute correlation with control columns')
    figs.append(fig)
    for f in figs:
        f.update_layout(template='plotly_white',font={'family':'Arial','size':13},margin={'l':70,'r':35,'t':90,'b':80})
    return figs


def make_report(root):
    root=Path(root);out=root/'outputs'
    r,data=validate_saved_result(out)
    diagnostics=json.loads((out/'diagnostics.json').read_text())
    if len(diagnostics)!=48:raise ValueError('Expected 24 timing diagnostics for each training fold')
    figs=figures(r,data,diagnostics)
    import plotly.io as pio
    pieces=[]
    for i,figure in enumerate(figs,1):
        save_json(out/'plots'/f'{i:02d}.json',json.loads(pio.to_json(figure)))
        pieces.append(pio.to_html(figure,full_html=False,include_plotlyjs=(i==1),div_id=f'round02-chart-{i}'))
    c=r['comparison']
    html='''<!doctype html><html><head><meta charset="utf-8"><title>OTTO · Timing replication</title>
<style>body{font:16px Arial;max-width:1180px;margin:35px auto;padding:0 24px;line-height:1.5} .note{border-left:4px solid #667085;padding:12px;background:#f6f7f9}h1{font-size:32px}</style></head><body>'''
    html+=f'<h1>OTTO · Frozen timing-feature replication</h1><p class="note">4,096 fitting sessions; 2,048 forward-validation sessions. Zero overlap with the Round 01 cohort. Same historical window, not an untouched holdout. No Kaggle score produced.</p><h2>{c["decision"]}</h2><p>Timing − control: <strong>{c["gain"]:+.6f}</strong>. One order hit changes the weighted score by {c["one_order_hit_weight"]:.6f}. No feature was automatically promoted.</p>'
    html+=''.join(pieces)+'</body></html>'
    write_once(out/'round02_report.html',html.encode())
    summary=f'''# OTTO Round 02 milestone

Status: ROUND02_REPORT_READY. Result source: result.json and independently replayed evaluation_statistics.npz.

Control: {r['arms']['control_shared']['weighted_recall_at_20']:.6f}; timing: {r['arms']['plus_timing']['weighted_recall_at_20']:.6f}; difference: {c['gain']:+.6f}.

Fold changes: {c['fold_gains']}. Descriptive interval: {c['descriptive_95_interval']}. Decision: {c['decision']}.

Twelve matched native models; existing graphs reused; prior models/data preserved. Source remains {r['source_commit']}. No cloud-resource, GitHub or Kaggle writes. Feature engineering remains open. This outcome is not a competition submission or a promoted feature recipe.
'''
    write_once(out/'MILESTONE_REVIEW.md',summary.encode())
    save_json(out/'report_receipt.json',{'status':'ROUND02_REPORT_READY','charts':7,
         'result_sha256':sha(out/'result.json'),'statistics_recomputed':True,'model_fits':0})
    print('RESULT: ROUND02_REPORT_READY',flush=True)
    return r


if __name__=='__main__':make_report(Path(__file__).resolve().parent)
