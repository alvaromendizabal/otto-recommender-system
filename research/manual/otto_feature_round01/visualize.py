"""Plotly views of saved real experiment results; no fitting or invented result data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html

LABELS = {'control_shared':'134-feature saved control', 'plus_support':'+ 24 support features',
          'plus_timing':'+ 24 timing features', 'plus_both':'+ all 48 features'}

def load_views(root):
    root = Path(root);out = root/'outputs'
    result = json.loads((out/'result.json').read_text())
    if result['status'] != 'ROUND01_SCREEN_COMPLETED':
        raise ValueError('No completed real-data screening result is available')
    arms = list(LABELS)
    with np.load(out/'evaluation_statistics.npz', allow_pickle=False) as z:
        den=z['denominator']
        if den.shape != (512,3) or (den<0).any() or (den>20).any():
            raise ValueError('Saved pooled denominator shape/range error')
        for arm in arms:
            hits=z[arm]
            if hits.shape != den.shape or (hits<0).any() or (hits>den).any():
                raise ValueError('Saved hit counts invalid')
            recomputed=float((hits.sum(axis=0)/den.sum(axis=0)) @ np.array([.1,.3,.6]))
            if abs(recomputed-result['arms'][arm]['weighted_recall_at_20'])>1e-12:
                raise ValueError('Independent pooled metric replay failed')
    figs = []
    fig = go.Figure(go.Bar(x=[LABELS[a] for a in arms],
                          y=[result['arms'][a]['weighted_recall_at_20'] for a in arms],
                          text=[f"{result['arms'][a]['weighted_recall_at_20']:.6f}" for a in arms],
                          textposition='auto'))
    fig.update_layout(title='Matched fitting-only score — NOT Kaggle leaderboard performance',
                      yaxis_title='Pooled weighted Recall@20', xaxis_title='Frozen experiment arm')
    figs.append(fig)
    names, gains, lo, hi = [], [], [], []
    for a in arms[1:]:
        c = result['comparisons'][a+'__minus__control_shared']
        names.append(LABELS[a]);gains.append(c['gain']);l,h=c['descriptive_95_interval']
        lo.append(max(0,c['gain']-l));hi.append(max(0,h-c['gain']))
    fig=go.Figure(go.Scatter(x=names,y=gains,mode='markers',
              error_y={'type':'data','symmetric':False,'array':hi,'arrayminus':lo}))
    fig.add_hline(y=0)
    fig.update_layout(title='Added feature value and descriptive paired uncertainty',
                      yaxis_title='Absolute Recall@20 gain; intervals are unadjusted',
                      xaxis_title='All gains against the same saved control')
    figs.append(fig)
    fig=go.Figure()
    for a in arms[1:]:
        c=result['comparisons'][a+'__minus__control_shared']
        fig.add_trace(go.Bar(name=LABELS[a],x=['Earlier forward fold','Later forward fold'],
                            y=c['fold_gains']))
    fig.add_hline(y=0);fig.update_layout(title='Does the gain survive both chronological folds?',
                                      barmode='group',yaxis_title='Gain versus control')
    figs.append(fig)
    fig=go.Figure()
    for a in arms:
        fig.add_trace(go.Bar(name=LABELS[a],x=['Clicks','Carts','Orders'],
                   y=[result['arms'][a]['recall'][o] for o in ('clicks','carts','orders')]))
    fig.update_layout(title='Which objective moves? Full target denominators retained',
                      barmode='group',yaxis_title='Pooled Recall@20')
    figs.append(fig)
    drop_names, drop_values = [], []
    for name, description in [('plus_both__minus__plus_timing','Contribution of support, with timing present'),
                               ('plus_both__minus__plus_support','Contribution of timing, with support present')]:
        drop_names.append(description);drop_values.append(result['comparisons'][name]['gain'])
    fig=go.Figure(go.Bar(x=drop_values,y=drop_names,orientation='h'))
    fig.add_vline(x=0);fig.update_layout(title='Matched group ablations',xaxis_title='Full 48 minus group-dropped score')
    figs.append(fig)
    diagnostics=json.loads((out/'support_diagnostics.json').read_text())
    features=sorted({r['feature'] for r in diagnostics})
    lookup={(r['fold'],r['feature']):r for r in diagnostics}
    fig=go.Figure(go.Heatmap(x=['Earlier fold TRAIN','Later fold TRAIN'],y=features,
              z=[[lookup[(f,n)]['nonzero_fraction'] for f in (0,1)] for n in features],
              zmin=0,zmax=1))
    fig.update_layout(title='Where does each feature have nonzero evidence? Training rows only',height=1150,
                      xaxis_title='Deterministic sample of at most 64 training sessions/fold')
    figs.append(fig)
    fig=go.Figure()
    for fold in (0,1):
        rows=[r for r in diagnostics if r['fold']==fold]
        fig.add_trace(go.Scatter(x=[r['nonzero_fraction'] for r in rows],
                  y=[r['max_abs_corr_baseline'] for r in rows],mode='markers',
                  text=[r['feature'] for r in rows],name=f'Train fold {fold+1}'))
    fig.update_layout(title='Support versus overlap with the existing representation',
                      xaxis_title='Nonzero row fraction',yaxis_title='Maximum absolute correlation with a control feature')
    figs.append(fig)
    return result, figs


def build_report(root):
    root=Path(root);result,figs=load_views(root)
    headings=''.join(f'<li>{s}</li>' for s in result['limitations'])
    body=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>OTTO Round 01</title>',
          '<style>body{font-family:system-ui;margin:32px auto;max-width:1200px;padding:0 20px}pre{white-space:pre-wrap}</style>',
          '</head><body><h1>OTTO · Observed-intent support</h1>',
          '<p>Real saved fitting-only measurements. Not a new competition submission. No feature is promoted automatically.</p>',
          '<h2>Primary decision</h2><pre>'+json.dumps(result['comparisons'][result['primary_comparison']],indent=2)+'</pre>']
    plot_dir=root/'outputs/plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    for i,f in enumerate(figs):
        (plot_dir/f'{i+1:02d}.json').write_text(f.to_json())
        body.append(to_html(f,full_html=False,include_plotlyjs=True if i==0 else False))
    body+=['<h2>Limitations</h2><ul>'+headings+'</ul></body></html>']
    destination=root/'outputs/round01_report.html'
    destination.write_text('\n'.join(body))
    primary=result['comparisons'][result['primary_comparison']]
    control=result['arms']['control_shared']['weighted_recall_at_20']
    full=result['arms']['plus_both']['weighted_recall_at_20']
    decisions='; '.join(f"{a}: {result['comparisons'][a+'__minus__control_shared']['decision']}"
                        for a in ('plus_support','plus_timing','plus_both'))
    review=f'''# Round 01 milestone review

| Accountability | Measured outcome |
|---|---|
| Attempted | 48 observed-prefix graph-support/timing features on fixed candidates. |
| Completed | 1,024 fitting-session feature construction and three matched new-arm comparisons on two forward folds. |
| Passed | Input identity, complete candidate/target parity, saved-control metric replay and native model reload checks. |
| Underperformed or uncertain | {decisions} |
| Actual metric | Control {control:.6f}; all 48 {full:.6f}; absolute gain {primary['gain']:+.6f}. Fitting-only, not Kaggle. |
| Saved artifacts | 64 feature chunks, 18 native model checkpoints, fold/session hits, source contracts, seven Plotly charts, logs and receipts on persistent space storage. No S3 upload. |
| GitHub updated | No. Existing checkout unchanged at {result['source_commit']}. This package and its results remain outside Git. |
| Learned | Primary assessment: {primary['decision']}. No feature is automatically retained or removed. |
| Next highest-value step | Review temporal signs, order support and paired intervals before deciding larger fitting-only replication or a different feature mechanism. |
| Compute justification | No further run starts automatically. Reuse every completed checkpoint; additional compute must resolve the remaining uncertainty. |

All data use and interpretation limitations remain in result.json and the interactive report.
'''
    (root/'outputs/milestone_review.md').write_text(review)
    return {'status':'ROUND01_REPORT_READY','plots':len(figs),'path':str(destination)}
