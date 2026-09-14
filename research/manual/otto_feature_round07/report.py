"""Saved-artifact-only report for the training-only neighbor feasibility pilot."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from core import sha,save_json,write_once


def render(out: Path):
    result=json.loads((out/'result.json').read_text());d=json.loads((out/'diagnostics.json').read_text());n=json.loads((out/'neighbor_diagnostics.json').read_text())
    if result['status']!='ROUND07_AUDIT_READY' or result['new_model_fits']!=0:raise ValueError('No completed pilot audit')
    with np.load(out/'audit_statistics.npz',allow_pickle=False) as z:
        denom=z['denominator'].sum(0)
        if denom.tolist()!=result['baseline_candidate_oracle']['denominators']:raise ValueError('Audit denominator mismatch')
        if z['baseline_oracle_hits'].sum(0).tolist()!=result['baseline_candidate_oracle']['hits']:raise ValueError('Baseline oracle mismatch')
        for arm in ('session_context','last_anchor'):
            hits=z[arm+'_expanded_oracle_hits'];saved=result['arms'][arm]['expanded_candidate_oracle']
            if hits.sum(0).tolist()!=saved['hits'] or (hits>z['denominator']).any() or (hits<z['baseline_oracle_hits']).any():raise ValueError('Expanded oracle mismatch')
            if (hits-z['baseline_oracle_hits']).sum(0).tolist()!=result['arms'][arm]['net_capped_hits']:raise ValueError('Net oracle gains mismatch')
    labels=('Clicks','Carts','Orders');figs=[]
    f=go.Figure()
    for name,row in [('Baseline 400',result['baseline_candidate_oracle']),('Session-context union',result['arms']['session_context']['expanded_candidate_oracle']),('Last-anchor union',result['arms']['last_anchor']['expanded_candidate_oracle'])]:
        f.add_bar(name=name,x=labels,y=[row['recall'][o] for o in ('clicks','carts','orders')])
    f.update_layout(title='01 · Candidate oracle by action — not ranker performance',barmode='group',yaxis_title='Oracle Recall@20');figs.append(f)
    f=go.Figure()
    for a,name in [('session_context','Session context'),('last_anchor','Last anchor')]:f.add_bar(name=name,x=labels,y=result['arms'][a]['net_capped_hits'])
    f.update_layout(title='02 · Additional metric-capped target hits in expanded pools',barmode='group',yaxis_title='Training-pilot oracle hits');figs.append(f)
    f=go.Figure()
    for a,name in [('session_context','Session context'),('last_anchor','Last anchor')]:f.add_bar(name=name,x=labels,y=result['arms'][a]['supported_baseline_positive_items'])
    f.update_layout(title='03 · Baseline positive items with historical feature support',barmode='group',yaxis_title='Distinct target items');figs.append(f)
    counts=np.asarray(n['neighbor_counts'])
    f=go.Figure()
    for j,name in enumerate(('Session context','Last anchor')):f.add_histogram(x=counts[:,j],name=name,opacity=.6,nbinsx=16)
    f.update_layout(title='04 · Neighbors available per query',barmode='overlay',xaxis_title='Selected historical sessions',yaxis_title='Pilot queries');figs.append(f)
    f=go.Figure(go.Histogram(x=n['query_anchor_counts'],xbins=dict(start=.5,end=4.5,size=1)))
    f.update_layout(title='05 · Observed multi-item context available to this pilot',xaxis_title='Distinct recent query anchors (maximum four)',yaxis_title='Pilot queries');figs.append(f)
    short=[x['name'].replace('session_neighbor_','') for x in d]
    f=go.Figure(go.Bar(x=[x['support_fraction'] for x in d],y=short,orientation='h'))
    f.update_layout(title='06 · Feature coverage on all baseline candidate rows',xaxis_title='Nonzero fraction',height=610,margin=dict(l=260));figs.append(f)
    f=go.Figure(go.Bar(x=[x['max_abs_corr_seven_cached_signals'] for x in d],y=short,orientation='h'))
    f.update_layout(title='07 · Redundancy against seven cached signals only',xaxis_title='Maximum absolute Pearson correlation',height=610,margin=dict(l=260));figs.append(f)
    f=go.Figure()
    for a,name in [('session_context','Session context'),('last_anchor','Last anchor')]:f.add_bar(name=name,x=labels,y=result['arms'][a]['mean_new_candidates_by_objective'])
    f.update_layout(title='08 · New products contributed beyond the original 400',barmode='group',yaxis_title='Mean novel candidates per query');figs.append(f)
    plots=out/'plots';plots.mkdir(exist_ok=True);hashes={};html_parts=[]
    for i,f in enumerate(figs,1):
        f.update_layout(font=dict(family='Arial',size=14),paper_bgcolor='white',plot_bgcolor='white',legend=dict(orientation='h',y=-.2))
        f.update_xaxes(automargin=True);f.update_yaxes(automargin=True)
        path=plots/f'{i:02d}.json';write_once(path,(json.dumps(f.to_plotly_json(),cls=PlotlyJSONEncoder)+'\n').encode());hashes[path.name]=sha(path)
        html_parts.append(f.to_html(full_html=False,include_plotlyjs=True if i==1 else False,div_id=f'round07-{i}'))
    body='''<!doctype html><html><head><meta charset="utf-8"><title>OTTO · Round 07 · Session-neighbor feasibility</title>
<style>body{font:16px/1.6 Arial,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;background:#f6f7f9;color:#182638}h1{line-height:1.2}section{background:white;border-radius:12px;padding:20px;margin:22px 0}.note{border-left:4px solid #536e82;padding:16px;background:white}code{font-size:13px}</style></head><body>
<h1>OTTO / Round 07<br>Historical-session evidence</h1><p class="note"><b>Training-only feasibility study. Zero ranker fits.</b> These charts measure feature support and perfect-ranking candidate coverage, not achieved model accuracy. No leaderboard score or feature promotion.</p>'''
    body+='<p><b>Decision:</b> '+html.escape(result['decision'])+'</p>'
    body+=''.join('<section>'+x+'</section>' for x in html_parts)
    body+='<h2>Interpretation limits</h2>' + ''.join('<p>'+html.escape(x)+'</p>' for x in result['limitations'])+'</body></html>'
    write_once(out/'round07_report.html',body.encode())
    save_json(out/'report_receipt.json',{'status':'ROUND07_REPORT_READY','result_sha256':sha(out/'result.json'),'plot_hashes':hashes,'html_sha256':sha(out/'round07_report.html'),'chart_count':8,'new_model_fits':0})
    return {'charts':8,'new_model_fits':0}

if __name__=='__main__':print(render(Path(__file__).resolve().parent/'outputs'))
