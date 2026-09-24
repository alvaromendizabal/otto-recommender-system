"""Data-free rendering helpers for the OTTO competition-frontier scorecard."""
from __future__ import annotations

import html
import json
from pathlib import Path

import plotly.graph_objects as go
from IPython.display import display


def load(root: Path | None = None):
    root = Path(root or Path.cwd())
    if not (root / "frontier_status_20260923.json").exists():
        root = root.parents[1] / "reports" / "research"
    return json.loads((root / "frontier_status_20260923.json").read_text())


def _svg_bar(labels, values, title, yaxis, width=940, height=430):
    left, bottom, top, right = 90, 85, 65, 30
    plot_w, plot_h = width-left-right, height-top-bottom
    lo, hi = min(0, min(values)), max(values)
    span = max(hi-lo, 1e-9)
    zero_y = top + plot_h * (hi / span)
    bar_w = plot_w/(len(values)*1.8)
    gap = plot_w/len(values)
    esc = lambda x: html.escape(str(x))
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
       f'<title>{esc(title)}</title>','<rect width="100%" height="100%" fill="white"/>',
       '<g font-family="Arial,sans-serif" fill="#172b4d" font-size="14">',
       f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">{esc(title)}</text>',
       f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#999"/>']
    for i,(label,value) in enumerate(zip(labels,values)):
        x=left+gap*(i+.5)-bar_w/2
        yv=top + plot_h*((hi-value)/span)
        y=min(zero_y,yv); h=abs(zero_y-yv)
        p += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h,1):.1f}" fill="#4c78a8"/>',
              f'<text x="{x+bar_w/2:.1f}" y="{(y-7 if value>=0 else y+h+18):.1f}" text-anchor="middle">{value:,.6g}</text>',
              f'<text x="{x+bar_w/2:.1f}" y="{height-40}" text-anchor="middle" font-size="11">{esc(label)}</text>']
    p += [f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle">{esc(yaxis)}</text>','</g></svg>']
    return ''.join(p)


def show_bar(labels, values, title, yaxis, *, width=940, height=430):
    fig=go.Figure(go.Bar(x=labels,y=values,text=[f"{v:,.6g}" for v in values],textposition="outside"))
    fig.update_layout(template=None,width=width,height=height,title=title,yaxis_title=yaxis,
                      margin=dict(l=80,r=35,t=70,b=110))
    display({"application/vnd.plotly.v1+json":fig.to_plotly_json(),
             "image/svg+xml":_svg_bar(labels,values,title,yaxis,width,height),
             "text/plain":title},raw=True)
