"""Data-free employer-facing replay for the OTTO neural-frontier snapshot."""
from __future__ import annotations
import html, json
from pathlib import Path
import plotly.graph_objects as go
from IPython.display import display

def load(root: Path | None = None):
    root = Path(root or Path.cwd())
    folder = root if (root / "status.json").exists() else root / "research/neural_stack"
    return json.loads((folder / "status.json").read_text()), json.loads((folder / "reproduction_matrix.json").read_text())

def _svg_bar(labels, values, title, yaxis, width=900, height=430):
    left,bottom,top,right=90,80,65,30
    plot_w=width-left-right; plot_h=height-top-bottom
    vmax=max(values)*1.08; bar_w=plot_w/(len(values)*1.8); gap=plot_w/len(values)
    def esc(x): return html.escape(str(x))
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',f'<title>{esc(title)}</title>','<rect width="100%" height="100%" fill="white"/>','<g font-family="Arial,sans-serif" fill="#172b4d" font-size="14">',f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">{esc(title)}</text>']
    for i,(label,value) in enumerate(zip(labels,values)):
        x=left+gap*(i+.5)-bar_w/2; h=(value/vmax)*plot_h; y=top+plot_h-h
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#4c78a8"/>',f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle">{value:,.6g}</text>',f'<text x="{x+bar_w/2:.1f}" y="{height-42}" text-anchor="middle" font-size="12">{esc(label)}</text>']
    parts += [f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle">{esc(yaxis)}</text>','</g></svg>']
    return ''.join(parts)

def show_bar(labels, values, title, yaxis, width=900, height=430):
    fig=go.Figure(go.Bar(x=labels,y=values,text=[f"{v:,.6g}" for v in values],textposition="outside"))
    fig.update_layout(template=None,width=width,height=height,title=title,yaxis_title=yaxis,margin=dict(l=80,r=35,t=70,b=105))
    display({"application/vnd.plotly.v1+json":fig.to_plotly_json(),"image/svg+xml":_svg_bar(labels,values,title,yaxis,width,height),"text/plain":title},raw=True)
