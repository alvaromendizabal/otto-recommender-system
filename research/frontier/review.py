"""Recompute public point metrics and draw portable, nonblocking evidence charts."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

WEIGHTS = (0.1, 0.3, 0.6)


def score(hits, denominators):
    if len(hits) != 3 or len(denominators) != 3:
        raise ValueError("Three objective totals required")
    for h, d in zip(hits, denominators, strict=True):
        if type(h) is not int or type(d) is not int or not 0 <= h <= d or d <= 0:
            raise ValueError("Invalid integer totals")
    return math.fsum(w * h / d for w, h, d in zip(WEIGHTS, hits, denominators, strict=True))


def read_evidence(path=None):
    path = Path(path) if path else Path(__file__).with_name("evidence.json")
    data = json.loads(path.read_text())
    if not data["comparable_only_within_study"] or data["baseline_is_submitted_model"]:
        raise ValueError("Comparison scope changed")
    for study in data["studies"]:
        base = study["metrics"][study["baseline"]]
        for row in study["metrics"].values():
            actual = score(row["hits"], row["denominators"])
            if row["denominators"] != base["denominators"]:
                raise ValueError("Unmatched denominators")
            if not math.isclose(actual, row["weighted_recall_at_20"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Point metric mismatch")
        challenger = study["metrics"][study["challenger"]]
        gain = score(challenger["hits"], challenger["denominators"]) - score(base["hits"], base["denominators"])
        reported = study["comparison"]
        if not math.isclose(gain, reported["gain"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Paired gain mismatch")
        lo, hi = reported["descriptive_95_interval"]
        if not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
            raise ValueError("Invalid reported interval")
    latest = data["latest_attempt"]
    if latest["new_models"] != 0 or latest["new_score"] is not None:
        raise ValueError("Blocked attempt cannot claim a new result")
    return data


def primary_rows(data):
    return [(s["title"], 100 * s["comparison"]["gain"],
             *[100 * v for v in s["comparison"]["descriptive_95_interval"]])
            for s in data["studies"]]


def coverage_rows(data):
    study = next(s for s in data["studies"] if s["id"] == "retrieval_frontier")
    return [(name, 100 * item["gain"], *[100 * v for v in item["descriptive_95_interval"]])
            for name, item in (("Candidate coverage (oracle)", study["coverage_comparison"]),
                               ("Achieved recommendations", study["comparison"]))]


def contribution_rows(data):
    study = next(s for s in data["studies"] if s["id"] == "latent_affinity")
    base = study["metrics"][study["baseline"]]
    new = study["metrics"][study["challenger"]]
    return [(name, 100 * w * (n - b) / d) for name, w, n, b, d in zip(
        ("Clicks", "Carts", "Orders"), WEIGHTS, new["hits"], base["hits"],
        base["denominators"], strict=True)]


def svg_chart(title, rows, axis="Weighted recall difference (percentage points)"):
    """Static fallback from the same numbers, not a separate statistical analysis."""
    width, height = 900, 150 + 58 * len(rows)
    low = min(0, *(r[2] if len(r) == 4 else r[1] for r in rows))
    high = max(0, *(r[3] if len(r) == 4 else r[1] for r in rows))
    span = max(high - low, 0.01)
    low -= span * 0.12
    high += span * 0.12
    def x(v):
        return 290 + (v - low) / (high - low) * 480
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{html.escape(title)}</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" fill="#172b4d" font-size="15">',
             f'<text x="24" y="30" font-size="21" font-weight="bold">{html.escape(title)}</text>',
             f'<line x1="{x(0):.2f}" x2="{x(0):.2f}" y1="54" y2="{height-76}" stroke="#8190a5" stroke-dasharray="4 4"/>']
    for i, row in enumerate(rows):
        name, value = row[:2]
        y = 80 + 58 * i
        parts.append(f'<text x="24" y="{y+5}">{html.escape(name)}</text>')
        if len(row) == 4:
            parts.append(f'<line x1="{x(row[2]):.2f}" x2="{x(row[3]):.2f}" y1="{y}" y2="{y}" stroke="#385d8a" stroke-width="3"/>')
        parts += [f'<circle cx="{x(value):.2f}" cy="{y}" r="6" fill="#385d8a"/>',
                  f'<text x="790" y="{y+5}">{value:+.3f}</text>']
    for v in (low + span * 0.12, 0, high - span * 0.12):
        parts.append(f'<text x="{x(v):.2f}" y="{height-49}" text-anchor="middle" font-size="13">{v:+.2f}</text>')
    parts += [f'<text x="530" y="{height-22}" text-anchor="middle">{html.escape(axis)}</text>', '</g></svg>']
    return ''.join(parts)


def show_chart(title, rows):
    import plotly.graph_objects as go
    from IPython.display import display
    values = [r[1] for r in rows]
    errors = None
    if all(len(r) == 4 for r in rows):
        errors = dict(type="data", symmetric=False,
                      array=[r[3] - r[1] for r in rows],
                      arrayminus=[r[1] - r[2] for r in rows])
    fig = go.Figure(go.Scatter(x=values, y=[r[0] for r in rows], mode="markers",
                              marker=dict(size=11), error_x=errors,
                              hovertemplate="%{y}<br>%{x:+.4f} percentage points<extra></extra>"))
    fig.update_layout(template=None, width=900, height=150 + 58 * len(rows),
                      title=title, margin=dict(l=265, r=60, t=65, b=65),
                      xaxis_title="Weighted recall difference (percentage points)",
                      yaxis=dict(autorange="reversed"), showlegend=False)
    fig.add_vline(x=0, line_dash="dot")
    display({"application/vnd.plotly.v1+json": fig.to_plotly_json(),
             "image/svg+xml": svg_chart(title, rows),
             "text/plain": title}, raw=True)


def execute_review():
    """Replay in this interpreter's kernel; refresh only this notebook and receipt."""
    import hashlib
    import sys
    import tempfile
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    folder = Path(__file__).resolve().parent
    path = folder / "01_frontier_review.ipynb"
    nb = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="otto-review-") as temporary:
        spec = Path(temporary) / "frontier"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "OTTO review", "language": "python"}))
        manager = KernelManager(kernel_name="frontier", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        try:
            NotebookClient(nb, km=manager, timeout=120, resources={"metadata": {"path": str(folder)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
    nb.metadata.pop("language_info", None)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    if [c.execution_count for c in cells] != list(range(1, 8)):
        raise ValueError("Incomplete notebook execution")
    outputs = [o for c in cells for o in c.outputs]
    charts = [o for o in outputs if "application/vnd.plotly.v1+json" in o.get("data", {})]
    if len(charts) != 3 or any("image/svg+xml" not in o.data for o in charts):
        raise ValueError("Missing inline charts")
    if "FRONTIER_REVIEW_COMPLETE" not in str(cells[-1].outputs):
        raise ValueError("Missing post-Plotly sentinel")
    path.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")) + "\n")
    nbformat.validate(nbformat.read(path, as_version=4))
    receipt = {"status": "PUBLIC_REVIEW_EXECUTED", "code_cells": 7, "inline_plotly": 3,
               "inline_svg": 3, "sentinel": True, "no_project_fits": True,
               "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "evidence_sha256": hashlib.sha256((folder / "evidence.json").read_bytes()).hexdigest(),
               "note": "Analysis replay from returned aggregate evidence, not experimental rerun."}
    (folder / "execution.json").write_text(json.dumps(receipt, separators=(",", ":")) + "\n")
    (folder / "chart_0.svg").write_text(charts[0].data["image/svg+xml"])
    print(json.dumps(receipt))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute_review()
    else:
        print(json.dumps({"studies_verified": len(read_evidence()["studies"])}))
