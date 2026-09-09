"""Build the public Plotly figures from verified research evidence, without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
COLORS = {"fusion": "#64748B", "core": "#3566C5", "selected": "#087F73"}
LABELS = {
    "fusion": "Candidate fusion",
    "core": "Compact ranker · 28 features",
    "selected": "Selected ranker · 102 features",
}
FAMILIES = {
    "history": "Item history",
    "context": "Session context",
    "repeat": "Repeat intent",
    "graph": "Graph affinity",
    "interaction": "Intent interactions",
    "source": "Direct source features",
}
REPOSITORY = "https://github.com/alvaromendizabal/otto-recommender-system"


@dataclass(frozen=True)
class Chart:
    name: str
    title: str
    caption: str
    figure: Any
    rows: list[dict[str, Any]]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def read_evidence(root: Path) -> dict[str, Any]:
    """Reject changed evidence, mixed model lineages, and incorrect headline arithmetic."""
    directory = root / "reports/research"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Research evidence checksum mismatch: {name}")
    names = ("evaluation", "ablations", "screening", "interpretation", "audit", "evaluation_seal")
    data = {name: json.loads((directory / f"{name}.json").read_text()) for name in names}
    data["manifest"] = manifest
    for name in ("evaluation", "ablations", "interpretation", "audit", "evaluation_seal"):
        if data[name]["seal_id"] != manifest["seal_id"]:
            raise ValueError(f"Mixed research lineage: {name}")
    if any(data[name]["status"] != "passed" for name in ("evaluation", "interpretation", "audit")):
        raise ValueError("Portfolio requires successful evaluation, interpretation and audit")
    if data["evaluation_seal"]["evaluation_labels_consulted"] is not False:
        raise ValueError("Model selection consulted evaluation labels")
    evaluation = data["evaluation"]
    denominators = None
    for name, score in evaluation["scores"].items():
        actual = 0.0
        current = []
        for objective, weight in WEIGHTS.items():
            row = score["objectives"][objective]
            recall = row["hits"] / row["denominator"]
            if not math.isclose(recall, row["recall_at_20"], abs_tol=1e-12):
                raise ValueError(f"Incorrect recall: {name}/{objective}")
            actual += weight * recall
            current.append(row["denominator"])
        if denominators is not None and denominators != current:
            raise ValueError("Model comparisons use different target denominators")
        denominators = current
        if not math.isclose(actual, score["weighted_recall_at_20"], abs_tol=1e-12):
            raise ValueError(f"Incorrect weighted score: {name}")
    for name, interval in evaluation["paired_intervals"].items():
        gain = (
            evaluation["scores"]["selected"]["weighted_recall_at_20"]
            - evaluation["scores"][name]["weighted_recall_at_20"]
        )
        if not math.isclose(gain, interval["absolute_gain"], abs_tol=1e-12):
            raise ValueError(f"Incorrect paired gain: {name}")
    counts = manifest["features"]
    if any(
        sum(f[stage] for f in counts["families"].values()) != counts[stage]
        for stage in ("engineered", "screened", "final")
    ):
        raise ValueError("Feature family counts do not reconcile")
    comparison = root / "reports/robustness/comparison.json"
    if comparison.is_file():
        data["robustness"] = json.loads(comparison.read_text())
        for name, expected in data["robustness"]["source_files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != expected:
                raise ValueError(f"Temporal comparison checksum mismatch: {name}")
    return data


def style(fig: Any, title: str, subtitle: str, *, height: int = 470, left: int = 80) -> Any:
    fig.update_layout(
        template="plotly_white",
        width=1120,
        height=height,
        title={
            "text": f"<b>{title}</b><br><sup>{subtitle}</sup>",
            "x": 0.03,
            "y": 1 - 44 / height,
            "yanchor": "top",
        },
        font={"family": "Arial, sans-serif", "size": 17, "color": "#172B4D"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": left, "r": 105, "t": 110, "b": 85},
        legend={"orientation": "h", "x": 0, "y": -0.18, "font": {"size": 15}},
        hoverlabel={"font_size": 16},
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor="#E7ECF2", zerolinecolor="#CAD3DE", automargin=True)
    fig.update_yaxes(gridcolor="#E7ECF2", zerolinecolor="#CAD3DE", automargin=True)
    return fig


def make_charts(data: dict[str, Any]) -> list[Chart]:
    go = importlib.import_module("plotly.graph_objects")
    evaluation, ablations = data["evaluation"], data["ablations"]
    interpretation, counts = data["interpretation"], data["manifest"]["features"]
    charts: list[Chart] = []
    models = ("selected", "core", "fusion")
    rows = [
        {"Model": LABELS[n], "Weighted Recall@20": evaluation["scores"][n]["weighted_recall_at_20"]}
        for n in models
    ]
    values = [r["Weighted Recall@20"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[r["Model"] for r in rows],
            orientation="h",
            marker_color=[COLORS[n] for n in models],
            text=[f"{v:.5f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Weighted Recall@20: %{x:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Weighted Recall@20 · higher is better", range=[0, 0.67])
    fig.update_yaxes(autorange="reversed")
    title = "Better features improve the matched ranking baseline"
    charts.append(
        Chart(
            "results",
            title,
            "All three systems use the same 400-candidate pools and 432,492 temporal evaluation "
            "sessions. The selected ranker gains 1.949 percentage points over the compact ranker "
            "and 4.915 points over fusion. These are offline results; Kaggle scoring is separate.",
            style(
                fig,
                title,
                "Reserved evaluation · 432,492 sessions · identical candidate pools",
                height=430,
                left=315,
            ),
            rows,
        )
    )

    rows = [
        {
            "Objective": o.capitalize(),
            **{LABELS[n]: evaluation["scores"][n]["objectives"][o]["recall_at_20"] for n in models},
        }
        for o in WEIGHTS
    ]
    fig = go.Figure()
    for name in reversed(models):
        values = [r[LABELS[name]] for r in rows]
        fig.add_bar(
            name=LABELS[name],
            x=[r["Objective"] for r in rows],
            y=values,
            marker_color=COLORS[name],
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Recall@20: %{y:.6f}<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Recall@20", range=[0, 0.79])
    title = "Orders drive the gain; clicks and carts retain tradeoffs"
    charts.append(
        Chart(
            "objectives",
            title,
            "The official weighted metric assigns 10% to clicks, 30% to carts and 60% to orders. "
            "Fusion retains the best click and cart recall. The selected ranker improves all three "
            "objectives over the compact ranker. No hybrid was chosen using these "
            "evaluation results.",
            style(fig, title, "Reserved evaluation · each action is scored separately", height=510),
            rows,
        )
    )

    quality = counts["engineered"] - sum(
        data["screening"]["rejections"][k] for k in ("constant", "near_constant", "duplicate")
    )
    rows = [
        {"Stage": name, "Features": value}
        for name, value in [
            ("Engineered formulas", counts["engineered"]),
            ("Pass quality checks", quality),
            ("Fit-only shortlist", counts["screened"]),
            ("Selected model", counts["final"]),
        ]
    ]
    fig = go.Figure(
        go.Bar(
            x=[r["Features"] for r in rows],
            y=[r["Stage"] for r in rows],
            orientation="h",
            marker_color=["#8D9DB3", "#64748B", COLORS["core"], COLORS["selected"]],
            text=[f"{r['Features']:,}" for r in rows],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} features<extra></extra>",
        )
    )
    fig.update_xaxes(title="Number of feature formulas", range=[0, 1650])
    fig.update_yaxes(autorange="reversed")
    title = "A broad search, followed by explicit rejection"
    charts.append(
        Chart(
            "feature_selection",
            title,
            "Quality checks remove 205 constant, near-constant or duplicated formulas. "
            "Fitting-only "
            "utility screening, correlation pruning and a capacity limit retain 128. "
            "Selection then "
            "removes 26 direct source features, leaving 102. Features below the capacity limit are "
            "not proved useless; every rejection reason is preserved in the catalog.",
            style(fig, title, "1,482 candidates → 128 shortlisted → 102 selected", left=265),
            rows,
        )
    )

    full = ablations["variants"]["full"]["weighted_recall_at_20"]
    rows = sorted(
        [
            {
                "Removed family": FAMILIES[f],
                "Features remaining": ablations["variants"][f"without_{f}"]["features"],
                "Change (pp)": 100
                * (ablations["variants"][f"without_{f}"]["weighted_recall_at_20"] - full),
            }
            for f in FAMILIES
        ],
        key=lambda r: float(r["Change (pp)"]),
        reverse=True,
    )
    values = [r["Change (pp)"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[r["Removed family"] for r in rows],
            orientation="h",
            marker_color=[COLORS["selected"] if v > 0 else COLORS["core"] for v in values],
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Remove %{y}<br>Change: %{x:+.4f} pp<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#7E8DA1", line_width=1)
    fig.update_xaxes(title="Change in weighted Recall@20 (percentage points)", range=[-0.8, 2.95])
    fig.update_yaxes(autorange="reversed")
    title = "Removing direct source features helped most"
    charts.append(
        Chart(
            "ablations",
            title,
            "Each bar compares a refitted leave-one-family-out model with the 128-feature model "
            "on the same 20,000 selection sessions. Positive means removal helped. Removing direct "
            "source features adds 2.312 points; removing context adds only 0.094. "
            "Other removals hurt. "
            "Graph and interaction features still carry retrieval information. These are selection "
            "comparisons, not final evaluation gains or independent causal family effects.",
            style(
                fig,
                title,
                "Selection only · 20,000 sessions · reference: 128-feature model",
                height=550,
                left=265,
            ),
            rows,
        )
    )

    rows = sorted(
        [
            {
                "Family": FAMILIES[f],
                "Mean drop (pp)": 100 * r["mean_weighted_recall_drop"],
                "Shuffle 1 (pp)": 100 * r["repeat_drops"][0],
                "Shuffle 2 (pp)": 100 * r["repeat_drops"][1],
            }
            for f, r in interpretation["group_permutation"].items()
        ],
        key=lambda r: float(r["Mean drop (pp)"]),
        reverse=True,
    )
    means = [r["Mean drop (pp)"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=means,
            y=[r["Family"] for r in rows],
            orientation="h",
            marker_color=COLORS["selected"],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [
                    max(r["Shuffle 1 (pp)"], r["Shuffle 2 (pp)"]) - r["Mean drop (pp)"]
                    for r in rows
                ],
                "arrayminus": [
                    r["Mean drop (pp)"] - min(r["Shuffle 1 (pp)"], r["Shuffle 2 (pp)"])
                    for r in rows
                ],
            },
            hovertemplate="%{y}<br>Mean drop: %{x:.3f} pp<extra></extra>",
        )
    )
    fig.update_xaxes(
        title="Weighted Recall@20 lost after shuffling (percentage points)", range=[-0.4, 8]
    )
    fig.update_yaxes(autorange="reversed")
    title = "Repeat intent carries the strongest diagnostic signal"
    charts.append(
        Chart(
            "feature_dependence",
            title,
            "Whole-query shuffling disrupts one feature family while keeping the trained model "
            "fixed. Larger drops indicate greater model dependence in this sample. Whiskers show "
            "the range of two shuffles, not confidence intervals. The diagnostic can break real "
            "feature relationships and does not measure causal effects.",
            style(
                fig,
                title,
                "Post-selection diagnostic · 1,000 selection sessions · two shuffles",
                height=490,
                left=245,
            ),
            rows,
        )
    )

    rows = [
        {
            "Representation": label,
            "p50 (ms)": interpretation["feature_benchmark"][name]["p50_ms"],
            "p95 (ms)": interpretation["feature_benchmark"][name]["p95_ms"],
        }
        for name, label in [
            ("broad_catalog", "Broad catalog · 1,482"),
            ("screened", "Shortlist · 128"),
            ("selected_models", "Selected · 102"),
        ]
    ]
    fig = go.Figure()
    for metric, color in [("p50 (ms)", COLORS["core"]), ("p95 (ms)", COLORS["selected"])]:
        values = [r[metric] for r in rows]
        fig.add_bar(
            x=[r["Representation"] for r in rows],
            y=values,
            name=metric,
            marker_color=color,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.3f} ms<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Warm candidate + feature time (ms)", range=[0, 27])
    title = "Compute only the features the selected model needs"
    charts.append(
        Chart(
            "feature_cost",
            title,
            "Matched measurements use 64 selection queries, the same machine and a 400-item "
            "candidate budget. Warm candidate generation plus feature computation has p95 "
            "22.10 ms for the broad catalog and 6.42 ms for selected features. This excludes "
            "model prediction, network transfer and online serving.",
            style(
                fig,
                title,
                "Matched warm benchmark · 64 queries · same process and hardware",
                height=510,
            ),
            rows,
        )
    )

    rows = [
        {
            "Observed events": label,
            "Sessions": evaluation["prefix_slices"][label]["sessions"],
            "Selected - fusion (pp)": 100
            * (
                evaluation["prefix_slices"][label]["selected"]["weighted_recall_at_20"]
                - evaluation["prefix_slices"][label]["fusion"]["weighted_recall_at_20"]
            ),
        }
        for label in ("1", "2-5", "6-20", "21+")
    ]
    values = [r["Selected - fusion (pp)"] for r in rows]
    fig = go.Figure(
        go.Scatter(
            x=values,
            y=[
                f"{r['Observed events']} {'event' if r['Observed events'] == '1' else 'events'}"
                f" · {r['Sessions']:,} sessions"
                for r in rows
            ],
            mode="markers+text",
            marker={"color": COLORS["selected"], "size": 15},
            text=[f"{v:+.3f} pp" for v in values],
            textposition="middle right",
            cliponaxis=False,
            hovertemplate="%{y}<br>Gain: %{x:+.4f} pp<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#7E8DA1")
    fig.update_xaxes(
        title="Selected - fusion weighted Recall@20 (percentage points)", range=[-1.5, 22]
    )
    fig.update_yaxes(autorange="reversed")
    title = "The overall gain is uneven across shopping sessions"
    charts.append(
        Chart(
            "session_slices",
            title,
            "The selected model loses 0.268 points to fusion for prefixes containing 2-5 events. "
            "The largest gain occurs in the smaller 21+ event group. Slice scores have different "
            "target mixes and are descriptive: their unweighted average is not the pooled score. "
            "No slice-specific model was selected after inspecting this evaluation.",
            style(
                fig,
                title,
                "Reserved evaluation · grouped by observed prefix length",
                height=470,
                left=360,
            ),
            rows,
        )
    )
    if "robustness" in data:
        replication = data["robustness"]
        records = replication["rows"]
        rows = []
        for row in records:
            interval = row["paired_intervals"]["core"]
            rows.append({
                "Window": row["window"].title(), "Seed": row["model_seed"],
                "Sessions": row["sessions"],
                "Selected Recall@20": row["scores"]["selected"]["weighted_recall_at_20"],
                "Compact Recall@20": row["scores"]["core"]["weighted_recall_at_20"],
                "Gain (pp)": 100 * interval["absolute_gain"],
                "95% lower (pp)": 100 * interval["gain_interval"][0],
                "95% upper (pp)": 100 * interval["gain_interval"][1],
            })
        fig = go.Figure()
        for window, color in (("Early", "#3566C5"), ("Middle", "#087F73"),
                              ("Reference", "#8B5FBF")):
            subset = [r for r in rows if r["Window"] == window]
            fig.add_scatter(
                name=window, mode="markers", marker={"color": color, "size": 12},
                y=[f"{r['Window']} · {r['Seed']}" for r in subset],
                x=[r["Gain (pp)"] for r in subset],
                error_x={"type": "data", "symmetric": False,
                         "array": [r["95% upper (pp)"] - r["Gain (pp)"] for r in subset],
                         "arrayminus": [r["Gain (pp)"] - r["95% lower (pp)"] for r in subset]},
                hovertemplate=("%{y}<br>Selected minus compact: %{x:.3f} pp"
                               "<extra>%{fullData.name}</extra>"),
            )
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B")
        fig.update_xaxes(title="Gain in weighted Recall@20 · percentage points", range=[0, 2.5])
        fig.update_yaxes(autorange="reversed")
        title = "Feature gains repeat across all nine audited runs"
        charts.append(Chart(
            "robustness", title,
            "Three temporal windows and three model seeds. All nine selected rankers beat their "
            "matched compact controls by 1.791 to 2.180 percentage points. Error bars are paired "
            "95% session-bootstrap intervals conditional on each fitted pair. Seeds share a "
            "cohort within each window; these are not nine independent datasets. The largest "
            "absolute score is 0.590759 in the middle window, whose sessions differ from the "
            "reference cohort. The original reference seed remains frozen for submission.",
            style(fig, title, "All planned outcomes retained · frozen seed policy",
                  height=710, left=250), rows,
        ))
    return charts


def table_html(rows: list[dict[str, Any]]) -> str:
    head = "".join(f"<th scope='col'>{escape(k)}</th>" for k in rows[0])
    body = []
    for row in rows:
        cells = [f"{v:.6f}" if isinstance(v, float) else str(v) for v in row.values()]
        body.append("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def html_report(charts: list[Chart]) -> str:
    sections = []
    for index, chart in enumerate(charts):
        fig = importlib.import_module("plotly.graph_objects").Figure(chart.figure)
        subtitle = str(fig.layout.title.text).split("<br>", 1)[1]
        fig.update_layout(width=None, autosize=True, title={"text": subtitle}, margin={"t": 75})
        fragment = fig.to_html(
            full_html=False,
            include_plotlyjs=index == 0,
            div_id=f"otto-{chart.name}",
            config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        )
        sections.append(
            f"<section id='{chart.name}'><h2>{escape(chart.title)}</h2>"
            f"<p>{escape(chart.caption)}</p><div class='chart'>{fragment}</div>"
            f"<details><summary>View exact values</summary>{table_html(chart.rows)}"
            "</details></section>"
        )
    navigation = " · ".join(
        f"<a href='#{c.name}'>{escape(c.name.replace('_', ' ').title())}</a>" for c in charts
    )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OTTO · Research results explorer</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#172b4d;
font:17px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:48px 24px}
header{padding:0 12px 24px}h1{font-size:clamp(30px,5vw,48px);line-height:1.15;letter-spacing:-1px}
h2{font-size:23px;line-height:1.3}p{max-width:90ch}a{color:#087f73;text-underline-offset:3px}
nav{line-height:2.2}section{background:white;padding:28px;margin:24px 0;border:1px solid #dce3ec;
border-radius:12px;scroll-margin-top:20px}.chart{overflow-x:auto}.plotly-graph-div{min-width:760px}
summary{cursor:pointer;color:#087f73;font-weight:600}details{overflow:auto}table{border-collapse:collapse;
font-size:14px;width:100%;margin-top:16px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e7ecf2}
th{background:#f4f6f9}.eyebrow{font-size:13px;letter-spacing:2px;font-weight:700;color:#087f73}
@media(max-width:640px){main{padding:24px 12px}section{padding:16px}h2{font-size:21px}}
</style></head><body><main><header><div class="eyebrow">OTTO / MACHINE LEARNING RESEARCH</div>
<h1>From shopping events to measured recommendations</h1>
<p>Explore the controlled feature study: compare ranking quality, inspect feature decisions,
and examine where the model helps. Hover for values, click a legend to compare models,
and use each chart's toolbar to zoom or export an image. Exact values are also available
as accessible tables.</p><p><strong>Scope:</strong> the original 432,492-session reference
evaluation plus all nine audited runs across three temporal windows. Selection,
diagnostic and replication cohorts are labeled separately. These are offline results.
The report includes its Plotly library; it needs no internet connection, Python,
dataset download or AWS account.</p>""" + (
        f"<p><a href='{REPOSITORY}'>Repository</a> · "
        f"<a href='{REPOSITORY}/blob/main/docs/PORTFOLIO.md'>Full case study</a> · "
        f"<a href='{REPOSITORY}/blob/main/notebooks/09_controlled_feature_study.ipynb'>"
        f"Executed research notebook</a></p><nav>{navigation}</nav></header>"
        + "".join(sections)
        + "</main></body></html>\n"
    )


def write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def report_archive(html: str) -> bytes:
    """Package one offline report; fixed ZIP metadata avoids timestamp-only changes."""
    buffer = io.BytesIO()
    entry = zipfile.ZipInfo("otto-research-report.html")
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, html.encode(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def build(root: Path, output: Path, *, check: bool = False, previews: Path | None = None) -> None:
    started = time.perf_counter()
    data = read_evidence(root)
    charts = make_charts(data)
    contract = {
        "research_manifest_sha256": digest(root / "reports/research/manifest.json"),
        "generator_sha256": digest(Path(__file__)),
        "analysis_lock_sha256": digest(root / "notebooks/requirements.txt"),
        "robustness_comparison_sha256": digest(root / "reports/robustness/comparison.json"),
        "seal_id": data["manifest"]["seal_id"],
    }
    specs = {f"{c.name}.json": canonical(json.loads(c.figure.to_json())) for c in charts}
    if check:
        receipt = json.loads((output / "manifest.json").read_text())
        if receipt["contract"] != contract:
            raise ValueError("Portfolio figures are stale: evidence, code or analysis lock changed")
        expected_names = {
            "index.html",
            "otto-research-report.zip",
            *specs,
            *(f"{c.name}.svg" for c in charts),
        }
        if set(receipt["files"]) != expected_names:
            raise ValueError("Portfolio receipt has an incomplete artifact set")
        for name, expected in receipt["files"].items():
            if digest(output / name) != expected:
                raise ValueError(f"Portfolio output checksum mismatch: {name}")
        if any((output / name).read_text() != spec for name, spec in specs.items()):
            raise ValueError("Committed chart values differ from verified research evidence")
        if (output / "index.html").read_text() != html_report(charts):
            raise ValueError("Interactive report differs from verified chart definitions")
        with zipfile.ZipFile(output / "otto-research-report.zip") as archive:
            if (
                archive.namelist() != ["otto-research-report.html"]
                or archive.read("otto-research-report.html") != (output / "index.html").read_bytes()
            ):
                raise ValueError("Download archive differs from the verified HTML report")
    else:
        for chart in charts:
            stage = time.perf_counter()
            write(output / f"{chart.name}.json", specs[f"{chart.name}.json"].encode())
            write(output / f"{chart.name}.svg", chart.figure.to_image(format="svg"))
            if previews is not None:
                write(previews / f"{chart.name}.png", chart.figure.to_image(format="png"))
            print(
                canonical(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "event": "figure_complete",
                        "figure": chart.name,
                        "stage_elapsed_seconds": time.perf_counter() - stage,
                        "total_elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        html = html_report(charts)
        write(output / "index.html", html.encode())
        write(output / "otto-research-report.zip", report_archive(html))
        names = [
            "index.html",
            "otto-research-report.zip",
            *specs,
            *(f"{c.name}.svg" for c in charts),
        ]
        receipt = {
            "contract": contract,
            "files": {n: digest(output / n) for n in sorted(names)},
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "figures": len(charts),
            "status": "passed",
            "plotly": importlib.import_module("plotly").__version__,
        }
        write(output / "manifest.json", canonical(receipt).encode())
    print(
        canonical(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "portfolio_verified",
                "figures": len(charts),
                "total_elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/portfolio"))
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build(root, args.output_dir.resolve(), check=args.check, previews=args.preview_dir)


if __name__ == "__main__":
    main()
