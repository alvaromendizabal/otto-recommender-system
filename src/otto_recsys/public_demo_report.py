"""Render a self-contained, offline report for the public synthetic walkthrough.

Rendering never fits a model or recalculates recommendations. Every displayed
result comes from the demo payload; future targets appear only in evaluation.
No third-party packages, remote assets, or generated JavaScript are required.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ACTIONS = ("clicks", "carts", "orders")
_LABELS = {"clicks": "Clicks", "carts": "Carts", "orders": "Orders"}
_CSS = """
:root{color-scheme:light;--paper:#f5f3ed;--card:#fff;--ink:#162c31;
 --muted:#52646a;--line:#d9e0dd;--accent:#075e52;--pale:#edf5f1;--warm:#f8efe1}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);
 color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--accent);text-underline-offset:4px}a:hover{text-decoration-thickness:2px}
:focus-visible{outline:3px solid #1d806b;outline-offset:4px}
.skip{position:absolute;left:20px;top:-80px;background:white;padding:10px;z-index:5}
.skip:focus{top:10px}.wrap{width:min(1160px,calc(100% - 48px));margin:auto}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px;
 padding:24px 0;border-bottom:1px solid var(--line)}.brand{font-weight:750;letter-spacing:.13em}
.topbar nav{display:flex;flex-wrap:wrap;gap:22px;font-size:13px}
.topbar nav a{color:var(--muted);text-decoration:none}.hero{padding:58px 0 32px}
.eyebrow{font:700 12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;
 letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.badge{display:inline-block;background:var(--pale);border:1px solid #b9d4c8;
 border-radius:4px;padding:5px 10px;margin-bottom:17px}
h1{font:400 clamp(38px,5.5vw,64px)/1.08 Georgia,"Times New Roman",serif;
 letter-spacing:-.045em;max-width:830px;margin:0 0 22px}
h2{font:400 30px/1.2 Georgia,"Times New Roman",serif;letter-spacing:-.025em;margin:0 0 12px}
h3{font-size:16px;line-height:1.4;margin:0 0 12px}p{margin:0 0 16px}
.lead{font-size:18px;line-height:1.6;max-width:760px;color:var(--muted)}
.scope{max-width:800px;color:var(--muted);font-size:13px}.section{padding:34px 0}
.section-head{display:flex;justify-content:space-between;align-items:end;gap:24px;
 margin-bottom:20px}
.section-head p{color:var(--muted);max-width:620px;margin:0}
.stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:24px}
.stat{font-variant-numeric:tabular-nums}.stat .value{font-size:38px;letter-spacing:-.05em;
 line-height:1.2;font-weight:650;margin:10px 0}.stat p{margin:0;font-size:13px;color:var(--muted)}
.stat .label{font-size:12px;color:var(--muted);font-weight:650}
.stat.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.stat.primary .label,.stat.primary p{color:#d6ece4}
.pipeline{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(5,1fr);
 border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}
.pipeline li{padding:20px 17px;border-right:1px solid var(--line)}
.pipeline li:last-child{border:0}.step-number{font:12px ui-monospace,monospace;color:var(--accent)}
.pipeline strong{display:block;font-size:14px;margin:9px 0 5px}
.pipeline small{display:block;color:var(--muted);font-size:12px;line-height:1.5}
.state{display:inline-block;font-size:12px;letter-spacing:.04em;margin-top:10px;
 color:var(--accent);background:var(--pale);padding:2px 6px;border-radius:3px}
.table-scroll{overflow-x:auto;max-width:100%;border:1px solid var(--line);border-radius:9px;
 background:#fff}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
caption{text-align:left;padding:16px 18px;font-size:13px;color:var(--muted)}
th,td{padding:12px 18px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;letter-spacing:.03em;color:var(--muted);background:#f8faf8}
tr:last-child td{border-bottom:0}td{font-size:13px}.number{text-align:right;white-space:nowrap}
.method-note{font-size:12px;color:var(--muted);margin-top:12px;max-width:900px}
.controls{display:flex;align-items:end;gap:18px;padding:20px 24px;border:1px solid var(--line);
 border-radius:10px;background:var(--card);margin:20px 0}
.controls label{display:block;font-size:12px;font-weight:650;margin-bottom:6px}
.controls select{min-width:180px;max-width:100%;background:#fff;border:1px solid #9cafa6;
 border-radius:5px;padding:9px 36px 9px 12px;color:var(--ink);font:inherit}
.live-note{margin:0 0 7px auto;color:var(--muted);font-size:12px}
.session-header{display:flex;justify-content:space-between;align-items:start;gap:20px;
 padding:4px 0 16px}.session-header h3{margin:3px 0 0;font-size:21px}
.session-meta{text-align:right;color:var(--muted);font-size:12px}
.prefix-card{padding:22px 24px;background:var(--pale);border:1px solid #ccded5;
 border-radius:10px;margin-bottom:20px}.prefix-card p{font-size:13px;color:var(--muted)}
.events{list-style:none;display:flex;flex-wrap:wrap;gap:9px;padding:0;margin:14px 0 0}
.events li{border:1px solid #c1d6cb;border-radius:6px;background:#fff;padding:8px 12px}
.events strong{display:block;font-size:13px}.events small{font-size:12px;color:var(--muted)}
.table-head{display:flex;justify-content:space-between;align-items:baseline;gap:20px;
 margin:24px 0 10px}
.table-head h3{margin:0}.table-head span{color:var(--muted);font-size:12px}
.ranking{min-width:630px}.rank{width:54px;color:var(--muted)}
.item-id{font-weight:650;white-space:nowrap}
.score{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.tags{display:flex;flex-wrap:wrap;gap:5px}.tag{display:inline-block;border-radius:4px;
 background:#eef3f1;color:#345b50;padding:2px 7px;font-size:12px}
.explanation{max-width:500px}.features{color:var(--muted);font-size:12px;margin-top:5px}
.evaluation{border:1px solid #dccbb0;background:var(--warm);border-radius:10px;margin-top:20px}
.evaluation summary{cursor:pointer;padding:20px 24px;font-weight:650}
.evaluation summary small{font-weight:400;color:#6f5b3b;margin-left:8px}
.eval-body{padding:0 24px 24px}.eval-body p{font-size:13px;color:#655945}
.eval-body table{background:#fff}.eval-body .table-scroll{border-color:#e2d3bc}
.evaluation-counts{display:flex;flex-wrap:wrap;gap:22px;margin:16px 0;font-size:13px}
.evaluation-counts strong{font-variant-numeric:tabular-nums}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.provenance dl{margin:0}
.provenance dt{font-size:12px;color:var(--muted);margin-top:12px}
.provenance dt:first-child{margin-top:0}.provenance dd{margin:3px 0 0;overflow-wrap:anywhere}
.hash{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}
.provenance p{font-size:13px;color:var(--muted)}code{font-size:.9em;overflow-wrap:anywhere}
footer{padding:28px 0 40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);
 margin-top:20px;display:flex;justify-content:space-between;gap:24px}
[hidden]{display:none!important}.empty{padding:24px;color:var(--muted)}
@media(max-width:760px){.wrap{width:calc(100% - 32px)}.topbar{align-items:start}
 .topbar nav{gap:8px 15px;justify-content:end}.hero{padding:36px 0 22px}
 .stat-grid{grid-template-columns:1fr}.stat .value{font-size:34px}
 .pipeline{grid-template-columns:1fr}.pipeline li{border-right:0;
 border-bottom:1px solid var(--line)}
 .pipeline strong{display:inline;margin-left:10px}.pipeline small{margin:5px 0 0 26px}
 .state{margin-left:26px}.section-head{display:block}.section-head p{margin-top:12px}
 .controls{flex-wrap:wrap;padding:18px;gap:12px}.controls>div{flex:1 1 160px}
 .controls select{width:100%;min-width:0}.live-note{margin:0;flex-basis:100%}
 .columns{grid-template-columns:1fr}.card{padding:20px}.session-header{display:block}
 .session-meta{text-align:left;margin-top:8px}.prefix-card{padding:18px}
 .table-head{display:block}.table-head span{display:block;margin-top:4px}
 .evaluation summary{padding:18px}.evaluation summary small{display:block;margin:5px 0 0}
 .eval-body{padding:0 18px 18px}th,td{padding:10px 12px}footer{display:block}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:#fff}.topbar nav,.controls,.skip{display:none}
 .wrap{width:100%}.card,.prefix-card,.evaluation{break-inside:avoid}
 .section{padding:18px 0}.hero{padding:20px 0}.table-scroll{overflow:visible}}
/* Extend the original mint/navy report with a clear prediction inspection surface. */
button{font:inherit;min-height:36px;cursor:pointer;border:1px solid #adc5b9;
 border-radius:5px;background:#fff;color:var(--ink);padding:7px 12px;font-size:12px}
button:disabled,select:disabled{cursor:default;opacity:.6}
.hero{position:relative}.hero-links{display:flex;align-items:center;gap:25px;
 flex-wrap:wrap;margin-top:27px;font-size:13px}.source-button{display:inline-flex;
 align-items:center;min-height:44px;padding:10px 16px;color:#fff;background:var(--accent);
 border-radius:5px;text-decoration:none}.source-button:hover{background:#064c43}
.controls{align-items:center;flex-wrap:wrap;background:#fff;border-top:3px solid var(--accent)}
.controls>div{flex:0 1 auto}.controls select{min-height:43px}.controls button{margin-top:23px}
.selection-exports{display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin:0 0 24px;
 font-size:12px;color:var(--muted)}.selection-exports>span{margin-right:7px}
.selection-exports p{flex-basis:100%;margin:2px 0 0;min-height:20px;font-size:12px}
.selection-error{padding:14px 17px;border:1px solid #bc9b68;background:#fff4df;
 color:#735126;border-radius:5px;font-size:13px}
.events{gap:0;flex-wrap:nowrap;overflow-x:auto;padding:5px 3px 12px;scrollbar-width:thin;
 scrollbar-color:#7da894 #edf5f1}.events li{padding:0;border:0;background:none;
 flex:0 0 175px;position:relative;margin-right:17px}
.events li:not(:last-child):after{content:'';position:absolute;right:-17px;top:28px;
 width:17px;border-top:1px solid #91b4a1}
.events button{width:100%;height:100%;text-align:left;padding:12px 14px;
 border-color:#a9c8b7;background:#fff;min-height:107px}
.events button:hover{background:#f6fbf6}.events button[aria-pressed=true]{background:#d9eee1;
 border-color:var(--accent);box-shadow:inset 3px 0 var(--accent)}
.event-number{display:block;font:12px ui-monospace,monospace;color:var(--accent);margin-bottom:7px}
.events strong{font-size:14px}.events small,.event-time{display:block;font-size:12px;
 color:var(--muted);line-height:1.7}.event-time{margin-top:2px}.prefix-card .event-status{
 margin:10px 0 0;font-size:12px}.ranking .is-observed-match{background:#e5f2e8}
.ranking .is-observed-match .item-id{box-shadow:inset 3px 0 var(--accent)}
.score-detail{margin-top:10px}.score-detail summary{font-size:12px;color:var(--accent);
 min-height:28px;cursor:pointer;padding:3px 0}.score-detail .table-scroll{margin-top:8px;
 border-radius:4px}.score-detail table{min-width:320px}.score-detail th,.score-detail td{
 font-size:12px;padding:8px 10px}.score-detail .features{margin:9px 0 0;line-height:1.7}
.coverage-comparison{background:#fffaf1;border:1px solid #ddcdb3;border-radius:5px;
 padding:16px 19px;margin:16px 0}.coverage-comparison>div{display:grid;
 grid-template-columns:minmax(180px,1fr) 72px 2fr;align-items:center;gap:16px;
 margin:8px 0;font-size:12px}.coverage-comparison strong{text-align:right;
 font-variant-numeric:tabular-nums}.coverage-track{display:block;height:9px;
 background:#e7e3d7;border-radius:2px;overflow:hidden}.coverage-track i{
 display:block;height:100%;background:#637e89}.coverage-track.ranked i{background:var(--accent)}
.coverage-comparison p{font-size:12px;margin:14px 0 0;line-height:1.7}
.evaluation summary{min-height:48px}.table-scroll:focus-visible{outline:3px solid #1d806b}
@media(max-width:760px){.hero-links{gap:17px}.controls>div{flex:1 1 160px}
 .controls button{margin:0}.selection-exports>span{width:100%;margin-bottom:4px}
 .coverage-comparison{padding:14px}.coverage-comparison>div{
 grid-template-columns:minmax(0,1fr) 66px;gap:8px;margin:13px 0}
 .coverage-track{grid-column:1/-1}.events li{flex-basis:165px}
 .ranking{min-width:590px}.score-detail table{min-width:300px}}

"""
_SCRIPT = r"""
(() => {
  'use strict';
  const byId = (id) => document.getElementById(id);
  const session = byId('session-select');
  const objective = byId('objective-select');
  const status = byId('selection-status');
  if (!session || !objective || !status) return;
  const panels = [...document.querySelectorAll('[data-demo-panel]')];
  const exportButtons = [byId('export-selection-json'), byId('export-selection-csv')];
  const reset = byId('reset-selection');
  let payload = null,
    selected = null,
    activePanel = null;
  const fail = (message) => {
    selected = null;
    activePanel = null;
    panels.forEach((panel) => {
      panel.hidden = true;
    });
    exportButtons.forEach((button) => {
      button.disabled = true;
    });
    byId('selection-error').textContent = message;
    byId('selection-error').hidden = false;
    byId('export-status').textContent = '';
    status.textContent = 'No current selection is available for export.';
  };
  const integer = (value) => Number.isSafeInteger(value) && value >= 0;
  const uniqueItems = (values) =>
    Array.isArray(values) &&
    values.every(integer) &&
    new Set(values).size === values.length;
  const actions = ['clicks', 'carts', 'orders'];
  const validate = (value) => {
    if (
      !value ||
      value.schema_version !== 1 ||
      value.evidence_type !== 'SYNTHETIC_ONLY' ||
      !/^[a-f0-9]{64}$/.test(value.predictions_sha256) ||
      !Array.isArray(value.examples) ||
      !value.examples.length
    )
      throw new Error('The embedded prediction data is unavailable or invalid.');
    const seen = new Set();
    for (const example of value.examples) {
      if (
        !example ||
        !integer(example.session) ||
        seen.has(example.session) ||
        !integer(example.query_ts) ||
        !Array.isArray(example.prefix)
      )
        throw new Error('Invalid synthetic session identity.');
      seen.add(example.session);
      for (const event of example.prefix)
        if (
          !integer(event.item) ||
          !integer(event.ts) ||
          event.ts >= example.query_ts ||
          !actions.includes(event.action)
        )
          throw new Error('Invalid observed event.');
      for (const action of actions) {
        const prediction = example.objectives?.[action];
        if (
          !prediction ||
          !uniqueItems(prediction.recommendations) ||
          !uniqueItems(prediction.candidates) ||
          prediction.recommendations.length > 20 ||
          !prediction.recommendations.every((item) =>
            prediction.candidates.includes(item),
          ) ||
          !Array.isArray(prediction.explanations) ||
          prediction.explanations.length !== prediction.recommendations.length
        )
          throw new Error('Invalid ranked prediction.');
        prediction.explanations.forEach((row, index) => {
          if (
            row.item !== prediction.recommendations[index] ||
            !Number.isFinite(row.score) ||
            !row.features ||
            typeof row.features !== 'object' ||
            Array.isArray(row.features) ||
            !Object.values(row.features).every(Number.isFinite) ||
            !Array.isArray(row.sources) ||
            !row.sources.every((source) => typeof source === 'string')
          )
            throw new Error('Invalid prediction evidence.');
        });
      }
    }
    return value;
  };
  const update = () => {
    selected = null;
    activePanel = null;
    panels.forEach((panel) => {
      panel.hidden = true;
      panel.querySelectorAll('details.evaluation').forEach((details) => {
        details.open = false;
      });
      panel
        .querySelectorAll('[data-event-item]')
        .forEach((button) => button.setAttribute('aria-pressed', 'false'));
      panel
        .querySelectorAll('[data-ranked-item]')
        .forEach((row) => row.classList.remove('is-observed-match'));
      const note = panel.querySelector('[data-event-status]');
      if (note)
        note.textContent =
          'Select an observed event to locate that item in this shortlist. ' +
          'The ranking stays unchanged.';
    });
    try {
      if (!payload || !/^\d+$/.test(session.value) || !actions.includes(objective.value))
        throw new Error('Choose a valid session and objective.');
      const example = payload.examples[Number(session.value)];
      const matches = panels.filter(
        (panel) =>
          panel.dataset.session === session.value &&
          panel.dataset.objective === objective.value,
      );
      if (!example || matches.length !== 1)
        throw new Error(
          'This session and objective are not available. Reset the explorer.',
        );
      const prediction = example.objectives[objective.value];
      selected = {
        schema: 'otto-synthetic-selected-predictions-v1',
        evidence_type: 'SYNTHETIC_ONLY',
        session: example.session,
        query_ts: example.query_ts,
        objective: objective.value,
        prefix: example.prefix,
        candidates: prediction.candidates,
        recommendations: prediction.recommendations,
        explanations: prediction.explanations,
        predictions_sha256: payload.predictions_sha256,
        scope:
          'Recorded synthetic prediction; no future targets or evaluation labels. ' +
          'Selecting an event does not recalculate the ranking.',
      };
      activePanel = matches[0];
      activePanel.hidden = false;
      byId('selection-error').hidden = true;
      byId('export-status').textContent = '';
      status.textContent = `Session ${example.session} / ${objective.value} · recorded prediction`;
      exportButtons.forEach((button) => {
        button.disabled = false;
      });
    } catch (error) {
      fail(error.message);
    }
  };
  const download = (format) => {
    if (!selected || !activePanel || exportButtons.some((button) => button.disabled))
      return;
    try {
      const fields = ['cooccurrence', 'prefix_recency', 'popularity'];
      const content =
        format === 'json'
          ? JSON.stringify(selected, null, 2) + '\n'
          : 'evidence_type,session,objective,rank,item,score,' +
            'cooccurrence,prefix_recency,popularity\n' +
            selected.explanations
              .map((row, index) =>
                [
                  'SYNTHETIC_ONLY',
                  selected.session,
                  selected.objective,
                  index + 1,
                  row.item,
                  row.score,
                  ...fields.map((field) => row.features[field] ?? ''),
                ].join(','),
              )
              .join('\n') +
            '\n';
      const blob = new Blob([content], {
        type: format === 'json' ? 'application/json' : 'text/csv;charset=utf-8',
      });
      const url = URL.createObjectURL(blob),
        link = document.createElement('a');
      link.href = url;
      link.download = `otto-synthetic-session-${selected.session}-` +
        `${selected.objective}-predictions.${format}`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      byId('export-status').textContent =
        `${format.toUpperCase()} prepared for this session and objective. ` +
        'Future targets are excluded.';
    } catch (error) {
      byId('export-status').textContent = `Download unavailable: ${error.message}`;
    }
  };
  try {
    payload = validate(JSON.parse(byId('demo-prediction-data')?.textContent || 'null'));
    session.addEventListener('change', update);
    objective.addEventListener('change', update);
    reset.addEventListener('click', () => {
      session.value = '0';
      objective.value = 'clicks';
      update();
    });
    exportButtons[0].addEventListener('click', () => download('json'));
    exportButtons[1].addEventListener('click', () => download('csv'));
    panels.forEach((panel) =>
      panel.querySelectorAll('[data-event-item]').forEach((button) =>
        button.addEventListener('click', () => {
          if (!selected || activePanel !== panel) return;
          const chosen = button.getAttribute('aria-pressed') !== 'true';
          panel
            .querySelectorAll('[data-event-item]')
            .forEach((event) => event.setAttribute('aria-pressed', 'false'));
          button.setAttribute('aria-pressed', String(chosen));
          let hits = 0;
          panel.querySelectorAll('[data-ranked-item]').forEach((row) => {
            const match = chosen && row.dataset.rankedItem === button.dataset.eventItem;
            row.classList.toggle('is-observed-match', match);
            if (match) hits++;
          });
          panel.querySelector('[data-event-status]').textContent = !chosen
            ? 'Event highlight cleared. The ranking is unchanged.'
            : `Observed item ${button.dataset.eventItem}: ` +
              `${hits ? 'present in the ranked shortlist' : 'not in the ranked shortlist'}. ` +
              'Highlight only; the ranking is unchanged.';
        }),
      ),
    );
    session.disabled = false;
    objective.disabled = false;
    reset.disabled = false;
    document.querySelectorAll('[data-event-item]').forEach((button) => {
      button.disabled = false;
    });
    update();
  } catch (error) {
    session.disabled = true;
    objective.disabled = true;
    reset.disabled = true;
    fail(
      `The interactive explorer could not start. ${error.message} ` +
        'The aggregate report remains a recorded synthetic result.',
    );
  }
})();
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Report values must be finite numbers")
    return float(value)


def _integer(value: Any) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("Report counts and identifiers must be nonnegative integers")
    return f"{value:,}"


def _ratio(value: Any) -> str:
    number = _number(value)
    if not 0 <= number <= 1:
        raise ValueError("Report recall values must lie in [0, 1]")
    return f"{number:.4f}"


def _stat(label: str, value: str, note: str, *, primary: bool = False) -> str:
    classes = "card stat primary" if primary else "card stat"
    return (
        f'<div class="{classes}"><div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div><p>{_escape(note)}</p></div>'
    )


def _metric_table(metrics: Mapping[str, Any]) -> str:
    rows = []
    for action in _ACTIONS:
        row = metrics["objectives"][action]
        weight = f"{100 * _number(row['weight']):.0f}%"
        rows.append(
            f"<tr><td><strong>{_LABELS[action]}</strong></td><td>{weight}</td>"
            f'<td class="number">{_integer(row["hits"])} / '
            f"{_integer(row['denominator'])}</td>"
            f'<td class="number">{_ratio(row["recall_at_20"])}</td>'
            f'<td class="number">{_integer(row["candidate_hits"])} / '
            f"{_integer(row['denominator'])}</td>"
            f'<td class="number">{_ratio(row["candidate_oracle"])}</td></tr>'
        )
    return (
        '<div class="table-scroll"><table><caption>Synthetic evaluation, pooled across '
        'all generated query sessions</caption><thead><tr><th scope="col">Objective</th>'
        '<th scope="col">Weight</th><th scope="col" class="number">Top-20 hits / targets</th>'
        '<th scope="col" class="number">Recall@20</th>'
        '<th scope="col" class="number">Pool hits / targets</th>'
        '<th scope="col" class="number">Candidate oracle</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _pipeline(stages: list[dict[str, Any]]) -> str:
    descriptions = {
        "generate": ("Generate", "Deterministic synthetic history and query sessions."),
        "fit": ("Build retrieval", "Fit source statistics using historical sessions."),
        "predict": ("Rank candidates", "Use observed prefixes; keep future targets separate."),
        "seal": ("Seal predictions", "Hash the completed recommendation artifact."),
        "evaluate": ("Evaluate", "Compare sealed lists with synthetic future targets."),
    }
    items = []
    for index, stage in enumerate(stages, 1):
        name = str(stage["name"])
        label, description = descriptions.get(name, (name, "Recorded pipeline stage."))
        items.append(
            f'<li><span class="step-number">{index:02d}</span>'
            f"<strong>{_escape(label)}</strong><small>{_escape(description)}</small>"
            f'<span class="state">{_escape(stage["status"])}</span></li>'
        )
    return '<ol class="pipeline" aria-label="Executed pipeline stages">' + "".join(items) + "</ol>"


def _prefix(events: list[dict[str, Any]]) -> str:
    items = []
    first_ts = events[0]["ts"] if events else 0
    for index, event in enumerate(events, 1):
        elapsed = event["ts"] - first_ts
        items.append(
            f'<li><button type="button" data-event-item="{event["item"]}" '
            f'data-event-order="{index}" aria-pressed="false" disabled '
            f'aria-label="Observed event {index}: item {_integer(event["item"])}, '
            f'{_escape(event["action"])}. Highlight in the shortlist.">'
            f'<span class="event-number">{index:02d}</span>'
            f"<strong>Item {_integer(event['item'])}</strong>"
            f"<small>{_escape(event['action'])} · +{_integer(elapsed)} time units</small>"
            f'<span class="event-time">Recorded t = {_integer(event["ts"])}</span>'
            "</button></li>"
        )
    return (
        '<div class="prefix-card"><div class="eyebrow">Available at prediction time</div>'
        "<h3>The observed session, event by event</h3><p>These events precede the query. "
        "Only this prefix and historical source statistics inform the recorded prediction. "
        "Repeated items remain eligible for recommendation.</p>"
        '<ol class="events" aria-label="Observed events, in recorded order">'
        + "".join(items)
        + '</ol><p class="event-status" data-event-status role="status">'
        "Enable JavaScript to highlight an observed item in the shortlist.</p></div>"
    )


def _explanation(value: Mapping[str, Any]) -> str:
    sources = value.get("sources", [])
    tags = "".join(f'<span class="tag">{_escape(source)}</span>' for source in sources)
    reason = f"<div>{_escape(value['reason'])}</div>" if value.get("reason") else ""
    features = value.get("features", {})
    weights = {"cooccurrence": 0.6, "prefix_recency": 0.3, "popularity": 0.1}
    weighted = all(name in features for name in weights) and "score" in value
    if weighted:
        reconstructed = sum(weight * _number(features[name]) for name, weight in weights.items())
        weighted = math.isclose(reconstructed, _number(value["score"]), abs_tol=1e-12)
    rows = []
    for name, number in sorted(features.items()):
        label = str(name).replace("_", " ")
        detail = f'<td class="number">{_number(number):.5f}</td>'
        if weighted:
            weight = weights.get(name, 0.0)
            detail += f'<td class="number">{weight:.1f}</td>'
            detail += f'<td class="number">{weight * _number(number):.5f}</td>'
        rows.append(f"<tr><td>{_escape(label)}</td>{detail}</tr>")
    feature_html = ""
    if rows:
        heading = '<th scope="col">Input feature</th><th scope="col">Value</th>'
        if weighted:
            heading += '<th scope="col">Weight</th><th scope="col">Contribution</th>'
        note = (
            "Weighted contributions sum to the recorded demonstration score before rounding. "
            "These fixed weights belong to this public example, not the private recommender."
            if weighted
            else "Recorded feature values; no unsupported score decomposition is implied."
        )
        feature_html = (
            '<details class="score-detail"><summary>Inspect score components</summary>'
            '<div class="table-scroll"><table><thead><tr>' + heading + "</tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
            f'<p class="features">{note}</p></details>'
        )
    if not tags and not reason and not feature_html:
        return '<span class="features">Source-only demonstration scoring rule.</span>'
    return f'<div class="tags">{tags}</div>{reason}{feature_html}'


def _ranking(example: Mapping[str, Any], action: str) -> str:
    recommendations = example["recommendations"][action]
    evidence = {row["item"]: row for row in example.get("explanations", {}).get(action, [])}
    rows = []
    for rank, item in enumerate(recommendations, 1):
        explanation = evidence.get(item, {})
        score = f"{_number(explanation['score']):.5f}" if "score" in explanation else "—"
        rows.append(
            f'<tr data-ranked-item="{item}"><td class="rank">{rank:02d}</td>'
            f'<td class="item-id">Item {_integer(item)}</td><td class="score">{score}</td>'
            f'<td class="explanation">{_explanation(explanation)}</td></tr>'
        )
    candidate_count = len(example["candidates"][action])
    return (
        '<div class="table-head"><h3>Ranked recommendations</h3>'
        f"<span>{len(recommendations)} shown from {candidate_count} "
        "retrieved candidates</span></div>"
        '<div class="table-scroll"><table class="ranking">'
        "<caption>Scores and explanations use prediction-time inputs. "
        "A demonstration score is not a calibrated probability.</caption><thead><tr>"
        '<th scope="col">Rank</th><th scope="col">Item</th>'
        '<th scope="col">Demo score</th><th scope="col">Source evidence</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _evaluation(example: Mapping[str, Any], action: str) -> str:
    targets = list(dict.fromkeys(example["targets"][action]))
    pool = set(example["candidates"][action])
    ranked = {item: rank for rank, item in enumerate(example["recommendations"][action], 1)}
    hits = sum(item in ranked for item in targets)
    pool_hits = min(20, sum(item in pool for item in targets))
    denominator = min(20, len(targets))
    rows = []
    achieved = hits / denominator if denominator else 0.0
    ceiling = pool_hits / denominator if denominator else 0.0
    coverage = (
        '<div class="coverage-comparison">'
        "<div><span>Retrieved target coverage</span>"
        f"<strong>{pool_hits} / {denominator}</strong>"
        f'<span class="coverage-track"><i style="width:{ceiling * 100:.8f}%"></i></span></div>'
        "<div><span>Achieved top-20 hits</span>"
        f"<strong>{hits} / {denominator}</strong>"
        '<span class="coverage-track ranked">'
        f'<i style="width:{achieved * 100:.8f}%"></i></span></div>'
        f"<p>Retrieval shortfall: {denominator - pool_hits}. "
        f"Additional ranking shortfall: {pool_hits - hits}. "
        "Counts use the same capped target denominator; this is evaluation, not input evidence.</p>"
        "</div>"
    )
    for item in targets:
        result = f"Rank {ranked[item]}" if item in ranked else "Outside top 20"
        rows.append(
            f"<tr><td>Item {_integer(item)}</td>"
            f"<td>{'Yes' if item in pool else 'No'}</td><td>{result}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="3">No future target for this objective.</td></tr>')
    return (
        '<details class="evaluation"><summary>Open future-target evaluation'
        '<small>Withheld from prediction inputs</small></summary><div class="eval-body">'
        "<p>These synthetic future items are used only after predictions have been sealed. "
        "They explain retrieval misses separately from ranking misses.</p>"
        + coverage
        + '<div class="evaluation-counts">'
        f"<span>Top-20 hits: <strong>{hits}</strong></span>"
        f"<span>Candidate-pool hits: <strong>{pool_hits}</strong></span>"
        f"<span>Capped target denominator: <strong>{denominator}</strong></span></div>"
        '<div class="table-scroll"><table><thead><tr><th scope="col">Future target</th>'
        '<th scope="col">Retrieved?</th><th scope="col">Ranked result</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></div></details>"
    )


def _panel(example: Mapping[str, Any], index: int, action: str) -> str:
    hidden = "" if index == 0 and action == "clicks" else " hidden"
    return (
        f'<article data-demo-panel data-session="{index}" data-objective="{action}"{hidden}>'
        '<div class="session-header"><div><div class="eyebrow">Synthetic example</div>'
        f"<h3>Session {_integer(example['session'])} · {_LABELS[action]}</h3></div>"
        f'<div class="session-meta">Query time: {_integer(example["query_ts"])}<br>'
        f"{len(example['prefix'])} observed events</div></div>"
        + _prefix(example["prefix"])
        + _ranking(example, action)
        + _evaluation(example, action)
        + "</article>"
    )


def _provenance(payload: Mapping[str, Any]) -> str:
    provenance = payload["provenance"]
    hashes = []
    for key, label in (
        ("history_sha256", "Synthetic history SHA-256"),
        ("queries_sha256", "Query artifact SHA-256"),
        ("predictions_sha256", "Sealed predictions SHA-256"),
    ):
        hashes.append(f'<dt>{label}</dt><dd class="hash">{_escape(provenance[key])}</dd>')
    sealed = "Yes" if provenance["predictions_sealed_before_evaluation"] else "No"
    targets = "Yes" if provenance["targets_used_in_fit_or_inference"] else "No"
    return (
        '<div class="columns"><div class="card provenance"><h3>Reproducible artifacts</h3>'
        "<p>Hashes identify this synthetic run's generated inputs and predictions. "
        "They are separate from the published research evidence.</p><dl>"
        + "".join(hashes)
        + '</dl></div><div class="card provenance">'
        "<h3>Information boundaries</h3><dl>"
        f"<dt>Predictions sealed before evaluation</dt><dd>{sealed}</dd>"
        f"<dt>Future targets used for fitting or inference</dt><dd>{targets}</dd>"
        f"<dt>Deterministic tie break</dt><dd>{_escape(provenance['tie_break'])}</dd>"
        f"<dt>Generator seed</dt><dd>{_integer(payload['seed'])}</dd>"
        "<dt>Execution setting</dt><dd>Local synthetic demonstration; no remote services.</dd>"
        "</dl></div></div>"
    )


def _prediction_data(payload: Mapping[str, Any]) -> str:
    examples = []
    for example in payload["examples"]:
        examples.append(
            {
                "session": example["session"],
                "query_ts": example["query_ts"],
                "prefix": [
                    {key: event[key] for key in ("item", "action", "ts")}
                    for event in example["prefix"]
                ],
                "objectives": {
                    action: {
                        "recommendations": example["recommendations"][action],
                        "candidates": example["candidates"][action],
                        "explanations": example["explanations"][action],
                    }
                    for action in _ACTIONS
                },
            }
        )
    data = {
        "schema_version": 1,
        "evidence_type": "SYNTHETIC_ONLY",
        "predictions_sha256": payload["provenance"]["predictions_sha256"],
        "examples": examples,
    }
    # JSON script content cannot terminate its own element or become markup.
    encoded = json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"))
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return '<script id="demo-prediction-data" type="application/json">' + encoded + "</script>"


def render_report(payload: Mapping[str, Any]) -> str:
    """Return accessible offline HTML for a version-1 synthetic demo payload.

    Dynamic values are escaped as text. Constant JavaScript switches pre-rendered
    panels and exports the separately embedded prediction-only snapshot.
    """
    if payload.get("synthetic") is not True or payload.get("schema_version") != 1:
        raise ValueError("The demo report accepts only schema-version-1 synthetic results")
    # This also rejects nonfinite metrics before any partial report is returned.
    json.dumps(dict(payload), allow_nan=False)
    configuration = payload["configuration"]
    metrics = payload["metrics"]
    examples = payload["examples"]
    stats = "".join(
        (
            _stat(
                "Synthetic weighted Recall@20",
                _ratio(metrics["weighted_recall_at_20"]),
                "Achieved by the generated top-20 lists.",
                primary=True,
            ),
            _stat(
                "Synthetic candidate oracle",
                _ratio(metrics["weighted_candidate_oracle"]),
                "Coverage ceiling of the retrieved pools; not achieved ranking quality.",
            ),
            _stat(
                "Generated query sessions",
                _integer(configuration["query_sessions"]),
                f"{_integer(configuration['history_sessions'])} historical sessions · "
                f"{_integer(configuration['catalog_items'])} synthetic item IDs",
            ),
        )
    )
    options = "".join(
        f'<option value="{index}">Session {_integer(example["session"])}</option>'
        for index, example in enumerate(examples)
    )
    panels = "".join(
        _panel(example, index, action)
        for index, example in enumerate(examples)
        for action in _ACTIONS
    )
    explorer = (
        (
            '<div class="controls"><div><label for="session-select">Example session</label>'
            f'<select id="session-select" disabled>{options}</select></div><div>'
            '<label for="objective-select">Recommendation objective</label>'
            '<select id="objective-select" disabled><option value="clicks">Clicks</option>'
            '<option value="carts">Carts</option><option value="orders">Orders</option></select>'
            '</div><button id="reset-selection" type="button" disabled>Reset selection</button>'
            '<p id="selection-status" class="live-note" role="status" aria-live="polite">'
            "The first recorded example is shown. Interactive controls load locally.</p>"
            '</div><p id="selection-error" class="selection-error" role="alert" hidden></p>'
            '<div class="selection-exports"><span>Take this prediction with you</span>'
            '<button id="export-selection-json" type="button" disabled>Export JSON ↓</button>'
            '<button id="export-selection-csv" type="button" disabled>Export CSV ↓</button>'
            '<p id="export-status" role="status"></p></div>'
            "<noscript><p>JavaScript is disabled. The first click example remains visible; "
            "the recorded report remains readable, and local exports require "
            "JavaScript.</p></noscript>" + panels
        )
        if examples
        else '<p class="empty">No query examples were included in this run.</p>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="description" content="An offline synthetic walkthrough of session '
        'retrieval, ranking, and pooled recommendation evaluation.">'
        "<title>OTTO Session Lab · From events to recommendations</title>"
        f'<style>{_CSS}</style></head><body><a class="skip" href="#main">Skip to report</a>'
        '<div class="wrap"><header class="topbar"><div class="brand">OTTO / DEMO</div>'
        '<nav aria-label="Report sections"><a href="#overview">Overview</a>'
        '<a href="#explore">Explore a session</a><a href="#provenance">Provenance</a></nav>'
        '</header><main id="main"><section class="hero">'
        '<div class="eyebrow badge">Synthetic · Local · Reproducible</div>'
        "<h1>From an observed session<br>to a ranked shortlist.</h1>"
        '<p class="lead">Inspect the path from historical source statistics to candidate '
        "retrieval, objective-specific ranking, and a sealed top-20 prediction.</p>"
        f'<p class="scope">{_escape(payload["scope"])} All item IDs, sessions and results '
        "on this page are synthetic. This walkthrough is separate from the project's "
        "published temporal study and scored release.</p>"
        '<div class="hero-links"><a class="source-button" '
        'href="https://github.com/alvaromendizabal/otto-recommender-system" '
        'target="_blank" rel="noopener noreferrer">Explore the source ↗</a>'
        '<a href="https://github.com/alvaromendizabal/otto-recommender-system/'
        'blob/main/docs/REVIEWER_GUIDE.md" '
        'target="_blank" rel="noopener noreferrer">The five-minute review ↗</a></div></section>'
        '<section class="section" id="overview" aria-labelledby="overview-title">'
        '<div class="section-head"><div class="eyebrow">01 / Run overview</div></div>'
        '<h2 id="overview-title">One pipeline. Two distinct questions.</h2>'
        '<p class="scope">Did retrieval make the future item available? '
        "Did ranking place it in the final twenty?</p>"
        f'<div class="stat-grid">{stats}</div></section>'
        '<section class="section" aria-labelledby="pipeline-title">'
        '<div class="section-head"><h2 id="pipeline-title">Follow the data boundary.</h2>'
        "<p>Historical evidence and observed prefixes produce predictions. "
        "Future targets are reserved for the final evaluation stage.</p></div>"
        + _pipeline(payload["stages"])
        + "</section>"
        '<section class="section" aria-labelledby="metrics-title">'
        '<div class="section-head"><h2 id="metrics-title">Measure the pooled result.</h2></div>'
        + _metric_table(metrics)
        + '<p class="method-note">For each objective, Recall@20 is total top-20 target hits '
        "divided by the sum of min(20, unique future targets) across sessions. "
        "Weighted recall uses 10% clicks, 30% carts and 60% orders. Candidate oracle uses "
        "the same denominator and caps each session's available target hits at twenty. "
        "It is a retrieval ceiling, not the mean of per-session recalls or a leaderboard score.</p>"
        '</section><section class="section" id="explore" aria-labelledby="explore-title">'
        '<div class="section-head"><div><div class="eyebrow">02 / Inspect a prediction</div>'
        '<h2 id="explore-title">Open the shortlist.</h2></div>'
        f"<p>Browse {len(examples)} generated examples. The pooled metrics above use all "
        f"{_integer(configuration['query_sessions'])} query sessions. Future targets stay "
        "in a separate evaluation panel.</p></div>" + explorer + "</section>"
        '<section class="section" id="provenance" aria-labelledby="provenance-title">'
        '<div class="section-head"><div><div class="eyebrow">03 / Inspect the evidence</div>'
        '<h2 id="provenance-title">Trace this run.</h2></div>'
        "<p>The companion JSON and manifest preserve exact values "
        "and artifact identities.</p></div>" + _provenance(payload) + "</section></main>"
        "<footer><span>OTTO session recommendation · Public engineering walkthrough</span>"
        "<span>Self-contained HTML · No external assets or network requests</span></footer>"
        f"</div>{_prediction_data(payload)}<script>{_SCRIPT}</script></body></html>"
    )


def write_report(payload: Mapping[str, Any], output: Path) -> Path:
    """Write one UTF-8 HTML report; the caller owns artifact manifest creation."""
    rendered = render_report(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output
