"""Build the canonical analytical notebook only from measured ranking reports."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

OBJECTIVES = ("clicks", "carts", "orders")
WEIGHTS = (0.1, 0.3, 0.6)


def validate_report(report: dict[str, Any]) -> None:
    if (report.get("status") != "passed"
            or not re.fullmatch(r"[0-9a-f]{64}", report.get("run_id", ""))
            or report.get("untouched_temporal_holdout") is not False
            or not report.get("validation_scope")):
        raise ValueError("expected a completed, provenance-labeled ranking report")
    folds = report.get("folds", [])
    if not folds or len({fold["outer_fold"] for fold in folds}) != len(folds):
        raise ValueError("ranking report requires unique evaluated outer folds")
    for system in ("learned", "baseline"):
        recalls = []
        for objective in OBJECTIVES:
            hits = denominator = 0
            for fold in folds:
                metric = fold["objectives"][objective][system]
                h, d = metric["hits"], metric["denominator"]
                if (isinstance(h, bool) or isinstance(d, bool)
                        or not isinstance(h, int) or not isinstance(d, int) or not 0 <= h <= d):
                    raise ValueError("invalid ranking numerator or denominator")
                hits += h
                denominator += d
            if denominator <= 0:
                raise ValueError("every objective must have a positive evaluation denominator")
            summary = report[system]["objectives"][objective]
            if (summary["hits"] != hits or summary["denominator"] != denominator
                    or not math.isclose(summary["recall_at_20"], hits / denominator)):
                raise ValueError("reported objective score does not match its fold counts")
            recalls.append(hits / denominator)
        expected = sum(weight * recall for weight, recall in zip(WEIGHTS, recalls, strict=True))
        if not math.isclose(report[system]["weighted_recall_at_20"], expected):
            raise ValueError("weighted score does not match the official aggregation")


SOURCES = (
    ("markdown", """# 08 | Learned ranking evaluation

**Question:** Does the learned ranker improve ordered top-20 recommendations over
source agreement and reciprocal-rank fusion on the same candidate pool?

This notebook reads measured results; it never launches training. The evaluation
window is exploratory, not an untouched temporal test. Ranker early stopping uses
an inner session partition. Frozen upstream retriever provenance remains uncertified.
"""),
    ("code", """import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import PercentFormatter

root = Path.cwd() if (Path.cwd() / 'reports').is_dir() else Path.cwd().parent
path = root / 'reports/metrics/ranking_evaluation.json'
report = json.loads(path.read_text())
assert report['status'] == 'passed'
assert report['untouched_temporal_holdout'] is False
objectives = ['clicks', 'carts', 'orders']
print('Read at UTC:', datetime.now(UTC).isoformat())
print('Run:', report['run_id'])
print('Scope:', report['validation_scope'])
print('Candidate budget per objective/session:', report['candidate_k'])
"""),
    ("markdown", """## Official score

The objective weights are 10% clicks, 30% carts and 60% orders. Per-session true-item
denominators are capped at 20. Fold numerators and denominators are pooled before
applying those weights. Queries with no retrieved candidates still contribute their
full denominator. The compression budget is a baseline choice, not an optimized result.
"""),
    ("code", """rows = []
for objective in objectives:
    baseline = report['baseline']['objectives'][objective]
    learned = report['learned']['objectives'][objective]
    rows.append({'Objective': objective, 'Baseline Recall@20': baseline['recall_at_20'],
                 'LambdaRank Recall@20': learned['recall_at_20'],
                 'Change (percentage points)': 100 * (learned['recall_at_20']
                                                      - baseline['recall_at_20']),
                 'Hits': learned['hits'], 'Full denominator': learned['denominator']})
scores = pd.DataFrame(rows).set_index('Objective')
display(scores.round(5))
base = report['baseline']['weighted_recall_at_20']
learned = report['learned']['weighted_recall_at_20']
print(f'Official weighted Recall@20: baseline={base:.6f}; LambdaRank={learned:.6f}')
print(f'Absolute change: {100 * (learned - base):+.3f} percentage points')
"""),
    ("code", """positions = np.arange(len(objectives))
fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
ax.barh(positions - 0.18, scores['Baseline Recall@20'], height=0.34, label='Matched baseline')
ax.barh(positions + 0.18, scores['LambdaRank Recall@20'], height=0.34, label='LambdaRank')
ax.set(yticks=positions, yticklabels=objectives, xlim=(0, 1),
       xlabel='Recall@20 | full held-out denominator',
       title='Ordered recommendations on the same candidate pool')
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.18)
ax.set_axisbelow(True)
ax.legend(loc='lower right')
plt.show()
"""),
    ("markdown", """## Selection, coverage and ranking diagnostics

Candidate coverage is an ideal top-20 ceiling, **not** the achieved ranking score.
NDCG and MRR are averaged over labeled queries; they complement, rather than replace,
the official metric. Fit time and outer evaluation time are different measurements:
evaluation includes data loading, prediction, sorting and metric computation.
"""),
    ("code", """diagnostics = []
for fold in report['folds']:
    for objective in objectives:
        result = fold['objectives'][objective]
        metric = result['learned']
        diagnostics.append({'Fold': fold['outer_fold'], 'Objective': objective,
            'Candidate ceiling': metric['candidate_ceiling_at_20'],
            'Recall@20': metric['recall_at_20'], 'NDCG@20': metric['ndcg_at_20'],
            'MRR@20': metric['mrr_at_20'], 'Best iteration': result['best_iteration'],
            'Inner Recall@20': result['inner_recall_at_20'],
            'Retained fit seconds': result['retained_fit_seconds'],
            'Outer evaluation seconds': result['evaluation_seconds']})
diagnostics = pd.DataFrame(diagnostics)
display(diagnostics.round(4))
timing = diagnostics.groupby('Objective')[
    ['Retained fit seconds', 'Outer evaluation seconds']].sum().reindex(objectives)
fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
ax.barh(positions - 0.18, timing['Retained fit seconds'], height=0.34, label='Model fitting')
ax.barh(positions + 0.18, timing['Outer evaluation seconds'], height=0.34,
        label='Outer evaluation, including I/O')
ax.set(yticks=positions, yticklabels=objectives, xlabel='Seconds',
       title='Compute cost by objective | saved work retained on resume')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.18)
ax.set_axisbelow(True)
ax.legend()
plt.show()
"""),
    ("markdown", """## Interpretation limits

A positive point-estimate change is not proof of statistical significance or a
leaderboard result. Paired session confidence intervals, candidate-budget and feature
ablations, certified neural-source fits, and full-test prediction remain separate
experiments. No Kaggle submission is implied by this notebook. Model files, feature
order, fit contracts and iteration checkpoints live in the durable run namespace.
"""),
)


def write_ranking_notebook(report: dict[str, Any], path: Path) -> None:
    validate_report(report)
    cells: list[dict[str, Any]] = []
    for kind, source in SOURCES:
        cell: dict[str, Any] = {
            "cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "metadata": {}, "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".ipynb.tmp")
    temporary.write_text(json.dumps(notebook, indent=1) + "\n")
    temporary.replace(path)
