"""Reconcile published real-data result bytes and scope without a new training fit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from otto_recsys.experiments.manifest import canonical_json_sha256
from otto_recsys.ranking.reporting import validate_report

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / 'reports/metrics'


def test_published_ranking_matches_source_checksums_and_arithmetic():
    path = METRICS / 'ranking_evaluation.json'
    report = json.loads(path.read_text())
    provenance = json.loads((METRICS / 'ranking_evaluation_provenance.json').read_text())
    validate_report(report)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['metrics_sha256']
    assert report['run_id'] == provenance['run_id']
    assert report['learned']['weighted_recall_at_20'] == pytest.approx(0.49731694683221855)
    for objective, result in report['folds'][0]['objectives'].items():
        receipt = provenance['evaluation_receipts'][objective]
        raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
        assert hashlib.sha256(raw).hexdigest() == receipt['files']['evaluation.json']
        assert result['model_sha256'] == receipt['files']['model.txt']
        assert receipt['input_id'] == canonical_json_sha256({
            'run_id': report['run_id'], 'fold': 0, 'objective': objective,
        })
        assert result['learned']['all_queries'] == result['baseline']['all_queries'] == 103468
        assert result['learned']['denominator'] == result['baseline']['denominator']
    assert report['untouched_temporal_holdout'] is False
    assert provenance['broad_feature_screening_performed'] is False
    assert provenance['feature_family_ablations_performed'] is False
    assert provenance['native_model_sha256_independently_verified'] is False
    assert provenance['model_feature_count'] == 30


def test_ranking_notebook_has_provenance_checks_and_explicit_scope():
    notebook = json.loads((ROOT / 'notebooks/08_ranking_evaluation.ipynb').read_text())
    text = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'metrics_sha256' in text and 'math.isclose' in text
    assert 'not an untouched temporal' in text
    assert 'candidate compression' in text
    assert 'training sessions' in text
    assert 'No Kaggle submission' in text
    assert sum(cell['cell_type'] == 'code' for cell in notebook['cells']) == 4


def test_published_notebook_receipts_match_committed_bytes_when_present():
    path = ROOT / 'notebooks/execution.json'
    # Sources are first tested on the results branch; the guarded publisher then commits renders.
    if not path.is_file():
        return
    record = json.loads(path.read_text())
    paths = sorted((ROOT / 'notebooks').glob('[0-9][0-9]_*.ipynb'))
    assert record['status'] == 'passed'
    assert record['notebooks'] == len(paths)
    assert {r['notebook'] for r in record['receipts']} == {p.name for p in paths}
    for receipt in record['receipts']:
        path = ROOT / 'notebooks' / receipt['notebook']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        notebook = json.loads(path.read_text())
        cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 and ''.join(cell['source']).strip()]
        assert len(cells) == receipt['code_cells']
        assert all(type(cell['execution_count']) is int for cell in cells)
        assert all(output['output_type'] != 'error' for cell in cells
                   for output in cell.get('outputs', []))
