"""Publication must never substitute unexecuted, changed or corrupt notebook bytes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/publish_notebooks.py'
spec = importlib.util.spec_from_file_location('publication', SCRIPT)
assert spec is not None and spec.loader is not None
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root, executed = tmp_path / 'project', tmp_path / 'executed'
    runtime = {'python': 'test-runtime'}
    monkeypatch.setattr(publication, 'runtime_versions', lambda: runtime)
    for path in ('reports/result.json', 'configs/run.json', 'notebooks/requirements.txt',
                 'scripts/execute_notebooks.py'):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}')
    receipts = []
    for number in (1, 2):
        name = f'{number:02d}_analysis.ipynb'
        notebook = {'cells': [{'cell_type': 'code', 'source': ['print(1)\n'],
                               'execution_count': None, 'outputs': []}]}
        write(root / 'notebooks' / name, notebook)
        identity = publication.identity(notebook, publication.dependencies(root), runtime, 300)
        notebook['cells'][0].update(execution_count=1, outputs=[
            {'output_type': 'stream', 'name': 'stdout', 'text': ['1\n']}
        ])
        write(executed / name, notebook)
        receipt = {'notebook': name, 'runtime': runtime, 'input_id': identity,
                   'sha256': publication.sha256(executed / name), 'code_cells': 1,
                   'compute_seconds': 1.0}
        write((executed / name).with_suffix('.json'), receipt)
        receipts.append(receipt)
    write(executed / 'manifest.json', {'status': 'passed', 'notebooks': 2, 'reused_notebooks': 2,
          'code_cells': 2, 'retained_compute_seconds': 2.0, 'receipts': receipts})
    return root, executed


def snapshot(root):
    return {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def rehash(executed, name):
    path = executed / name
    receipt = publication.load(path.with_suffix('.json'))
    receipt['sha256'] = publication.sha256(path)
    write(path.with_suffix('.json'), receipt)
    manifest = publication.load(executed / 'manifest.json')
    manifest['receipts'] = [receipt if r['notebook'] == name else r for r in manifest['receipts']]
    write(executed / 'manifest.json', manifest)


def test_publishes_outputs_and_durable_receipts_without_changing_inputs(workspace):
    root, executed = workspace
    inputs = publication.dependencies(root)
    result = publication.publish(root, executed)
    assert result['notebooks'] == result['code_cells'] == 2
    assert result['status'] == 'passed'
    assert publication.dependencies(root) == inputs
    for receipt in result['receipts']:
        assert publication.sha256(root / 'notebooks' / receipt['notebook']) == receipt['sha256']
    assert (root / 'notebooks/execution.json').is_file()
    publication.publish(root, executed)
    assert publication.dependencies(root) == inputs


@pytest.mark.parametrize('change', ['raw_bytes', 'source', 'evidence', 'runtime', 'receipt'])
def test_mismatch_fails_before_any_canonical_file_is_replaced(workspace, monkeypatch, change):
    root, executed = workspace
    path = executed / '02_analysis.ipynb'
    if change == 'raw_bytes':
        path.write_text('{}')
    elif change == 'source':
        value = publication.load(path)
        value['cells'][0]['source'] = ['print(2)']
        write(path, value)
        rehash(executed, path.name)
    elif change == 'evidence':
        (root / 'reports/result.json').write_text('{"changed": true}')
    elif change == 'runtime':
        monkeypatch.setattr(publication, 'runtime_versions', lambda: {'python': 'other'})
    else:
        path.with_suffix('.json').unlink()
    before = snapshot(root)
    with pytest.raises((ValueError, FileNotFoundError)):
        publication.publish(root, executed)
    assert snapshot(root) == before


@pytest.mark.parametrize('bad_output', ['error', 'warning', 'unexecuted'])
def test_bad_execution_cannot_be_published_even_with_updated_hashes(workspace, bad_output):
    root, executed = workspace
    path = executed / '02_analysis.ipynb'
    value = publication.load(path)
    cell = value['cells'][0]
    if bad_output == 'error':
        cell['outputs'] = [{'output_type': 'error', 'ename': 'ValueError'}]
    elif bad_output == 'warning':
        cell['outputs'] = [{'output_type': 'stream', 'text': 'RuntimeWarning: test'}]
    else:
        cell['execution_count'] = None
    write(path, value)
    rehash(executed, path.name)
    before = snapshot(root)
    with pytest.raises(ValueError):
        publication.publish(root, executed)
    assert snapshot(root) == before


@pytest.mark.parametrize('change', ['no_reuse', 'duplicate', 'unexpected', 'cell_total'])
def test_manifest_requires_exact_complete_reused_set(workspace, change):
    root, executed = workspace
    path = executed / 'manifest.json'
    value = publication.load(path)
    if change == 'no_reuse':
        value['reused_notebooks'] = 0
    elif change == 'duplicate':
        value['receipts'][1] = value['receipts'][0]
    elif change == 'unexpected':
        value['receipts'][1]['notebook'] = '../../untrusted.ipynb'
    else:
        value['code_cells'] = 99
    write(path, value)
    before = snapshot(root)
    with pytest.raises(ValueError):
        publication.publish(root, executed)
    assert snapshot(root) == before


def test_output_directory_cannot_overwrite_source_evidence(workspace):
    root, _ = workspace
    with pytest.raises(ValueError):
        publication.publish(root, root / 'notebooks')
