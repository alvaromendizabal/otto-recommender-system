"""Small user-run import integration check. No model fits, data scans or AWS API calls."""
import hashlib
import json
from pathlib import Path
import sys
from legacy_imports import load_replication

root = Path(__file__).resolve().parent
reference = json.loads((root/'evidence/ROUND02_REFERENCE.json').read_text())['round02_files']
legacy = Path.home()/'otto_feature_round02'
helper = load_replication(legacy, reference)
for name in ('paths_for', 'labels_for'):
    if not callable(getattr(helper, name, None)):
        raise RuntimeError('Required legacy interface missing: '+name)
for name, filename in (('core', 'core.py'), ('intent_features', 'intent_features.py'),
                       ('round02_certified_helpers', 'replication.py')):
    path = Path(sys.modules[name].__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference[filename]:
        raise RuntimeError('Imported source changed: '+name)
print('LEGACY_IMPORT_PREFLIGHT_PASSED', flush=True)
print(json.dumps({'model_fits':0,'data_scans':0,'modules_checked':3,'round':root.name}))
