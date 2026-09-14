"""Manual Round 01: observed-intent support and evidence timing.

Never downloads raw data, changes Git, builds candidate frontiers, or submits.
Screening reuses six previously fitted control models' saved per-session hits.
Only three NEW feature arms (18 objective/fold models) are fitted.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile
import numpy as np
from intent_features import NAMES, SUPPORT, TIMING, FAMILIES, transform, catalog

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
REPO_SHA = '2638faa34afa427ed4ac4e92bba04deda57c688e'
CUTOFF = 1660687200000
OBJECTIVES = ('clicks', 'carts', 'orders')
ARMS = ('plus_support', 'plus_timing', 'plus_both')
MODEL_PARAMS = {'learning_rate': 0.05, 'num_leaves': 15, 'min_data_in_leaf': 20,
                'lambda_l2': 1.0, 'lambdarank_truncation_level': 25,
                'objective': 'lambdarank', 'metric': 'None', 'verbosity': -1,
                'deterministic': True, 'force_col_wise': True, 'num_threads': 4,
                'seed': 20260911, 'feature_fraction': 1.0,
                'bagging_fraction': 1.0, 'bagging_freq': 0}
REF_PREFIX = 'ranking/research/feature-value-pilot/b9f53c29297a/output/'
NEG_KEY = ('ranking/research/55ad451e895863af311e4a917a6fe0d4ab9165ad6b406fffb066d25bf0af4754/'
           'robustness/1136d945b56c5ae4cd5f149e14736108430accee7a3b9dbf0da8681b694c8388/'
           'windows/early/inputs/fit_cache/contract.json')
REFERENCES = {
    'fold-0-shared.json': (REF_PREFIX+'fold-0-shared.json', '8KlO6sVC.L7qdLkBlZiV1l2kqXQbYfRa', 13855),
    'fold-1-shared.json': (REF_PREFIX+'fold-1-shared.json', 'Hy9rhn2qhxHaul3z8KcuOWDInsXDcnhV', 13917),
    'fold-0.json': (REF_PREFIX+'fold-0.json', 'VPgsF.igW.ckJL1u1KCuKVbL9AQyTof3', 9236),
    'fold-1.json': (REF_PREFIX+'fold-1.json', 'bq2NXNFPQnn3uJDLdDK9bgy5ngHOYpPy', 13264),
    'contract.json': (REF_PREFIX+'contract.json', '_.uD6hWaNTsTQRb1k2QiG7EetDvvPBHJ', 3415),
    'negative_contract.json': (NEG_KEY, 'xm1odUhK_uiBLnpT28T9mc8zCNPCFEic', 4282),
}
EXPECTED_BLOBS = {
    'scripts/run_feature_value_pilot.py': 'ebe2875bfcd0d78e70ff23acf39f6143765af68e',
    'src/otto_recsys/research/dataset.py': '2ff98f48a659c53093e8245fdb52f250ab27902c',
    'src/otto_recsys/research/graph_signals.py': 'a66986fa82d6576366c34a8ecdb556dbf1d5593c',
}
PROGRESS = {'stage': 'start', 'completed': 0, 'total': 0}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def encode(x):
    return (json.dumps(x, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def digest_data(b):
    return hashlib.sha256(b).hexdigest()


def write_once(path, data):
    """Never overwrite differing evidence; only remove this call's own temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Symlink output rejected')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Conflicting evidence preserved: {path}')
        return
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, x):
    write_once(path, encode(x))


def save_npz(path, **arrays):
    b = io.BytesIO(); np.savez_compressed(b, **arrays)
    data = b.getvalue()
    write_once(path, data)
    return digest_data(data)


def log(event, **kw):
    print(json.dumps({'utc': utc(), 'event': event, **kw}, sort_keys=True), flush=True)


def check_file(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f'Missing or changed verified input: {path}')


def check_sources(home):
    repo = home / 'otto-recommender-system'
    readiness_path = home / 'otto_manual_workspace/reports/readiness.json'
    readiness = json.loads(readiness_path.read_text())
    if readiness.get('readiness') != 'SOURCE_AND_INPUTS_VERIFIED':
        raise ValueError('Returned workspace readiness prerequisite is not passed')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True,
                                   timeout=10).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True,
                                    timeout=10).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True,
                                     timeout=10).strip()
    if head != REPO_SHA or dirty or branch != 'main':
        raise ValueError('Source changed or worktree not clean main; preserve local work for review')
    if readiness['environment'].get('status') != 'PIN_MATCH':
        raise ValueError('Existing version inventory does not match')
    import importlib.metadata
    for item in readiness['environment']['packages']:
        if importlib.metadata.version(item['package']) != item['expected']:
            raise ValueError(f'Environment changed: {item["package"]}; do not install blindly')
    if sys.version_info[:2] != (3, 13):
        raise ValueError('Use the existing project .venv Python 3.13')
    for relative, expected in EXPECTED_BLOBS.items():
        data = (repo / relative).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            raise ValueError(f'Reference API source mismatch: {relative}')
    sys.path.insert(0, str(repo / 'src'))
    spec = importlib.util.spec_from_file_location('otto_reference_pilot', repo / 'scripts/run_feature_value_pilot.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    base = home / 'otto-artifacts/shared-feature-smoke/early-smoke-91d3dcd1'
    paths = {'repo': repo, 'corpus': base / 'inputs/corpus',
             'graphs': base / 'output/graphs',
             'scale': home / 'otto-artifacts/shared-feature-scale/early-scale-29439ec2/output'}
    log('source_and_runtime_passed', commit=head, installs=0, git_changes=0)
    return paths, module


def load_cohort(paths, with_targets=False):
    import polars as pl
    repo, corpus, scale = (paths[k] for k in ('repo', 'corpus', 'scale'))
    schema_path = repo / 'configs/shared_feature_validation.json'
    cfg = json.loads(schema_path.read_text())
    names = cfg['baseline_features'] + cfg['added_features']
    if len(names) != 134 or len(set(names)) != 134 or set(names).intersection(NAMES):
        raise ValueError('Frozen feature schema differs')
    source = json.loads((repo / 'reports/research/early_shared_feature_scale.json').read_text())
    if source['status'] != 'EARLY_SHARED_FEATURE_SCALE_PASSED':
        raise ValueError('Saved scale result not passed')
    cols = ['session', 'aid', 'candidate_position']
    if with_targets:
        cols += names + [f'target_{o}' for o in OBJECTIVES]
    frames, hashes = [], {'schema': sha(schema_path)}
    for part in source['outputs']['parts']:
        p = scale / part['part']
        if p.parent.resolve() != scale.resolve():
            raise ValueError('Input path escapes scale directory')
        check_file(p, part['sha256']); hashes[p.name] = part['sha256']
        frames.append(pl.read_parquet(p, columns=cols))
    frame = pl.concat(frames).sort('session', 'candidate_position')
    ids = frame['session'].unique().sort().to_numpy()
    if (len(ids) != 1024 or frame.height != 409600
            or not np.array_equal(frame['session'].to_numpy(), np.repeat(ids, 400))
            or not np.array_equal(frame['candidate_position'].to_numpy(), np.tile(np.arange(400), 1024))):
        raise ValueError('Expected 1024 complete candidate groups in canonical order')
    aids = frame['aid'].to_numpy().reshape(1024, 400)
    if any(len(np.unique(a)) != 400 for a in aids):
        raise ValueError('Duplicate candidate IDs')
    ledger_path = scale / 'fit_query_ledger.parquet'
    check_file(ledger_path, source['outputs']['query_ledger']['sha256'])
    hashes[ledger_path.name] = source['outputs']['query_ledger']['sha256']
    manifest = json.loads((corpus / 'manifest.json').read_text())
    if manifest['protocol']['history_end'] != CUTOFF or manifest['status'] != 'passed':
        raise ValueError('Wrong historical corpus')
    filenames = ['queries.parquet'] + (['labels.parquet'] if with_targets else ['observed.parquet'])
    tables = {}
    for filename in filenames:
        check_file(corpus / filename, manifest['files'][filename])
        hashes['corpus/'+filename] = manifest['files'][filename]
        tables[filename] = (pl.scan_parquet(corpus / filename)
                           .filter((pl.col('split_role') == 'fit') & pl.col('session').is_in(ids.tolist()))
                           .collect())
    ledger = tables['queries.parquet']
    if ledger.height != 3072:
        raise ValueError('Expected one full ledger row per session and objective')
    query_ts = np.empty(1024, dtype=np.int64)
    denom = np.empty((1024, 3), dtype=np.int64)
    true_counts = np.empty_like(denom)
    prefix_count = None
    for j, o in enumerate(OBJECTIVES):
        rows = ledger.filter(pl.col('objective') == o).sort('session')
        if not np.array_equal(rows['session'].to_numpy(), ids):
            raise ValueError('Objective ledger session mismatch')
        q = rows['query_ts'].to_numpy()
        if j == 0:
            query_ts[:] = q; prefix_count = rows['observed_events'].to_numpy()
        elif not np.array_equal(query_ts, q):
            raise ValueError('Inconsistent query timestamp across objectives')
        denom[:, j] = rows['recall_denominator'].to_numpy()
        true_counts[:, j] = rows['true_items'].to_numpy()
    if not np.array_equal(denom, np.minimum(true_counts, 20)):
        raise ValueError('Incomplete denominator ledger')
    frozen_ledger = pl.read_parquet(ledger_path).sort('session')
    if (not np.array_equal(frozen_ledger['session'].to_numpy(), ids)
            or not np.array_equal(frozen_ledger.select([f'denominator_{o}' for o in OBJECTIVES]).to_numpy(), denom)):
        raise ValueError('Frozen denominator ledger does not match corpus ledger')
    if denom.sum(axis=0).tolist() != [987, 340, 168]:
        raise ValueError('Frozen cohort denominator mismatch')
    data = {'ids': ids, 'aids': aids, 'query_ts': query_ts, 'denom': denom,
            'names': names, 'hashes': hashes}
    if with_targets:
        x = frame.select(names).to_numpy().astype(np.float32).reshape(1024, 400, 134)
        y = frame.select([f'target_{o}' for o in OBJECTIVES]).to_numpy().reshape(1024, 400, 3)
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Invalid saved features or targets')
        by_label, events = {}, {}
        for session, aid, objective, stamp, index in tables['labels.parquet'].select(
                'session', 'aid', 'objective', 'label_ts', 'label_event_index').iter_rows():
            j = OBJECTIVES.index(objective)
            by_label.setdefault(int(session), []).append((int(aid), j, int(stamp)))
            events.setdefault((int(session), j), []).append((int(aid), int(stamp), int(index)))
        max_index = dict(ledger.filter(pl.col('objective') == 'clicks').select(
            'session', 'observed_last_index').iter_rows())
        for i, session in enumerate(ids):
            for j in range(3):
                rows = events.get((int(session), j), [])
                truth = [r[0] for r in rows]
                if len(truth) != len(set(truth)) or len(truth) != true_counts[i, j]:
                    raise ValueError('Duplicate/missing full target set')
                if any(t < query_ts[i] or t >= 1660946400000 or k <= max_index[int(session)]
                       for _, t, k in rows):
                    raise ValueError('Future target boundary/index violation')
                if not np.array_equal(y[i, :, j], np.isin(aids[i], truth)):
                    raise ValueError('Saved candidate target parity failed')
        data.update(x=x, y=y.astype(np.int8), labels=by_label)
    else:
        prefixes = {}
        obs = tables['observed.parquet'].sort('session', 'event_index')
        for group in obs.partition_by('session', maintain_order=True):
            session = int(group['session'][0]); i = int(np.searchsorted(ids, session))
            if (group.height != prefix_count[i]
                    or group['event_index'].to_list() != list(range(group.height))):
                raise ValueError('Missing or non-contiguous observed event prefix')
            ts = group['ts'].to_numpy()
            if int(ts[-1]) != query_ts[i] or int(ts[0]) < CUTOFF:
                raise ValueError('Observed query cutoff differs')
            prefixes[session] = (group['aid'].to_numpy(), ts, group['event_type'].to_numpy())
        if set(prefixes) != set(map(int, ids)):
            raise ValueError('Missing fitting prefix')
        data['prefixes'] = prefixes
    return data


def build_features(paths):
    data = load_cohort(paths, False)
    source = json.loads((paths['repo'] / 'reports/research/early_shared_feature_smoke.json').read_text())
    manifests = {}
    for f in FAMILIES:
        m = json.loads((paths['graphs'] / f / 'manifest.json').read_text())
        if (m['input_id'] != source['graphs'][f]['input_id'] or m['history_end'] != CUTOFF
                or m['query_labels_used'] is not False or m['observed_history_max_ts'] >= CUTOFF):
            raise ValueError('Graph cutoff or identity differs')
        manifests[f] = sha(paths['graphs'] / f / 'manifest.json')
    contract = {'source_commit': REPO_SHA, 'source_hashes': data['hashes'],
                'graph_manifests': manifests, 'names': list(NAMES),
                'code_sha256': sha(ROOT / 'intent_features.py'),
                'runner_sha256': sha(Path(__file__)),
                'protocol_sha256': sha(ROOT / 'protocol.json'),
                'ids': data['ids'].tolist(), 'self_edges': False}
    save_json(OUT / 'feature_contract.json', contract)
    cid = digest_data(encode(contract))
    PROGRESS.update(stage='load_verified_graphs', completed=0, total=1024)
    GraphSignals = importlib.import_module('otto_recsys.research.graph_signals').GraphSignals
    graph_objects = {f: GraphSignals(paths['graphs'] / f, f) for f in FAMILIES}
    graphs = {f: g.graphs for f, g in graph_objects.items()}
    PROGRESS['stage'] = 'feature_chunks'
    receipts = []
    for begin in range(0, 1024, 16):
        end = begin+16; chunk = OUT / 'features' / f'part-{begin//16:03d}.npz'
        receipt_path = chunk.with_suffix('.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['contract_id'] != cid or receipt['start'] != begin:
                raise ValueError('Feature chunk belongs to another contract')
            check_file(chunk, receipt['sha256'])
            with np.load(chunk, allow_pickle=False) as z:
                if not np.array_equal(z['ids'], data['ids'][begin:end]):
                    raise ValueError('Feature chunk ID mismatch')
            log('feature_chunk_reused', start=begin, end=end)
        else:
            if chunk.exists():
                raise ValueError('Orphan feature checkpoint preserved for inspection')
            tick = time.monotonic(); blocks = []
            for i in range(begin, end):
                args = (*data['prefixes'][int(data['ids'][i])], data['aids'][i], graphs, CUTOFF)
                x = transform(*args)
                # Full numerical reconstruction checks for the first 16; later checks use hashes.
                if begin == 0 and not np.array_equal(x, transform(*args)):
                    raise ValueError('16-session smoke replay differs')
                blocks.append(x)
            array = np.stack(blocks)
            h = save_npz(chunk, ids=data['ids'][begin:end], x=array)
            receipt = {'start': begin, 'end': end, 'contract_id': cid, 'sha256': h,
                       'rows': 16*400, 'features': 48, 'seconds': time.monotonic()-tick}
            save_json(receipt_path, receipt)
            log('feature_chunk_complete', start=begin, end=end, seconds=receipt['seconds'])
        receipts.append({'path': str(chunk.relative_to(OUT)), **receipt})
        PROGRESS['completed'] = end
    result = {'status': 'ROUND01_FEATURES_READY', 'contract_id': cid, 'sessions': 1024,
              'rows': 409600, 'new_features': 48, 'parts': receipts,
              'smoke_16_sessions': 'deterministic_replay_passed', 'model_fits': 0,
              'candidate_changes': 0, 'graph_rebuilds': 0, 'selection_access': False,
              'evaluation_access': False, 'raw_competition_test_access': False}
    save_json(OUT / 'feature_manifest.json', result)
    save_json(OUT / 'feature_catalog.json', catalog())
    return result


def load_references():
    """Only ~58 KB of pinned previous experiment receipts, not datasets or new runs."""
    refs = {}
    for name, (key, version, size) in REFERENCES.items():
        dest = OUT / 'reference' / name; receipt = dest.with_suffix('.download.json')
        if dest.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['version_id'] != version:
                raise ValueError('Reference version changed')
            check_file(dest, saved['sha256'])
        else:
            if dest.exists():
                raise ValueError('Orphan reference file retained for review')
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.transfer')
            if tmp.exists():
                raise ValueError('Interrupted small reference transfer; return report for review')
            cmd = ['aws', 's3api', 'get-object', '--region', 'us-west-2', '--bucket',
                   'otto-recsys-560403859723-us-west-2', '--key', key, '--version-id', version,
                   '--expected-bucket-owner', '560403859723', '--cli-connect-timeout', '5',
                   '--cli-read-timeout', '20', str(tmp)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=35,
                                    env={**os.environ, 'AWS_MAX_ATTEMPTS': '1', 'AWS_PAGER': ''})
            if result.returncode:
                raise RuntimeError('Reference download failed: '+result.stderr[-1500:])
            meta = json.loads(result.stdout)
            if tmp.stat().st_size != size or meta.get('VersionId') != version:
                raise ValueError('Pinned reference length/version mismatch')
            os.link(tmp, dest); tmp.unlink()
            save_json(receipt, {'key': key, 'version_id': version, 'bytes': size, 'sha256': sha(dest)})
        refs[name] = json.loads(dest.read_text())
    if sha(OUT / 'reference/negative_contract.json') != refs['contract.json']['sources']['candidate_contract']:
        raise ValueError('Negative-sampling reference mismatch')
    return refs


def validate_reference(data, paths, ref, folds, helper):
    source = ref['contract.json']
    for key, expected in source['sources'].items():
        if key in data['hashes'] and data['hashes'][key] != expected:
            raise ValueError(f'Control cohort artifact mismatch: {key}')
    if (source['schema_sha256'] != data['hashes']['schema']
            or source['runner_sha256'] != sha(paths['repo'] / 'scripts/run_feature_value_pilot.py')):
        raise ValueError('Previous control source/schema no longer matches')
    if source['protocol']['lightgbm'] != {k: MODEL_PARAMS[k] for k in source['protocol']['lightgbm']}:
        raise ValueError('Model settings mismatch')
    if source['protocol']['rounds'] != 150 or source['protocol']['seed'] != 20260911:
        raise ValueError('Control rounds/seed mismatch')
    controls = []
    for j, fold in enumerate(folds):
        old_fold, control = ref[f'fold-{j}.json'], ref[f'fold-{j}-shared.json']
        ti, vi = fold['train'], fold['valid']
        if (old_fold['train_ids'] != data['ids'][ti].tolist()
                or old_fold['valid_ids'] != data['ids'][vi].tolist()
                or control['session_ids'] != data['ids'][vi].tolist()
                or old_fold['cutoff_ms'] != fold['cutoff']):
            raise ValueError('Saved control cannot be compared: fold identity differs')
        h = np.asarray(control['hits_by_session'], dtype=np.int64)
        metric = helper.pooled(h, data['denom'][vi])
        if (metric['hits'] != control['metrics']['hits']
                or metric['denominators'] != control['metrics']['denominators']
                or abs(metric['weighted_recall_at_20']-control['metrics']['weighted_recall_at_20']) > 1e-12):
            raise ValueError('Saved control metric replay mismatch')
        controls.append(h)
    return controls


def paired_interval(delta, den, fold_ids, seed=20260911):
    """Paired session bootstrap within each fold; descriptive, not independent-time uncertainty."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(fold_ids == f) for f in np.unique(fold_ids)]
    values = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        d = den[idx].sum(axis=0)
        if (d > 0).all():
            values.append(float((delta[idx].sum(axis=0)/d) @ np.array([.1, .3, .6])))
    if len(values) < 900:
        raise ValueError('Insufficient bootstrap objective support')
    return {'descriptive_95_interval': np.quantile(values, [.025, .975]).tolist(),
            'replicates': len(values), 'method': 'paired, stratified by forward fold; unadjusted'}


def classify(gain, fold_gains, order_gain, interval):
    if gain <= 0:
        return 'STOP_THIS_VARIANT'
    if min(fold_gains) < 0 or order_gain < 0:
        return 'MIXED_DIAGNOSE_NO_AUTOMATIC_SCALE'
    if gain < .003:
        return 'SMALL_GAIN_LOW_PRIORITY'
    return ('REPLICATE_LARGER_FITTING_ONLY' if interval[0] > 0
            else 'PROMISING_POINT_ESTIMATE_UNCERTAIN_REPLICATE_ONLY')


def screen(paths, helper):
    data = load_cohort(paths, True)
    manifest = json.loads((OUT / 'feature_manifest.json').read_text())
    contract = json.loads((OUT / 'feature_contract.json').read_text())
    if manifest['contract_id'] != digest_data(encode(contract)):
        raise ValueError('Feature manifest identity mismatch')
    if (contract['code_sha256'] != sha(ROOT / 'intent_features.py')
            or contract['runner_sha256'] != sha(Path(__file__))
            or contract['protocol_sha256'] != sha(ROOT / 'protocol.json')):
        raise ValueError('Feature code/protocol changed after generation')
    if contract['ids'] != data['ids'].tolist():
        raise ValueError('Feature cohort changed')
    blocks = []
    for entry in manifest['parts']:
        path = OUT / entry['path']; check_file(path, entry['sha256'])
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['ids'], data['ids'][entry['start']:entry['end']]):
                raise ValueError('Feature identity/order mismatch')
            blocks.append(z['x'])
    extra = np.concatenate(blocks)
    if extra.shape != (1024, 400, 48) or not np.isfinite(extra).all():
        raise ValueError('New feature shape/finiteness mismatch')
    refs = load_references()
    folds = helper.purged_folds(data['query_ts'])
    controls = validate_reference(data, paths, refs, folds, helper)
    negative = refs['negative_contract.json']
    if negative['negative_budget'] != 60 or negative['seed'] != 20260908:
        raise ValueError('Negative sampling protocol differs')
    from otto_recsys.research.dataset import sampled_rows
    all_names = data['names']+list(NAMES)
    all_data = np.concatenate([data['x'], extra], axis=2)
    colsets = {'plus_support': list(range(134))+[134+i for i in SUPPORT],
               'plus_timing': list(range(134))+[134+i for i in TIMING],
               'plus_both': list(range(182))}
    study = {'feature_contract_id': manifest['contract_id'], 'source_hashes': data['hashes'],
             'source_commit': REPO_SHA, 'protocol_sha256': sha(ROOT / 'protocol.json'),
             'reference_hashes': {n: sha(OUT / 'reference' / n) for n in REFERENCES},
             'params': MODEL_PARAMS, 'rounds': 150, 'code_sha256': sha(Path(__file__))}
    save_json(OUT / 'screen_contract.json', study)
    PROGRESS.update(stage='fitting_new_arms', completed=0, total=18)
    per_arm = {'control_shared': controls, **{a: [] for a in ARMS}}
    all_den, all_ids, group_fold, records, diagnostics = [], [], [], [], []
    fits, reused = 0, 0
    for k, fold in enumerate(folds):
        train, valid = fold['train'], fold['valid']
        den = data['denom'][valid]; all_den.append(den); all_ids.append(data['ids'][valid])
        group_fold.extend([k]*len(valid))
        yt = helper.targets_asof(data['ids'][train], data['aids'][train], data['query_ts'][train],
                                 data['labels'], fold['cutoff'])
        chosen = [sampled_rows(yt[j], data['aids'][i], int(data['ids'][i]), 60, 20260908)
                  for j, i in enumerate(train)]
        groups = [len(c) for c in chosen]
        xt = np.concatenate([all_data[i, c] for i, c in zip(train, chosen, strict=True)])
        yy = np.concatenate([yt[j, c] for j, c in enumerate(chosen)])
        if yy.sum(axis=0).tolist() != refs[f'fold-{k}.json']['positive_training_rows']:
            raise ValueError('Training positives differ from saved control; stop before training')
        for group_name, group_columns in [('support', SUPPORT), ('timing', TIMING)]:
            if not (np.ptp(extra[train][:, :, list(group_columns)], axis=(0, 1)) > 0).any():
                raise ValueError(f'{group_name} has no variation in this training fold; stop before pointless fits')
        # Unsupervised redundancy/support diagnostic on TRAIN sessions only, no feature deletion.
        sample = np.sort(np.random.default_rng(20260911).choice(train, min(64, len(train)), False))
        diagnostic_x = all_data[sample].reshape(-1, 182).astype(np.float64)
        st = diagnostic_x.std(axis=0); centered = diagnostic_x-diagnostic_x.mean(axis=0)
        normalized = centered/np.where(st > 0, st, 1)
        corr = np.abs(normalized[:, 134:].T @ normalized[:, :134]/len(normalized))
        for j, name in enumerate(NAMES):
            diagnostics.append({'fold': k, 'feature': name,
                'group': 'support' if j in SUPPORT else 'timing',
                'sampled_train_sessions': len(sample), 'std': float(st[134+j]),
                'nonzero_fraction': float((diagnostic_x[:, 134+j] != 0).mean()),
                'max_abs_corr_baseline': float(corr[j].max()),
                'decision': 'diagnostic only; no automatic feature removal'})
        valid_data = all_data[valid].reshape(-1, 182)
        records.append({'fold': k, 'arm': 'control_shared', 'features': 134,
                        **helper.pooled(controls[k], den), 'model_fits': 0,
                        'source': 'verified saved per-session control hits'})
        for arm, cols in colsets.items():
            arm_start = time.monotonic(); hits = np.zeros((len(valid), 3), dtype=np.int64)
            names = [all_names[i] for i in cols]
            for j, objective in enumerate(OBJECTIVES):
                contract_m = {'study': study, 'fold': k, 'arm': arm, 'objective': objective,
                              'features': names, 'train_ids': data['ids'][train].tolist(),
                              'valid_ids': data['ids'][valid].tolist()}
                prediction, count = helper.fit_or_reuse(OUT / 'models' / f'fold-{k}' / arm / objective,
                    contract_m, xt[:, cols], yy[:, j], groups, valid_data[:, cols],
                    MODEL_PARAMS, 150, names)
                fits += count; reused += 1-count
                hits[:, j] = helper.session_hits(prediction.reshape(-1, 400), data['aids'][valid],
                                                 data['y'][valid, :, j])
                PROGRESS['completed'] += 1
                log('objective_checkpoint', fold=k, arm=arm, objective=objective,
                    new_fit=count, completed=PROGRESS['completed'], total=18)
            per_arm[arm].append(hits)
            metric = helper.pooled(hits, den)
            evidence = {'fold': k, 'arm': arm, 'session_ids': data['ids'][valid].tolist(),
                        'hits_by_session': hits.tolist(), 'denominators': den.tolist(),
                        'metrics': metric, 'features': names}
            save_json(OUT / 'arm_reports' / f'fold-{k}-{arm}.json', evidence)
            records.append({'fold': k, 'arm': arm, 'features': len(names), **metric})
            log('arm_complete', fold=k, arm=arm, seconds=time.monotonic()-arm_start, **metric)
    h = {a: np.concatenate(v) for a, v in per_arm.items()}
    den = np.concatenate(all_den); fids = np.array(group_fold)
    aggregate = {a: helper.pooled(v, den) for a, v in h.items()}
    comparisons = {}
    pairs = [(a, 'control_shared') for a in ARMS] + [('plus_both', 'plus_support'),
                                                    ('plus_both', 'plus_timing')]
    for a, b in pairs:
        diff = aggregate[a]['weighted_recall_at_20']-aggregate[b]['weighted_recall_at_20']
        fg = [helper.pooled(per_arm[a][k], all_den[k])['weighted_recall_at_20'] -
              helper.pooled(per_arm[b][k], all_den[k])['weighted_recall_at_20'] for k in range(2)]
        ci = paired_interval(h[a]-h[b], den, fids)
        order_gain = aggregate[a]['recall']['orders']-aggregate[b]['recall']['orders']
        comparisons[a+'__minus__'+b] = {'gain': diff, 'fold_gains': fg,
            'order_gain': order_gain, **ci,
            'decision': classify(diff, fg, order_gain, ci['descriptive_95_interval'])}
    save_json(OUT / 'support_diagnostics.json', diagnostics)
    save_npz(OUT / 'evaluation_statistics.npz', ids=np.concatenate(all_ids), denominator=den,
             folds=fids, **h)
    # Execution counts go to run receipts, not the immutable scientific result.
    result = {'status': 'ROUND01_SCREEN_COMPLETED', 'source_commit': REPO_SHA,
              'sessions': 1024, 'validated_sessions': 512, 'new_feature_count': 48,
              'control_features': 134, 'control_models_refitted': 0,
              'maximum_new_models': 18, 'comparisons': comparisons, 'arms': aggregate,
              'fold_results': records, 'primary_comparison': 'plus_both__minus__control_shared',
              'selection_access': False, 'evaluation_access': False, 'competition_test_access': False,
              'graph_rebuilds': 0, 'candidate_changes': 0, 'feature_retention_decisions': 0,
              'feature_engineering_complete': False,
              'limitations': ['Small previously used fitting cohort, one model seed, two dependent forward folds.',
                  'Retrospective prefixes; training targets censored at fold cutoff, not an online deployment backtest.',
                  'Only 92 pooled order-denominator units. Bootstrap intervals descriptive/unadjusted.',
                  'A development gain is not a Kaggle gain or proof of beating 0.60503.',
                  'Pending-cart means observed cart after last observed order; cart removals and actual inventory unavailable.'],
              'metric': 'pooled 0.1 click + 0.3 cart + 0.6 order Recall@20; complete candidate pools'}
    inventory = {str(p.relative_to(OUT)): sha(p) for p in sorted((OUT/'models').rglob('model.txt'))}
    if len(inventory) != 18:
        raise ValueError('Expected 18 new-arm native model checkpoints')
    result['model_inventory'] = inventory
    result['native_reload_prediction_parity'] = True
    save_json(OUT / 'result.json', result)
    log('screen_complete', new_models=fits, reused_new_models=reused, control_refits=0,
        primary=comparisons[result['primary_comparison']])
    return {'status': result['status'], 'new_models': fits, 'reused_new_models': reused,
            'control_refits': 0}


def bundle():
    """Only small diagnostics/receipts: no raw data or feature/model bulk in return ZIP."""
    ROOT.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    manifest = []
    files = []
    for p in sorted(OUT.rglob('*')):
        if (p.is_file() and not p.is_symlink() and '.tmp' not in p.name
                and 'features' not in p.relative_to(OUT).parts and 'models' not in p.relative_to(OUT).parts
                and p.suffix in ('.json', '.log', '.html', '.npz', '.md')):
            files.append(p)
    if sum(p.stat().st_size for p in files) > 30*1024**2:
        raise ValueError('Return bundle exceeds 30 MiB; artifacts preserved locally')
    with zipfile.ZipFile(ROOT / 'otto_round01_return.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = str(p.relative_to(ROOT)); z.write(p, rel)
            manifest.append({'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size})
        z.writestr('RETURN_MANIFEST.json', encode(manifest))
    return ROOT / 'otto_round01_return.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('features', 'screen', 'report', 'bundle'))
    parser.add_argument('--home', type=Path, default=Path.home())
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rid = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    status = {'phase': args.phase, 'started_utc': utc(), 'status': 'RUNNING'}
    stop = threading.Event(); start = time.monotonic()
    def beat():
        while not stop.wait(15):
            import psutil
            rss = psutil.Process().memory_info().rss/1024**3
            log('heartbeat', elapsed_seconds=round(time.monotonic()-start, 2), rss_GiB=round(rss, 3), **PROGRESS)
            if rss > 26:
                os.kill(os.getpid(), signal.SIGUSR1)
    def time_limit(signum, frame):
        raise TimeoutError('Phase time/memory cap reached; completed checkpoints preserved')
    signal.signal(signal.SIGALRM, time_limit); signal.signal(signal.SIGUSR1, time_limit)
    signal.signal(signal.SIGTERM, time_limit)
    signal.alarm({'features': 300, 'screen': 240, 'report': 60, 'bundle': 30}[args.phase])
    lock = (OUT / '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('STOPPED_REVIEW_REQUIRED: another Round 01 process holds the lock', flush=True)
        return 2
    thread = threading.Thread(target=beat, daemon=True); thread.start()
    code = 0
    try:
        # Package files are locally fingerprinted; no runtime code downloads.
        if args.phase in ('features', 'screen'):
            paths, helper = check_sources(args.home)
            value = build_features(paths) if args.phase == 'features' else screen(paths, helper)
        elif args.phase == 'report':
            from visualize import build_report
            value = build_report(ROOT)
        else:
            value = {'status': 'RETURN_BUNDLE_READY'}
        status.update(value)
    except Exception as error:
        import traceback
        status.update(status='STOPPED_REVIEW_REQUIRED', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(limit=6), progress=PROGRESS.copy())
        code = 2
    finally:
        signal.alarm(0); stop.set(); thread.join(timeout=2)
        status.update(finished_utc=utc(), elapsed_seconds=round(time.monotonic()-start, 3))
        save_json(OUT / 'runs' / f'{rid}.json', status)
        try:
            return_zip = bundle()
            print('RETURN_FILE: '+str(return_zip), flush=True)
        except Exception as error:
            print('BUNDLE_ERROR: '+str(error), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    print('RESULT: '+status['status'], flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
