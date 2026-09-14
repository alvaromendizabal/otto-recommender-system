#!/usr/bin/env python3
"""Owner-run PUBLIC-only OTTO source publication and separately gated import recovery.

Only alvaromendizabal/otto-recommender-system can be a publication target.
For the diagnosed staged snapshot: self-test -> resume-public -> review -> publish-public.
No repository creation/visibility changes, forced updates, AWS API calls, training,
or mutation of the original execution checkout. Run self-test, prepare-public,
review the files, then publish-public --approve-public. finish-public is separate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock
import urllib.request
import zipfile

VERSION = 'otto-public-sync-20260914-index-recovery1'
TARGET = 'alvaromendizabal/otto-recommender-system'
OWNER = 'alvaromendizabal'
HOME = Path.home()
STATE = HOME / 'otto_public_sync_state'
ARCHIVE = 'otto_research_and_portfolio.zip'
ARCHIVE_SHA = '36c2acab11fd8e862243cdf13f3361243fce43fbfa2b2cf5fe097cb0fb357197'
GH_VERSION = '2.100.0'
GH = None
RUN = None
EVENTS = []
MAX_FILE = 20 * 1024**2
MAX_TOTAL = 250 * 1024**2
TEXT_EXT = {'.py', '.ipynb', '.md', '.rst', '.json', '.toml', '.yaml', '.yml',
            '.cfg', '.ini', '.sh', '.txt', '.sha256', '.html', '.svg', '.lock', '.in'}
SPECIAL_NAMES = {'LICENSE', 'NOTICE', 'COPYING', 'Makefile', 'Dockerfile',
                 '.gitignore', '.gitattributes', '.python-version', 'MANIFEST.sha256'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints', '.aws',
             '.ssh', '.config', '.cache', 'node_modules', 'data', 'raw', 'artifacts',
             'checkpoints', 'models', 'index', 'shared', 'features',
             'representation_inputs', 'representations'}
BAD_NAMES = {'kaggle.json', 'credentials', 'credentials.json', 'id_rsa', 'id_ed25519',
             'hosts.yml', '.netrc', '.npmrc', '.pypirc', '.git-credentials'}
SECRET_RULES = [
    ('aws_access_key', rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    ('github_token', rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    ('private_key', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    ('signed_url', rb'X-Amz-(?:Signature|Credential)='),
    ('credential_in_url', rb'https?://[^\s/"<>:]+:[^\s/@"<>]+@'),
    ('assigned_secret', rb'''(?im)^\s*(?:export\s+)?(?:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|GITHUB_TOKEN|GH_TOKEN|HF_TOKEN|KAGGLE_KEY)\s*[:=]\s*["']?[A-Za-z0-9/+_=-]{16,}'''),
]
ROW_KEYS = {'session', 'session_id', 'session_ids', 'ids', 'excluded_sessions',
            'aid', 'aids', 'labels', 'predictions', 'targets', 'candidate_items',
            'true_items', 'query_ids'}
OUTPUT_NAMES = {'result.json', 'candidate_coverage.json', 'feature_diagnostics.json',
                'report_receipt.json', 'feature_manifest.json', 'representation_diagnostics.json'}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def safe_relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('Unsafe relative path')
    return p.as_posix()


def safe_path(root, relative):
    p = Path(root) / safe_relative(relative)
    if not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes its root')
    for q in [p, *p.parents]:
        if q.is_symlink():
            raise ValueError('Symlink path refused')
        if q == Path(root):
            break
    return p


def atomic(path, data):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Symlink write refused')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    if temp.exists():
        raise ValueError('Earlier incomplete write preserved: ' + temp.name)
    with temp.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def redact(value):
    b = str(value).encode('utf-8', 'replace')
    b = re.sub(rb'X-Amz-(?:Signature|Credential|Security-Token)=[^\s&\"<>]+',
               b'[REDACTED_QUERY_PARAMETER]', b)
    for _, pattern in SECRET_RULES:
        b = re.sub(pattern, b'[REDACTED]', b)
    b = re.sub(rb'\x1b\[[0-?]*[ -/]*[@-~]', b'', b)
    return b.decode('utf-8', 'replace')


def record(kind, **values):
    row = {'utc': utc(), 'event': kind, **values}
    EVENTS.append(row)
    print(kind, json.dumps(values, sort_keys=True), flush=True)
    if RUN:
        atomic(RUN / 'receipt.json', encoded({'version': VERSION, 'target': TARGET,
                                            'events': EVENTS, 'model_fits': 0}))


def command(args, cwd=None, timeout=90, allowed=(0,), interactive=False, binary=False,
            result_object=False):
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0',
           'GH_PROMPT_DISABLED': '1', 'GH_NO_UPDATE_NOTIFIER': '1',
           'GH_TELEMETRY_DISABLED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'LC_ALL': 'C', 'LANG': 'C'}
    env.pop('GH_DEBUG', None)
    if interactive:
        env.pop('GH_PROMPT_DISABLED', None)
        result = subprocess.run(args, cwd=cwd, env=env, timeout=timeout, check=False)
    else:
        # Drain to bounded files rather than pipes; report process-output progress without credentials.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=output, stderr=errors,
                                    start_new_session=True)
            start = beat = time.monotonic()
            try:
                while proc.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed > timeout:
                        raise TimeoutError(f'{Path(args[0]).name} exceeded its {timeout}s process cap')
                    if output.tell() + errors.tell() > 64 * 1024**2:
                        raise RuntimeError('Command output limit reached; stop and review')
                    if time.monotonic() - beat >= 15:
                        record('COMMAND_HEARTBEAT', command=Path(args[0]).name,
                               elapsed_seconds=round(elapsed, 1),
                               output_bytes=output.tell() + errors.tell())
                        beat = time.monotonic()
                    time.sleep(0.1)
                output.seek(0); errors.seek(0)
                result = subprocess.CompletedProcess(args, proc.returncode, output.read(), errors.read())
            finally:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=3)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=3)

    if result.returncode not in allowed:
        stdout = '' if interactive else result.stdout.decode('utf-8', 'replace')
        stderr = '' if interactive else result.stderr.decode('utf-8', 'replace')
        if '--check' in args:
            # Whitespace diagnostics include added source lines; do not echo them.
            stdout = '\n'.join(line for line in stdout.splitlines() if not line.startswith('+'))
        details = {'command': redact(shlex.join([str(a) for a in args])),
                   'exit_code': result.returncode,
                   'stdout': redact(stdout)[-6000:], 'stderr': redact(stderr)[-6000:]}
        if RUN:
            atomic(RUN / 'command_failure.json', encoded(details))
        raise RuntimeError(f"{Path(args[0]).name} exited {result.returncode}; "
                           f"command: {details['command']}; "
                           f"stdout: {details['stdout'] or '<empty>'}; "
                           f"stderr: {details['stderr'] or '<empty>'}")
    if result_object:
        return result
    output = b'' if interactive else result.stdout
    return output if binary else output.decode('utf-8', 'replace')


def git(repo, *args, binary=False, timeout=90, allowed=(0,)):
    options = ['git']
    if GH:
        options += ['-c', 'credential.helper=', '-c',
                    'credential.helper=!' + shlex.quote(str(GH)) + ' auth git-credential']
    return command(options + ['-C', str(repo), *args], timeout=timeout,
                   binary=binary, allowed=allowed)


def api(endpoint, method='GET', payload=None):
    # This is an allowlist of the single project plus authenticated identity.
    if endpoint != 'user' and not endpoint.startswith('repos/' + TARGET):
        raise ValueError('Only the designated public repository is allowed')
    args = [str(GH), 'api', '--hostname', 'github.com', endpoint, '--method', method]
    if payload is None:
        result = command(args, timeout=60)
    else:
        p = RUN / 'api-input.json'
        atomic(p, encoded(payload))
        p.chmod(0o600)
        try:
            result = command(args + ['--input', str(p)], timeout=60)
        finally:
            p.unlink(missing_ok=True)
    return json.loads(result) if result.strip() else {}


def download(url, limit):
    if not url.startswith('https://github.com/cli/cli/releases/download/'):
        raise ValueError('Only official GitHub CLI release downloads allowed')
    request = urllib.request.Request(url, headers={'User-Agent': VERSION})
    with urllib.request.urlopen(request, timeout=45) as response:
        value = response.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Download exceeds size limit')
    return value


def github_cli(install=False):
    global GH
    candidates = [shutil.which('gh'), HOME / '.local/bin/gh',
                  STATE / 'tools' / ('gh-' + GH_VERSION)]
    GH = next((str(p) for p in candidates if p and Path(p).is_file() and os.access(p, os.X_OK)), None)
    if GH is None:
        if not install:
            raise RuntimeError('GitHub CLI is absent. Use --install-gh-if-needed on prepare-public, not pip/conda.')
        arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
        if not arch or platform.system() != 'Linux':
            raise RuntimeError('This optional CLI installation is for the SageMaker Linux terminal.')
        name = f'gh_{GH_VERSION}_linux_{arch}.tar.gz'
        base = f'https://github.com/cli/cli/releases/download/v{GH_VERSION}/'
        lines = download(base + f'gh_{GH_VERSION}_checksums.txt', 1024**2).decode().splitlines()
        expected = next((v[0] for line in lines if len(v := line.split()) == 2
                         and v[1].lstrip('*') == name), None)
        if not expected or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Official CLI checksum entry is absent')
        data = download(base + name, 50 * 1024**2)
        if digest(data) != expected:
            raise ValueError('Official CLI archive checksum mismatch')
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            member = tar.getmember(f'gh_{GH_VERSION}_linux_{arch}/bin/gh')
            if not member.isfile() or member.size > 90 * 1024**2:
                raise ValueError('Unexpected CLI archive member')
            with tar.extractfile(member) as handle:
                binary = handle.read()
        p = STATE / 'tools' / ('gh-' + GH_VERSION)
        atomic(p, binary)
        p.chmod(0o700)
        GH = str(p)
        record('GITHUB_CLI_INSTALLED_SEPARATELY', version=GH_VERSION, archive_sha256=expected)
    try:
        command([GH, 'auth', 'status', '--hostname', 'github.com'], timeout=20)
    except RuntimeError:
        print('Use the browser URL and device code below. Do not share credentials in ChatGPT.', flush=True)
        command([GH, 'auth', 'login', '--hostname', 'github.com', '--git-protocol',
                 'https', '--web', '--scopes', 'workflow'], interactive=True, timeout=600)
    identity = api('user')
    if identity.get('login') != OWNER:
        raise ValueError('Authenticated user must be ' + OWNER)
    info = api('repos/' + TARGET)
    if info.get('full_name') != TARGET or info.get('private') is not False:
        raise ValueError('Existing target must already be the designated PUBLIC repository. No visibility changes are made.')
    if info.get('default_branch') != 'main' or not info.get('permissions', {}).get('push'):
        raise ValueError('Expected main branch and push permission are not available')
    return identity


def source_allowed(relative, round_source=False):
    p = PurePosixPath(safe_relative(relative))
    if any(v in SKIP_DIRS for v in p.parts):
        return False, 'runtime data, environment, cache, or credential directory'
    if p.name in BAD_NAMES or p.name.startswith('.env') or p.suffix.lower() in {'.pem', '.p12', '.key'}:
        return False, 'credential/key filename'
    if p.suffix.lower() not in TEXT_EXT and p.name not in SPECIAL_NAMES:
        return False, 'binary/dataset/archive or non-allowlisted extension'
    if 'evidence' in p.parts:
        return False, 'local provenance can contain session-level dataset records; retained in AWS'
    if 'outputs' in p.parts and p.name not in OUTPUT_NAMES and 'charts' not in p.parts:
        return False, 'raw logs, row-level records, or unreviewed output; retained in AWS'
    return True, ''


def row_data(value):
    if isinstance(value, dict):
        if any(str(k).lower() in ROW_KEYS and isinstance(v, (list, dict)) for k, v in value.items()):
            return True
        return any(row_data(v) for v in value.values())
    if isinstance(value, list):
        return any(row_data(v) for v in value)
    return False


def secret_findings(data):
    return [{'rule': name, 'line': data[:m.start()].count(b'\n') + 1}
            for name, pattern in SECRET_RULES for m in re.finditer(pattern, data)]


def inspect_bytes(relative, data):
    if len(data) > MAX_FILE:
        return 'over 20 MiB file limit'
    if b'\0' in data:
        return 'binary content'
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return 'not UTF-8 source'
    flags = secret_findings(data)
    if flags:
        raise ValueError('Possible credential in ' + relative + ': ' + json.dumps(flags))
    suffix = Path(relative).suffix.lower()
    if suffix == '.json':
        value = json.loads(text)
        if row_data(value):
            return 'row-level dataset-derived JSON; retained in AWS'
    if suffix == '.ipynb':
        nb = json.loads(text)
        if not isinstance(nb.get('cells'), list) or nb.get('nbformat') != 4:
            raise ValueError('Notebook schema invalid: ' + relative)
        if nb.get('metadata', {}).get('widgets') or any(c.get('attachments') for c in nb['cells']):
            return 'embedded widgets/attachments require separate publication review'
        for cell in nb['cells']:
            for output in cell.get('outputs', []):
                if row_data(output.get('data', {}).get('application/json', {})):
                    return 'notebook contains row-level JSON output; retained in AWS'
    return None


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def choose_change(base, local, remote):
    """Three-way identity decision; never overwrite independent upstream edits."""
    if local == remote or local == base:
        return 'keep_remote'
    if remote == base:
        return 'apply_local'
    return 'conflict'


def tree_entries(repo, revision):
    entries = {}
    for record_bytes in git(repo, 'ls-tree', '-r', '-z', revision, binary=True).split(b'\0'):
        if not record_bytes:
            continue
        header, name = record_bytes.split(b'\t', 1)
        mode, kind, oid = header.decode().split()
        if kind == 'blob':
            entries[name.decode('utf-8')] = (mode, oid)
    return entries


def notebook_signature(data):
    return [{'cell_type': c['cell_type'], 'source': ''.join(c['source'])
             if isinstance(c['source'], list) else c['source']} for c in json.loads(data)['cells']]


def aggregate_results():
    rounds, warnings = [], []
    for root in sorted(HOME.glob('otto_feature_round[0-9][0-9]')):
        p = root / 'outputs/result.json'
        if not p.is_file() or p.is_symlink() or p.stat().st_size > MAX_FILE:
            continue
        arms = []
        for name, value in read(p).get('arms', {}).items():
            if not isinstance(value, dict) or 'weighted_recall_at_20' not in value:
                continue
            try:
                if not re.fullmatch('[A-Za-z0-9_ -]{1,100}', name):
                    raise ValueError('arm name')
                hits, den = value['hits'], value['denominators']
                if len(hits) != 3 or len(den) != 3 or any(type(x) is not int for x in hits + den):
                    raise ValueError('integer support')
                if any(d <= 0 or h < 0 or h > d for h, d in zip(hits, den)):
                    raise ValueError('support range')
                score = float(value['weighted_recall_at_20'])
                calculated = sum(w * h / d for w, h, d in zip((.1, .3, .6), hits, den))
                if not math.isfinite(score) or abs(calculated - score) > 1e-10:
                    raise ValueError('metric arithmetic')
                arms.append({'arm': name, 'weighted_recall_at_20': score, 'hits': hits, 'denominators': den})
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append({'round': root.name, 'arm': name, 'reason': str(exc)})
        if arms:
            rounds.append({'round': int(root.name[-2:]), 'arms': arms, 'source_sha256': file_digest(p)})
    return {'evaluation': 'Exploratory internal fitting folds; NOT leaderboard scores.',
            'limitations': 'Compare only within a matched round. Reused cohorts and adaptive hypothesis selection limit inference.',
            'provenance': 'Recorded owner-run AWS results; aggregate integer arithmetic checked, no model replay during publication.',
            'rounds': rounds, 'unpublished_metric_records': warnings}


def make_public_materials(clone, entries, exclusions, source_head, source_status):
    results = aggregate_results()
    atomic(clone / 'reports/manual_research/results.json', encoded(results))
    table = ['| Round | Representation | Weighted Recall@20 |', '| --- | --- | ---: |']
    for item in results['rounds']:
        for arm in item['arms']:
            table.append(f"| {item['round']:02d} | {arm['arm']} | {arm['weighted_recall_at_20']:.6f} |")
    text = ('# OTTO | Owner-run feature research\n\n' + results['evaluation'] + '\n\n'
            + results['limitations'] + '\n\n' + '\n'.join(table)
            + '\n\n## Source and evidence\n\nImplementation and executed notebooks are public in `research/manual/`. '
              'Dataset-derived provenance, raw events, per-session labels, fitted arrays, model weights, credentials, '
              'and environments stay in AWS. See the exclusions inventory. Source snapshots include negative results '
              'and pending work; a commit does not certify an experiment or imply improvement. '
              'Restore exact local-only dependencies before executing a historical round from a fresh checkout.\n\n'
              '[Source](../research/manual/README.md) · [Interactive report](../reports/manual_research/progress.html) · '
              '[Notebook](../reports/manual_research/summary.ipynb)\n\n'
              'The runtime checkout remains pinned while a separate publication clone incorporates the public updates. '
              'This is a reviewed source publication, not a byte-for-byte data mirror or a runtime migration.\n')
    atomic(clone / 'docs/MANUAL_RESEARCH.md', text.encode())
    atomic(clone / 'research/manual/README.md', (
        '# OTTO | Public manual research source\n\n'
        'Each `otto_feature_roundXX/` directory preserves the source and available saved notebook outputs from AWS. '
        'Canonical runtimes use the same folder names directly under the SageMaker home folder. '
        'Research dependencies, metadata hashes and immutable data remain necessary; source publication does not '
        'make a fresh checkout independently runnable without these artifacts.\n\n'
        'Nothing here hides the implementation in a separate repository. Dataset artifacts are excluded to respect '
        'competition/data restrictions and repository size. Existing licenses and third-party notices are retained.\n\n'
        '[Research results and limits](../../docs/MANUAL_RESEARCH.md)\n').encode())
    payload = json.dumps(results, allow_nan=False).replace('<', '\\u003c')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>OTTO | Research</title>'
            '<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:24px;line-height:1.6}</style>'
            '<h1>OTTO · Feature research</h1><p>Internal matched-fold evidence, not leaderboard results. '
            'Compare within each round. Rendering requires internet for Plotly.</p>'
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><div id="charts"></div><script>'
            'const data=' + payload + ';for(const r of data.rounds){let e=document.createElement("section");'
            'document.getElementById("charts").appendChild(e);Plotly.newPlot(e,[{type:"bar",'
            'x:r.arms.map(x=>x.arm),y:r.arms.map(x=>x.weighted_recall_at_20)}],'
            '{title:"Round "+r.round,yaxis:{title:"Weighted Recall@20"},margin:{b:180},height:520},'
            '{responsive:true,displaylogo:false});}</script></html>')
    atomic(clone / 'reports/manual_research/progress.html', page.encode())
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}, 'cells': [
        {'cell_type': 'markdown', 'id': 'intro', 'metadata': {}, 'source':
         '# OTTO | Recorded feature results\n\nPublic implementation and internal results. Supplied unexecuted; '
         'no outputs are fabricated. Run with the existing Plotly environment, save with Ctrl+S, reopen to check '
         'the plots. GitHub does not execute interactive JavaScript in its notebook preview.'},
        {'cell_type': 'code', 'id': 'setup', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source':
         'from pathlib import Path\nimport json\nimport plotly.graph_objects as go\nimport plotly.io as pio\n'
         'pio.renderers.default = "notebook+plotly_mimetype"\n'
         'results = json.loads(' + repr(json.dumps(results)) + ')\n'
         'out = Path("rendered_figures"); out.mkdir(exist_ok=True)\nprint(results["evaluation"])'},
        {'cell_type': 'code', 'id': 'charts', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source':
         'for item in results["rounds"]:\n'
         '    fig = go.Figure(go.Bar(x=[a["arm"] for a in item["arms"]], y=[a["weighted_recall_at_20"] for a in item["arms"]]))\n'
         '    fig.update_layout(title=f"Round {item[\'round\']}: matched internal scores", xaxis_title="Representation", yaxis_title="Weighted Recall@20", height=520)\n'
         '    fig.show()\n'
         '    fig.write_html(out / f"round{item[\'round\']:02d}.html", include_plotlyjs=True)\n'
         '    fig.write_json(out / f"round{item[\'round\']:02d}.json")\n'
         'print("Save this notebook; close and reopen it to verify the saved inline plots.")'}]}
    atomic(clone / 'reports/manual_research/summary.ipynb', encoded(nb))
    readme = clone / 'README.md'
    content = readme.read_text() if readme.is_file() else '# OTTO recommender system\n'
    begin, end = '<!-- OTTO_PUBLIC_RESEARCH -->', '<!-- /OTTO_PUBLIC_RESEARCH -->'
    section = (begin + '\n## Latest owner-run feature research\n\n'
               '[Public implementation and saved notebooks](research/manual/README.md) · '
               '[Measured results and limitations](docs/MANUAL_RESEARCH.md) · '
               '[Interactive report](reports/manual_research/progress.html)\n\n'
               'Complete research source is public; data and credentials are excluded. '
               'Pending and negative experiments remain explicit.\n' + end)
    if begin in content:
        if end not in content:
            raise ValueError('Existing README research block is malformed')
        content = re.sub(re.escape(begin) + '.*?' + re.escape(end), lambda _: section, content, flags=re.S)
    else:
        first, _, rest = content.partition('\n')
        content = first + '\n\n' + section + '\n\n' + rest
    atomic(readme, content.encode())
    manifest = {'version': VERSION, 'target': TARGET, 'source_head': source_head,
                'source_working_tree_status': source_status, 'files': entries,
                'excluded': exclusions, 'model_fits_during_publication': 0,
                'excluded_directory_names_not_walked': sorted(SKIP_DIRS),
                'note': 'Hashes refer to selected AWS bytes. Excluded directory contents were not enumerated. Generated documentation is additional. No Git history is rewritten.'}
    atomic(clone / 'reports/manual_research/source_manifest.json', encoded(manifest))


class PublicationNotPrepared(RuntimeError):
    """A failed preparation is not a publishable snapshot. No state is synthesized."""


def latest():
    pointer = STATE / 'latest.json'
    if not pointer.is_file():
        raise PublicationNotPrepared(
            'No completed public preparation exists. Nothing was committed, pushed, or merged by this command. '
            'Run self-test and prepare-public successfully; review REVIEW.md before publish-public.')
    if pointer.is_symlink():
        raise ValueError('Saved preparation pointer is a symlink; preserved for review')
    try:
        info = read(pointer)
    except (OSError, ValueError) as exc:
        raise ValueError('Saved preparation pointer is unreadable; preserved for review') from exc
    run_name = info.get('run') if isinstance(info, dict) else None
    if not isinstance(run_name, str) or not re.fullmatch(r'run-[A-Za-z0-9_-]+', run_name):
        raise ValueError('Invalid saved run; existing state is preserved')
    path = safe_path(STATE, run_name)
    plan_file = safe_path(path, 'plan.json')
    if not plan_file.is_file():
        raise PublicationNotPrepared(
            'The saved preparation has no completed plan.json. Nothing was committed, pushed, or merged '
            'by this command. Preserve the existing state and return the evidence ZIP.')
    try:
        plan = read(plan_file)
    except (OSError, ValueError) as exc:
        raise ValueError('Saved publication plan is unreadable; preserved for review') from exc
    if not isinstance(plan, dict):
        raise ValueError('Saved publication plan must be an object; preserved for review')
    return path, plan


def require_helper_tests():
    path = STATE / 'self_test.json'
    if not path.is_file() or path.is_symlink():
        raise ValueError('Run self-test on this exact script first; no valid local test receipt exists')
    receipt = read(path)
    if receipt.get('script_sha256') != file_digest(__file__) or receipt.get('status') != 'PUBLIC_SYNC_SELF_TESTS_PASSED':
        raise ValueError('Run self-test on this exact script first')


def helper_source_preflight():
    """Inspect the actual distributable bytes, not a hand-selected synthetic sample."""
    data = Path(__file__).read_bytes()
    reason = inspect_bytes('tools/otto_workspace_sync.py', data)
    if reason is not None:
        raise ValueError('The publication helper cannot publish itself: ' + reason)
    return {'script_sha256': digest(data), 'source_bytes': len(data)}


def prepare_public(install):
    require_helper_tests()
    source_receipt = helper_source_preflight()
    record('PUBLIC_HELPER_SOURCE_SCAN_PASSED', **source_receipt)
    if (STATE / 'latest.json').is_file():
        prior, plan = latest()
        if plan.get('status') in ('PREPARED_PUBLIC', 'COMMITTED', 'PUSHED', 'PR_CREATED'):
            verify_plan(prior, plan)
            record('EXISTING_PUBLIC_PLAN_PRESERVED', review=str(prior / 'REVIEW.md'),
                   next_step='Review and use publish-public, or finish-public after a PR exists.')
            return
    identity = github_cli(install)
    source = HOME / 'otto-recommender-system'
    if not (source / '.git').exists() or source.is_symlink():
        raise ValueError('Expected original AWS OTTO checkout is absent; no replacement is created')
    if shutil.disk_usage(HOME).free < 2 * 1024**3:
        raise ValueError('Publication needs 2 GiB free disk; nothing is deleted')
    head = git(source, 'rev-parse', 'HEAD').strip()
    status = git(source, 'status', '--porcelain=v1')
    clone = RUN / 'checkout'
    command(['git', '-c', 'credential.helper=', '-c',
             'credential.helper=!' + shlex.quote(GH) + ' auth git-credential',
             'clone', '--no-tags', 'https://github.com/' + TARGET + '.git', str(clone)], timeout=180)
    git(clone, 'fetch', '--no-tags', str(source), 'HEAD', timeout=60)
    base = git(clone, 'merge-base', head, 'origin/main').strip()
    remote_head = git(clone, 'rev-parse', 'origin/main').strip()
    base_entries, remote_entries = tree_entries(clone, base), tree_entries(clone, remote_head)
    local_names = [p for p in git(source, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0') if p]
    entries, exclusions, changes, conflicts = [], [], [], []
    total = 0
    snapshot_start = snapshot_beat = time.monotonic()

    def add(source_path, relative, destination, mode=None):
        nonlocal total, snapshot_beat
        allowed, why = source_allowed(relative)
        if not allowed:
            exclusions.append({'path': destination, 'reason': why}); return
        if source_path.is_symlink() or not source_path.is_file():
            exclusions.append({'path': destination, 'reason': 'symlink or non-file'}); return
        if mode is None:
            mode = '100755' if os.access(source_path, os.X_OK) else '100644'
        data = source_path.read_bytes()
        reason = inspect_bytes(destination, data)
        if reason:
            exclusions.append({'path': destination, 'reason': reason}); return
        total += len(data)
        if total > MAX_TOTAL:
            raise ValueError('Selected source exceeds 250 MiB; no push occurred')
        atomic(RUN / 'selected' / destination, data)
        target = safe_path(clone, destination)
        old_hash = file_digest(target) if target.is_file() else None
        if old_hash != digest(data):
            atomic(target, data)
            changes.append(destination)
        target.chmod(0o755 if mode == '100755' else 0o644)
        entries.append({'path': destination, 'sha256': digest(data), 'bytes': len(data)})
        if time.monotonic() - snapshot_beat >= 15:
            record('SOURCE_SNAPSHOT_PROGRESS', selected_files=len(entries), selected_bytes=total,
                   elapsed_seconds=round(time.monotonic() - snapshot_start, 1))
            snapshot_beat = time.monotonic()
        if file_digest(source_path) != digest(data):
            raise ValueError('Source changed while snapshotting. Stop concurrent notebook work.')

    for relative in sorted(set(local_names) | set(base_entries)):
        relative = safe_relative(relative)
        allowed, why = source_allowed(relative)
        if not allowed:
            exclusions.append({'path': relative, 'reason': why}); continue
        p = safe_path(source, relative)
        if p.exists() and not p.is_file():
            raise ValueError('Unsupported tracked source type: ' + relative)
        if p.is_file():
            b = p.read_bytes()
            mode = '100755' if os.access(p, os.X_OK) else '100644'
            local_value = (mode, git_blob(b))
        else:
            local_value = None
        decision = choose_change(base_entries.get(relative), local_value, remote_entries.get(relative))
        if decision == 'conflict':
            conflicts.append(relative); continue
        if decision == 'apply_local':
            if local_value is None:
                dest = safe_path(clone, relative)
                if dest.is_file():
                    dest.unlink(); changes.append(relative)
            else:
                add(p, relative, relative, local_value[0])
    if conflicts:
        atomic(RUN / 'conflicts.json', encoded(conflicts))
        raise ValueError('Both AWS and GitHub changed these paths: ' + ', '.join(conflicts[:10]) + '. No push; preserved for reconciliation.')
    for root in sorted(HOME.glob('otto_feature_round[0-9][0-9]')):
        if not root.is_dir() or root.is_symlink():
            continue
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                p = Path(directory) / name
                relative = p.relative_to(root).as_posix()
                add(p, relative, 'research/manual/' + root.name + '/' + relative)
    make_public_materials(clone, entries, exclusions, head, status)
    # Include the public-only helper as auditable source, not the prior private-target helper.
    atomic(clone / 'tools/otto_workspace_sync.py', Path(__file__).read_bytes())
    atomic(clone / PUBLIC_CHECK_NOTES, public_check_notes())
    git(clone, 'config', '--local', 'user.name', OWNER)
    git(clone, 'config', '--local', 'user.email', f"{identity['id']}+{OWNER}@users.noreply.github.com")
    git(clone, 'diff', '--check')
    # Only files changed in this isolated clone are approved; record exact bytes before staging.
    modified = [p for p in git(clone, 'diff', '--no-renames', '--name-only', '-z').split('\0') if p]
    untracked = [p for p in git(clone, 'ls-files', '--others', '--exclude-standard', '-z').split('\0') if p]
    available_paths = set(modified + untracked) | set(remote_entries)
    omitted = [row['path'] for row in entries if row['path'] not in available_paths]
    if omitted:
        atomic(RUN / 'ignored_selected_files.json', encoded(omitted))
        raise ValueError('Selected source is ignored by repository rules; nothing pushed. Review ignored_selected_files.json.')
    approved = []
    for relative in sorted(set(modified + untracked)):
        path = safe_path(clone, relative)
        if path.is_file():
            reason = inspect_bytes(relative, path.read_bytes())
            if reason:
                raise ValueError('Generated or staged content needs review: ' + relative + ' (' + reason + ')')
        approved.append({'path': relative, 'sha256': file_digest(path) if path.is_file() else None})
    if git(source, 'rev-parse', 'HEAD').strip() != head or git(source, 'status', '--porcelain=v1') != status:
        raise ValueError('Original AWS source changed during publication preparation; snapshot preserved, nothing pushed')
    plan = {'version': VERSION, 'target': TARGET, 'status': 'PREPARED_PUBLIC',
            'source_head': head, 'source_status': status, 'base': remote_head,
            'files': approved, 'selected_files': len(entries), 'selected_bytes': total,
            'excluded_files': len(exclusions), 'checkout': 'checkout',
            'script_sha256': file_digest(__file__), 'source_checkout_changed': False}
    atomic(RUN / 'plan.json', encoded(plan))
    atomic(RUN / 'selected_files.json', encoded(entries))
    atomic(RUN / 'exclusions.json', encoded(exclusions))
    review = ('# PUBLIC publication review\n\nTarget: **' + TARGET + '**. No other repository.\n\n'
              f"{len(approved)} changed files; {len(entries)} selected AWS files; {total / 1024**2:.2f} MiB selected.\n\n"
              'Review checkout/ in JupyterLab, selected_files.json, exclusions.json, and this diff before publishing. '
              'The credential scan is heuristic, not a guarantee. Do not approve notebook outputs containing raw/session-level data. '
              'The original working tree is untouched. Commit/push does not run experiments.\n\n'
              '```bash\ncd ' + shlex.quote(str(clone)) + '\ngit status --short\ngit diff --stat\ngit diff -- README.md docs/MANUAL_RESEARCH.md\n```\n\n'
              'After review, run:\n```bash\n/opt/conda/bin/python "$HOME/otto_workspace_sync.py" publish-public --approve-public\n```\n')
    atomic(RUN / 'REVIEW.md', review.encode())
    atomic(STATE / 'latest.json', encoded({'run': RUN.name}))
    record('PUBLIC_PREPARED_NOT_PUSHED', target=TARGET, review=str(RUN / 'REVIEW.md'),
           files=len(approved), selected_files=len(entries), excluded=len(exclusions), source_checkout_changed=False)


def verify_plan(run, plan):
    if plan.get('target') != TARGET:
        raise ValueError('Target differs from the reviewed plan')
    if plan.get('script_sha256') != file_digest(__file__):
        raise ValueError('Controller differs from saved plan. For the diagnosed uncommitted snapshot, run self-test then resume-public; do not rebuild it.')
    clone = run / plan['checkout']
    for row in plan['files']:
        p = safe_path(clone, row['path'])
        current = file_digest(p) if p.is_file() else None
        if current != row['sha256']:
            raise ValueError('File changed after preparation: ' + row['path'])
    return clone


# The legacy plan and archival files below were identified in the owner's return
# bundle. These are exact-byte archival exceptions, not filename exemptions.
LEGACY_HELPER_SHA = '22d549b6b0561b05d2b8dbbba4292e4d2f6220afe7a1e84fd6c4d30061e9dbf6'
LEGACY_PLAN_SHA = '407810f491d21c3c0f2e901879bc2496bbdd257bf291c0a2797c8bddd804d9fa'
EOF_SNAPSHOTS = {'research/manual/otto_feature_round04/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round05/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round06/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round08/source_contract.py': '2cd4d6a82e6cd71381bb083ba92eaef881e8b61c68536448821121766a18a8a0', 'research/manual/otto_feature_round08/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round09/source_contract.py': '2cd4d6a82e6cd71381bb083ba92eaef881e8b61c68536448821121766a18a8a0', 'research/manual/otto_feature_round09/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round10/source_contract.py': '2cd4d6a82e6cd71381bb083ba92eaef881e8b61c68536448821121766a18a8a0', 'research/manual/otto_feature_round10/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round11/source_contract.py': 'cbf7de4a4c8c3f7cc7b375562fe634a692ce72aff558fee59735dba8da5d0343', 'research/manual/otto_feature_round11/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91', 'research/manual/otto_feature_round12/source_contract.py': 'cbf7de4a4c8c3f7cc7b375562fe634a692ce72aff558fee59735dba8da5d0343', 'research/manual/otto_feature_round12/tests/test_metric_contracts.py': '9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91'}
PUBLIC_CHECK_NOTES = 'docs/PUBLICATION_CHECKS.md'


def publication_delta(clone):
    """HEAD -> worktree includes staged changes; index -> worktree does not."""
    tracked = git(clone, 'diff', 'HEAD', '--no-renames', '--name-only', '-z')
    untracked = git(clone, 'ls-files', '--others', '--exclude-standard', '-z')
    return {safe_relative(p) for p in (tracked + untracked).split('\0') if p}


def validate_snapshot(run, plan, helper_sha=None):
    expected_helper = helper_sha or file_digest(__file__)
    if plan.get('target') != TARGET or plan.get('script_sha256') != expected_helper:
        raise ValueError('Target or controller hash does not match the saved plan')
    if plan.get('checkout') != 'checkout':
        raise ValueError('Unexpected publication checkout location')
    clone = safe_path(run, 'checkout')
    if not (clone / '.git').is_dir() or (clone / '.git').is_symlink():
        raise ValueError('The original publication checkout must remain present')
    paths = [r['path'] for r in plan['files']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in saved plan')
    for row in plan['files']:
        path = safe_path(clone, row['path'])
        observed = file_digest(path) if path.is_file() else None
        if observed != row['sha256']:
            raise ValueError('File changed after preparation: ' + row['path'])
        if path.is_file():
            reason = inspect_bytes(row['path'], path.read_bytes())
            if reason:
                raise ValueError('Publication inspection failed: ' + row['path'] + ' (' + reason + ')')
    return clone


def parse_diff_check(data):
    """Retain locations and explanations, never echo the added source lines."""
    diagnostics, other = [], []
    for line in data.decode('utf-8', 'replace').splitlines():
        if not line.strip() or line.startswith('+'):
            continue
        match = re.fullmatch(r'(.+):(\d+): (.+)', line)
        if match:
            diagnostics.append({'path': safe_relative(match[1]),
                                'line': int(match[2]), 'issue': match[3]})
        else:
            other.append(redact(line))
    return diagnostics, other


def archival_eof_notice(row, clone, planned):
    relative = row['path']
    expected = EOF_SNAPSHOTS.get(relative)
    if row['issue'] != 'new blank line at EOF.' or not expected:
        return False
    if planned.get(relative) != expected:
        return False
    path = safe_path(clone, relative)
    data = path.read_bytes()
    if digest(data) != expected:
        return False
    lines = data.splitlines()
    start = row['line'] - 1
    return 0 <= start < len(lines) and all(line == b'' for line in lines[start:])


def checked_public_diff(clone, rows):
    """Never suppress a return code without validating every Git diagnostic.

    Only exact, hash-bound, empty EOF lines in frozen archived snapshots may
    be notices. All other whitespace/conflict diagnostics remain blocking.
    No Git configuration, hook, attributes, CI check or source file is changed.
    """
    result = command(['git', '--no-pager', '--literal-pathspecs', '-C', str(clone),
                      '-c', 'color.ui=false', 'diff', '--cached', '--no-ext-diff',
                      '--no-textconv', '--check'], timeout=60, allowed=(0, 1, 2),
                     result_object=True)
    diagnostics, other = parse_diff_check(result.stdout)
    planned = {r['path']: r['sha256'] for r in rows}
    notices, blocked = [], []
    for row in diagnostics:
        if archival_eof_notice(row, clone, planned):
            notices.append(row)
        else:
            blocked.append(row)
    receipt = {'command': 'git diff --cached --no-ext-diff --no-textconv --check',
               'exit_code': result.returncode, 'archival_eof_notices': notices,
               'blocking_diagnostics': blocked, 'unparsed_messages': other,
               'stderr': redact(result.stderr.decode('utf-8', 'replace')),
               'source_bytes_rewritten': False, 'github_checks_bypassed': False}
    if RUN:
        atomic(RUN / 'git_diagnostics.json', encoded(receipt))
    unexplained = result.returncode != 0 and not diagnostics
    if blocked or other or unexplained or (result.returncode and result.stderr.strip()):
        locations = '; '.join(f"{x['path']}:{x['line']}: {x['issue']}" for x in blocked[:8])
        raise ValueError('PUBLIC_DIFF_CHECK_BLOCKED: ' +
                         (locations or 'Git reported an unexpected diagnostic; see git_diagnostics.json'))
    if result.returncode == 0 and diagnostics:
        raise ValueError('Git diagnostics conflict with its success return code')
    if notices:
        record('PUBLIC_ARCHIVE_FORMATTING_NOTICES', files=len(notices),
               reason='Exact frozen source bytes retain empty EOF lines; see docs/PUBLICATION_CHECKS.md')
    else:
        record('PUBLIC_STAGED_DIFF_CHECK_PASSED')
    return receipt


def stage_reviewed_files(run, clone, rows):
    expected = {r['path'] for r in rows}
    if publication_delta(clone) != expected:
        raise ValueError('Publication files differ from the reviewed plan, including staged changes')
    indexed = {p for p in git(clone, 'diff', '--cached', '--no-renames', '--name-only', '-z').split('\0') if p}
    if not indexed <= expected:
        raise ValueError('An unreviewed file is already staged; index preserved')
    pathspec = run / 'paths.nul'
    atomic(pathspec, b''.join(safe_relative(r['path']).encode() + b'\0' for r in rows))
    git(clone, '--literal-pathspecs', 'add', '--all', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul')
    staged = {p for p in git(clone, 'diff', '--cached', '--no-renames', '--name-only', '-z').split('\0') if p}
    if staged != expected:
        raise ValueError('Staged file set does not equal the reviewed set')
    tree = git(clone, 'write-tree').strip()
    index_entries = tree_entries(clone, tree)
    for row in rows:
        if row['sha256'] is not None:
            path = safe_path(clone, row['path'])
            if file_digest(path) != row['sha256'] or index_entries.get(row['path'], ('', ''))[1] != git_blob(path.read_bytes()):
                raise ValueError('Staging/filter changed reviewed bytes: ' + row['path'])
        elif row['path'] in index_entries:
            raise ValueError('Reviewed deletion remains in the staged tree: ' + row['path'])
    checked_public_diff(clone, rows)
    return tree


def public_check_notes():
    lines = ['# Public snapshot checks', '',
             'This repository preserves historical source bytes and their recorded checksums.',
             'The local publisher records an empty-end-of-file notice only when both the exact',
             'archived path and whole-file SHA-256 match the table below. It does not rewrite',
             'historical source or pretend its hashes were produced by a reformatted copy.', '',
             'All other whitespace errors, conflict markers, unexpected diagnostics, credential',
             'findings and byte mismatches remain blocking. Git configuration, hooks and GitHub',
             'checks are not disabled. This is not evidence of a new model result.', '',
             '| Frozen source path | SHA-256 |', '|---|---|']
    lines += [f'| `{path}` | `{sha}` |' for path, sha in sorted(EOF_SNAPSHOTS.items())]
    return ('\n'.join(lines) + '\n').encode()


def resume_public():
    """Local recovery only: reuse the successful prepared snapshot, no network."""
    require_helper_tests()
    helper_source_preflight()
    run, plan = latest()
    if plan.get('status') != 'PREPARED_PUBLIC':
        raise ValueError('Recovery accepts only an uncommitted prepared plan; later states are preserved')
    clone = safe_path(run, 'checkout')
    if os.environ.get('GIT_INDEX_FILE') or os.environ.get('GIT_DIR') or os.environ.get('GIT_WORK_TREE'):
        raise ValueError('Unexpected Git path overrides in environment; no mutation performed')
    if git(clone, 'rev-parse', 'HEAD').strip() != plan.get('base'):
        raise ValueError('Publication HEAD moved; do not overwrite or recommit it')
    origin = git(clone, 'remote', 'get-url', 'origin').strip()
    if origin not in ('https://github.com/' + TARGET + '.git', 'git@github.com:' + TARGET + '.git'):
        raise ValueError('Publication origin differs from the single approved public target')
    if git(clone, 'ls-files', '--unmerged', '-z'):
        raise ValueError('Unmerged index entries preserved; resolve before publication')
    backup = safe_path(run, 'publisher_recovery')
    intent_path = safe_path(backup, 'intent.json')
    current_sha = file_digest(__file__)
    if plan.get('script_sha256') == current_sha:
        clone = validate_snapshot(run, plan)
    else:
        if intent_path.is_file():
            intent = read(intent_path)
            if intent.get('new_helper_sha256') != current_sha or intent.get('legacy_plan_sha256') != LEGACY_PLAN_SHA:
                raise ValueError('Recovery journal differs from this exact helper; preserved')
            old_bytes = safe_path(backup, 'plan.before.json').read_bytes()
        else:
            old_bytes = safe_path(run, 'plan.json').read_bytes()
        if digest(old_bytes) != LEGACY_PLAN_SHA or plan.get('script_sha256') != LEGACY_HELPER_SHA:
            raise ValueError('Only the exact returned prepared plan can be migrated without rebuilding')
        old = json.loads(old_bytes)
        old_files = {r['path']: r['sha256'] for r in old['files']}
        tool = 'tools/otto_workspace_sync.py'
        if old_files.get(tool) != LEGACY_HELPER_SHA or PUBLIC_CHECK_NOTES in old_files:
            raise ValueError('Legacy helper/notes differ from the recorded recovery contract')
        new_tool = Path(__file__).read_bytes()
        note_bytes = public_check_notes()
        proposals = {tool: new_tool, PUBLIC_CHECK_NOTES: note_bytes}
        delta = publication_delta(clone)
        if not set(old_files) <= delta or not delta <= (set(old_files) | {PUBLIC_CHECK_NOTES}):
            raise ValueError('Unexpected pending change before recovery; nothing overwritten')
        for relative, sha in old_files.items():
            path = safe_path(clone, relative)
            observed = file_digest(path) if path.is_file() else None
            permitted = {sha}
            if intent_path.is_file() and relative == tool:
                permitted.add(current_sha)
            if observed not in permitted:
                raise ValueError('Prepared source changed; preserved: ' + relative)
            if path.is_file():
                reason = inspect_bytes(relative, path.read_bytes())
                if reason:
                    raise ValueError('Source publication inspection failed: ' + relative)
        note_path = safe_path(clone, PUBLIC_CHECK_NOTES)
        if note_path.exists() and (not intent_path.is_file() or note_path.read_bytes() != note_bytes):
            raise ValueError('Existing publication notes differ; not overwritten')
        if not intent_path.is_file():
            backup.mkdir(parents=True, exist_ok=True)
            for relative, data in [('plan.before.json', old_bytes),
                                   ('helper.before.py', safe_path(clone, tool).read_bytes())]:
                dest = safe_path(backup, relative)
                if dest.exists():
                    if dest.read_bytes() != data:
                        raise ValueError('Earlier recovery backup differs; preserved')
                else:
                    atomic(dest, data)
            index = clone / '.git/index'
            if index.is_file() and not (backup / 'index.before').exists():
                atomic(backup / 'index.before', index.read_bytes())
            atomic(intent_path, encoded({'legacy_plan_sha256': LEGACY_PLAN_SHA,
                                         'new_helper_sha256': current_sha,
                                         'new_notes_sha256': digest(note_bytes)}))
        for relative, data in proposals.items():
            if inspect_bytes(relative, data):
                raise ValueError('Generated recovery content did not pass publication inspection')
            atomic(safe_path(clone, relative), data)
        new = json.loads(old_bytes)
        for row in new['files']:
            if row['path'] == tool:
                row['sha256'] = current_sha
        new['files'].append({'path': PUBLIC_CHECK_NOTES, 'sha256': digest(note_bytes)})
        new['files'].sort(key=lambda row: row['path'])
        new.update(version=VERSION, script_sha256=current_sha,
                   publisher_recovery={'old_plan_sha256': LEGACY_PLAN_SHA,
                                       'old_helper_sha256': LEGACY_HELPER_SHA,
                                       'selected_research_source_bytes_preserved': True})
        atomic(run / 'plan.json', encoded(new))
        plan = new
        clone = validate_snapshot(run, plan)
    tree = stage_reviewed_files(run, clone, plan['files'])
    review = ('# Resume the existing public snapshot\n\n'
              'No commit, push, authentication, clone, source rescan or experiment was performed by resume-public.\n\n'
              'Target: **' + TARGET + '**. Original prepared run and research files are preserved.\n\n'
              'The controller copy was updated and docs/PUBLICATION_CHECKS.md was added. '
              'Empty EOF notices require exact historical file hashes; all other diagnostics remain blocking.\n\n'
              'Review the staged changes:\n\n```bash\ncd ' + shlex.quote(str(clone)) +
              '\ngit diff --cached --stat\ngit diff --cached -- docs/PUBLICATION_CHECKS.md\n```\n\n'
              'After review only:\n\n```bash\n/opt/conda/bin/python "$HOME/otto_workspace_sync.py" publish-public --approve-public\n```\n')
    atomic(run / 'RESUME_REVIEW.md', review.encode())
    summary = {'status': 'PUBLIC_RESUME_READY_NOT_PUSHED', 'prepared_run': run.name,
               'files': len(plan['files']), 'tree': tree, 'target': TARGET,
               'script_sha256': current_sha, 'legacy_plan_sha256': LEGACY_PLAN_SHA,
               'network_calls': 0, 'project_model_fits': 0,
               'original_execution_checkout_changed': False,
               'selected_research_source_bytes_preserved': True}
    atomic(RUN / 'publisher_recovery.json', encoded(summary))
    record('PUBLIC_RESUME_READY_NOT_PUSHED', review=str(run / 'RESUME_REVIEW.md'),
           files=len(plan['files']), tree=tree, network_calls=0,
           source_checkout_changed=False)


def publish_public(approved, install):
    if not approved:
        raise ValueError('Read REVIEW.md first, then explicitly use --approve-public.')
    run, plan = latest()
    require_helper_tests()
    clone = verify_plan(run, plan)
    github_cli(install)
    if plan['status'] in ('PR_CREATED', 'MERGED'):
        record('EXISTING_PUBLIC_PR_PRESERVED', pr=plan.get('pr'), next_step='Use finish-public; no new push is needed.')
        return
    if not plan['files']:
        record('NO_PUBLIC_CHANGES', target=TARGET); return
    if plan['status'] == 'PREPARED_PUBLIC':
        tree = stage_reviewed_files(run, clone, plan['files'])
        branch = 'sync/aws-public-' + tree[:12]
        branches = git(clone, 'for-each-ref', '--format=%(refname:short)', 'refs/heads/').splitlines()
        if branch in branches:
            if git(clone, 'rev-parse', branch).strip() != plan['base']:
                raise ValueError('Publication branch already moved; preserve it and review before retrying')
            git(clone, 'switch', branch)
        else:
            git(clone, 'switch', '-c', branch)
        git(clone, 'commit', '-m', 'Publish owner-run AWS feature research and notebook evidence')
        plan.update(status='COMMITTED', branch=branch, head=git(clone, 'rev-parse', 'HEAD').strip())
        atomic(run / 'plan.json', encoded(plan))
        record('PUBLIC_COMMIT_CREATED', head=plan['head'], files=len(plan['files']))
    if plan['status'] == 'COMMITTED':
        if api('repos/' + TARGET).get('private') is not False:
            raise ValueError('Target visibility changed; no push')
        git(clone, 'push', 'origin', plan['head'] + ':refs/heads/' + plan['branch'], timeout=180)
        remote = api('repos/' + TARGET + '/git/ref/heads/' + plan['branch'])['object']['sha']
        if remote != plan['head']:
            raise ValueError('Remote commit differs from pushed commit')
        plan['status'] = 'PUSHED'
        atomic(run / 'plan.json', encoded(plan))
        record('PUBLIC_PUSH_VERIFIED', head=plan['head'], target=TARGET)
    pulls = api('repos/' + TARGET + '/pulls?state=open&head=' + OWNER + ':' + plan['branch'] + '&base=main')
    if len(pulls) > 1:
        raise ValueError('More than one matching PR; inspect before continuing')
    if pulls:
        pr = pulls[0]
    else:
        pr = api('repos/' + TARGET + '/pulls', 'POST', {
            'title': 'Publish AWS feature research, saved notebooks, and measured results',
            'head': plan['branch'], 'base': 'main',
            'body': 'Owner-approved public source snapshot from AWS. Includes implementation, tests, saved notebooks and aggregate evidence. '
                    'Excludes credentials, raw/session-level data, models and environments. Original runtime checkout is unchanged. '
                    'Negative and pending experiments remain explicit. No training was done by the publisher. '
                    'Please review Files changed and all checks; no force push, admin bypass or visibility change is used.'})
    plan.update(status='PR_CREATED', pr=pr['number'], pr_url=pr['html_url'])
    atomic(run / 'plan.json', encoded(plan))
    record('PUBLIC_PR_READY_FOR_CHECKS', pr=pr['number'], url=pr['html_url'], head=plan['head'],
           next_step='Inspect the PR checks and Files changed, then run finish-public. No experiment rerun.')


def check_state(checks, statuses):
    # Latest context takes precedence over superseded attempts.
    latest_checks = {}
    for row in checks:
        key = (row.get('app', {}).get('id'), row['name'])
        if row.get('id', 0) > latest_checks.get(key, {}).get('id', -1):
            latest_checks[key] = row
    latest_status = {}
    for row in statuses:
        key = row['context']
        if row.get('id', 0) > latest_status.get(key, {}).get('id', -1):
            latest_status[key] = row
    bad, pending = [], []
    for row in latest_checks.values():
        if row.get('status') != 'completed':
            pending.append(row['name'])
        elif row.get('conclusion') not in ('success', 'neutral', 'skipped'):
            bad.append(row['name'])
    for row in latest_status.values():
        if row.get('state') in ('error', 'failure'):
            bad.append(row['context'])
        elif row.get('state') != 'success':
            pending.append(row['context'])
    return bad, pending, len(latest_checks) + len(latest_status)


def finish_public(install):
    run, plan = latest()
    verify_plan(run, plan)
    github_cli(install)
    if not plan.get('pr'):
        raise ValueError('No saved public PR; run reviewed publish-public first')
    endpoint = 'repos/' + TARGET
    pr = api(endpoint + '/pulls/' + str(plan['pr']))
    if pr['head']['sha'] != plan['head'] or pr['base']['ref'] != 'main':
        raise ValueError('PR head/base changed; do not merge another revision')
    if not pr.get('merged'):
        checks = api(endpoint + '/commits/' + plan['head'] + '/check-runs?per_page=100&filter=latest')
        statuses = api(endpoint + '/commits/' + plan['head'] + '/status?per_page=100')
        if checks.get('total_count', 0) > 100 or statuses.get('total_count', 0) > 100:
            raise ValueError('More than 100 checks; inspect full list manually before merge')
        bad, pending, count = check_state(checks.get('check_runs', []), statuses.get('statuses', []))
        if bad:
            record('PUBLIC_MERGE_BLOCKED', failed_checks=bad, pr=plan['pr']); return
        if pending or count == 0:
            record('PUBLIC_MERGE_PENDING', pending_checks=pending, reported_checks=count, pr=plan['pr'],
                   note='No waiting loop or automatic retry. Check GitHub and run finish-public again after checks finish.')
            return
        command([GH, 'pr', 'merge', str(plan['pr']), '--repo', TARGET, '--merge',
                 '--match-head-commit', plan['head']], timeout=60)
        pr = api(endpoint + '/pulls/' + str(plan['pr']))
    if not pr.get('merged'):
        record('PUBLIC_MERGE_NOT_CONFIRMED', pr=plan['pr']); return
    merge = pr['merge_commit_sha']
    comparison = api(endpoint + '/compare/' + merge + '...main')
    if comparison.get('status') not in ('identical', 'ahead'):
        raise ValueError('Merged commit is not verified in main')
    plan.update(status='MERGED', merge_commit=merge)
    atomic(run / 'plan.json', encoded(plan))
    record('PUBLIC_MAIN_UPDATED_VERIFIED', target=TARGET, pr=plan['pr'], merge_commit=merge,
           source_checkout_changed=False, note='Public source is synchronized; original runtime remains pinned.')


def diagnosed_failures(root):
    workers, failures = [], []
    for p in (root / 'outputs/runs').glob('prepare-*.json'):
        row = read(p)
        if (row.get('error') == "ModuleNotFoundError: No module named 'intent_features'"
                and row.get('exit_code') == 2 and row.get('completed') == 0 and row.get('total') == 0):
            workers.append((p, row))
    for p in (root / 'outputs/runs').glob('launcher-*.json'):
        row = read(p)
        if row.get('exit_code') == 0 and not row.get('reason'):
            continue
        if row.get('phase') != 'prepare' or row.get('exit_code') != 2 or row.get('reason'):
            raise ValueError('Unrelated failure preserved: ' + p.name)
        stamp = dt.datetime.fromisoformat(row['utc'])
        matches = [w for _, w in workers if abs((stamp - dt.datetime.fromisoformat(w['utc'])).total_seconds()) < 8]
        if len(matches) != 1:
            raise ValueError('Failure cannot be matched to the known import error: ' + p.name)
        failures.append(p)
    return failures


def repair():
    """Run only after publication evidence is reviewed. No training or data scans."""
    archive = HOME / ARCHIVE
    if not archive.is_file():
        archive = Path(__file__).resolve().parent / ARCHIVE
    if not archive.is_file() or file_digest(archive) != ARCHIVE_SHA:
        raise ValueError('Upload this delivery\'s exact ' + ARCHIVE + '; identity does not match')
    writes, failures, changed_rounds = [], {}, set()
    with zipfile.ZipFile(archive) as z:
        contract = json.loads(z.read('patch_contract.json'))
        for folder, after in contract['after'].items():
            if folder not in ('otto_feature_round11', 'otto_feature_round12'):
                raise ValueError('Unexpected patch target')
            root = HOME / folder
            signatures = json.loads(z.read(folder + '/notebook_sources.json'))
            for relative, expected in after.items():
                data = z.read(folder + '/' + safe_relative(relative))
                if digest(data) != expected:
                    raise ValueError('Archive checksum mismatch')
                dest = safe_path(root, relative)
                if dest.exists():
                    old = dest.read_bytes()
                    if relative.endswith('.ipynb'):
                        if notebook_signature(old) != signatures[relative]:
                            raise ValueError('Edited notebook source preserved: ' + relative)
                        continue
                    if digest(old) == expected:
                        continue
                    permitted = contract['before'][folder].get(relative, [])
                    if isinstance(permitted, str):
                        permitted = [permitted]
                    if digest(old) not in permitted:
                        raise ValueError('Unrecognized source edit preserved: ' + relative)
                    if relative.endswith('.py'):
                        changed_rounds.add(folder)
                writes.append((dest, data))
            if folder in changed_rounds:
                for rel in ('representation_inputs/manifest.json', 'representations/manifest.json',
                            'feature_contract.json', 'result.json'):
                    if (root / 'outputs' / rel).exists():
                        raise ValueError('Real data-stage artifacts already exist; do not patch their identity: ' + folder)
            failures[folder] = diagnosed_failures(root)
        for p, data in writes:
            if p.exists():
                atomic(RUN / 'original_sources' / p.relative_to(HOME), p.read_bytes())
            atomic(p, data)
    interpreter = HOME / 'otto-recommender-system/.venv/bin/python'
    if not interpreter.is_file():
        raise ValueError('Existing project interpreter missing; do not reinstall it')
    for folder in contract['after']:
        root = HOME / folder
        result = command([str(interpreter), '-B', str(root / 'check_imports.py')], cwd=root, timeout=30)
        if 'LEGACY_IMPORT_PREFLIGHT_PASSED' not in result:
            raise ValueError('Import integration check did not pass')
        print(result, flush=True)
        evidence = root / 'outputs/recovery' / VERSION
        for p in failures[folder]:
            destination = evidence / p.name
            if destination.exists():
                raise ValueError('Existing recovery evidence preserved; no overwrite')
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, destination)
        if folder in changed_rounds:
            for p in (root / 'outputs/runs').glob('launcher-tests-*.json'):
                destination = evidence / p.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError('Conflicting test evidence preserved')
                os.replace(p, destination)
        record('IMPORT_REPAIR_PREFLIGHT_PASSED', round=folder, model_fits=0,
               next_step='Save evidence. Full notebook tests remain required before any data stage.')


def bundle():
    target = HOME / 'otto_public_sync_return.zip'
    temp = target.with_suffix('.assembling')
    rows = []
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        folders = set(sorted(STATE.glob('run-*'))[-8:])
        try:
            folders.add(latest()[0])
        except (OSError, ValueError, PublicationNotPrepared):
            pass
        for folder in sorted(folders):
            for name in ('receipt.json', 'error.json', 'plan.json', 'exclusions.json', 'conflicts.json', 'ignored_selected_files.json',
                         'command_failure.json', 'git_diagnostics.json', 'publisher_recovery.json',
                         'RESUME_REVIEW.md'):
                p = folder / name
                if p.is_file() and not p.is_symlink() and p.stat().st_size < 3 * 1024**2:
                    data = p.read_bytes()
                    if secret_findings(data):
                        continue
                    relative = folder.name + '/' + name
                    z.writestr(relative, data)
                    rows.append({'path': relative, 'sha256': digest(data)})
        test_receipt = STATE / 'self_test.json'
        if test_receipt.is_file() and not test_receipt.is_symlink():
            data = test_receipt.read_bytes()
            if not secret_findings(data):
                z.writestr('self_test.json', data)
                rows.append({'path': 'self_test.json', 'sha256': digest(data)})
        z.writestr('RETURN_MANIFEST.json', encoded(rows))
    os.replace(temp, target)
    print('RETURN_FILE: ' + str(target), flush=True)


class PublicationContracts(unittest.TestCase):
    """Owner-executed helper tests; no project model, network or Git operations."""
    def test_single_target(self): self.assertEqual(TARGET, OWNER + '/otto-recommender-system')
    def test_absolute_path(self):
        with self.assertRaises(ValueError): safe_relative('/etc/passwd')
    def test_parent_path(self):
        with self.assertRaises(ValueError): safe_relative('x/../a')
    def test_newline_path(self):
        with self.assertRaises(ValueError): safe_relative('a\nb')
    def test_backslash_path(self):
        with self.assertRaises(ValueError): safe_relative('x\\a')
    def test_regular_path(self): self.assertEqual(safe_relative('research/a.py'), 'research/a.py')
    def test_data_directory_excluded(self): self.assertFalse(source_allowed('data/sample.json')[0])
    def test_environment_excluded(self): self.assertFalse(source_allowed('.venv/a.py')[0])
    def test_credentials_excluded(self): self.assertFalse(source_allowed('credentials.json')[0])
    def test_hidden_dotenv_excluded(self): self.assertFalse(source_allowed('.env.local')[0])
    def test_private_key_excluded(self): self.assertFalse(source_allowed('x/key.pem')[0])
    def test_source_included(self): self.assertTrue(source_allowed('latent_features.py')[0])
    def test_notebook_included(self): self.assertTrue(source_allowed('01_build.ipynb')[0])
    def test_weights_excluded(self): self.assertFalse(source_allowed('weights.npz')[0])
    def test_evidence_excluded(self): self.assertFalse(source_allowed('evidence/raw.json')[0])
    def test_result_included(self): self.assertTrue(source_allowed('outputs/result.json')[0])
    def test_unreviewed_log_excluded(self): self.assertFalse(source_allowed('outputs/screen.log')[0])
    def test_token_detection(self): self.assertTrue(secret_findings(('ghp_' + 'A' * 35).encode()))
    def test_safe_source_not_token(self): self.assertFalse(secret_findings(b'PASSWORD = os.environ.get("PW")'))
    def test_signed_url_detection(self):
        fixture = b'https://example.invalid/?' + b'X-Amz-' + b'Signature' + b'=' + b'abc'
        self.assertTrue(secret_findings(fixture))
    def test_row_ids_detected(self): self.assertTrue(row_data({'nested': {'ids': [1, 2]}}))
    def test_scalar_session_count_allowed(self): self.assertFalse(row_data({'sessions': 4096}))
    def test_aggregate_allowed(self): self.assertIsNone(inspect_bytes('x.json', b'{"hits": [1,2,3]}'))
    def test_dataset_json_excluded(self): self.assertIsNotNone(inspect_bytes('x.json', b'{"ids": [1]}'))
    def test_binary_rejected(self): self.assertIsNotNone(inspect_bytes('a.py', b'\0x'))
    def test_only_upstream_change_preserved(self): self.assertEqual(choose_change('a', 'a', 'b'), 'keep_remote')
    def test_only_local_change_applied(self): self.assertEqual(choose_change('a', 'b', 'a'), 'apply_local')
    def test_independent_edits_conflict(self): self.assertEqual(choose_change('a', 'b', 'c'), 'conflict')
    def test_same_edits_do_not_conflict(self): self.assertEqual(choose_change('a', 'b', 'b'), 'keep_remote')
    def test_local_deletion(self): self.assertEqual(choose_change('a', None, 'a'), 'apply_local')
    def test_remote_deletion_preserved(self): self.assertEqual(choose_change('a', 'a', None), 'keep_remote')
    def test_new_local_file(self): self.assertEqual(choose_change(None, 'a', None), 'apply_local')
    def test_new_file_conflict(self): self.assertEqual(choose_change(None, 'a', 'b'), 'conflict')
    def test_latest_check_retry_used(self):
        a = {'id': 1, 'name': 'ci', 'status': 'completed', 'conclusion': 'failure'}
        b = dict(a, id=2, conclusion='success')
        self.assertEqual(check_state([a, b], []), ([], [], 1))
    def test_pending_check_detected(self):
        self.assertEqual(check_state([{'id': 1, 'name': 'ci', 'status': 'queued'}], [])[1], ['ci'])
    def test_failed_check_detected(self):
        self.assertEqual(check_state([{'id': 1, 'name': 'ci', 'status': 'completed', 'conclusion': 'failure'}], [])[0], ['ci'])
    def test_absent_checks_explicit(self): self.assertEqual(check_state([], []), ([], [], 0))
    def test_latest_status_used(self):
        self.assertEqual(check_state([], [{'id': 2, 'context': 'ci', 'state': 'success'},
                                         {'id': 1, 'context': 'ci', 'state': 'failure'}]), ([], [], 1))
    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a'; atomic(p, b'first'); atomic(p, b'second'); self.assertEqual(p.read_bytes(), b'second')
    def test_orphan_write_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a'; p.with_name('a.writing').write_bytes(b'old')
            with self.assertRaises(ValueError): atomic(p, b'new')
    def test_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'x').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError): safe_path(root, 'x/a')
    def test_notebook_outputs_do_not_change_source_signature(self):
        a = {'cells': [{'cell_type': 'code', 'source': '1+1', 'outputs': []}]}
        b = {'cells': [{'cell_type': 'code', 'source': ['1+1'], 'outputs': [{'text': '2'}]}]}
        self.assertEqual(notebook_signature(encoded(a)), notebook_signature(encoded(b)))


    def test_actual_helper_has_no_literal_secret_findings(self):
        self.assertEqual(secret_findings(Path(__file__).read_bytes()), [])

    def test_actual_helper_passes_production_publication_inspection(self):
        self.assertIsNone(inspect_bytes('tools/otto_workspace_sync.py', Path(__file__).read_bytes()))

    def test_actual_helper_preflight_binds_actual_bytes(self):
        self.assertEqual(helper_source_preflight()['script_sha256'], file_digest(__file__))

    def test_helper_filename_is_not_a_scanner_exemption(self):
        fixture = b'https://example.invalid/?' + b'X-Amz-' + b'Signature' + b'=' + b'abc'
        data = Path(__file__).read_bytes() + b'\n# ' + fixture + b'\n'
        with self.assertRaisesRegex(ValueError, 'Possible credential'):
            inspect_bytes('tools/otto_workspace_sync.py', data)

    def test_credential_query_marker_is_still_blocked(self):
        fixture = b'https://example.invalid/?' + b'X-Amz-' + b'Credential' + b'=' + b'example'
        with self.assertRaisesRegex(ValueError, 'Possible credential'):
            inspect_bytes('docs/example.md', fixture)

    def test_token_in_helper_is_still_blocked(self):
        fixture = b'# ' + b'ghp_' + b'A' * 35
        with self.assertRaisesRegex(ValueError, 'Possible credential'):
            inspect_bytes('tools/otto_workspace_sync.py', fixture)

    def test_signed_url_in_notebook_output_is_still_blocked(self):
        fixture = b'https://example.invalid/?' + b'X-Amz-' + b'Signature' + b'=' + b'abc'
        nb = {'nbformat': 4, 'metadata': {}, 'cells': [
            {'cell_type': 'code', 'source': '', 'outputs': [
                {'output_type': 'stream', 'name': 'stdout', 'text': fixture.decode()}]}]}
        with self.assertRaisesRegex(ValueError, 'Possible credential'):
            inspect_bytes('notebooks/example.ipynb', encoded(nb))

    def test_missing_latest_has_explicit_preparation_gate(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}):
            with self.assertRaises(PublicationNotPrepared):
                latest()
            self.assertFalse((Path(d) / 'latest.json').exists())

    def test_missing_plan_preserves_existing_pointer(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}):
            p = Path(d) / 'latest.json'
            atomic(p, encoded({'run': 'run-fixture'}))
            before = p.read_bytes()
            with self.assertRaises(PublicationNotPrepared):
                latest()
            self.assertEqual(p.read_bytes(), before)
            self.assertFalse((Path(d) / 'run-fixture').exists())

    def test_publish_without_preparation_never_authenticates(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}), \
                mock.patch(__name__ + '.github_cli') as auth:
            with self.assertRaises(PublicationNotPrepared):
                publish_public(True, False)
            auth.assert_not_called()
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_finish_without_preparation_never_authenticates(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}), \
                mock.patch(__name__ + '.github_cli') as auth:
            with self.assertRaises(PublicationNotPrepared):
                finish_public(False)
            auth.assert_not_called()

    def test_invalid_pointer_preserved(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}):
            p = Path(d) / 'latest.json'
            for value in ({'run': '../elsewhere'}, {'run': ''}, {'run': 1}, []):
                atomic(p, encoded(value))
                before = p.read_bytes()
                with self.assertRaises(ValueError):
                    latest()
                self.assertEqual(p.read_bytes(), before)

    def test_valid_preparation_pointer_is_read_without_mutation(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}):
            root = Path(d)
            plan = {'status': 'PREPARED_PUBLIC', 'target': TARGET}
            atomic(root / 'latest.json', encoded({'run': 'run-fixture'}))
            atomic(root / 'run-fixture/plan.json', encoded(plan))
            got_path, got_plan = latest()
            self.assertEqual(got_path, root / 'run-fixture')
            self.assertEqual(got_plan, plan)

    def test_missing_self_test_receipt_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}):
            with self.assertRaisesRegex(ValueError, 'Run self-test'):
                require_helper_tests()

    def test_prepare_scans_actual_helper_before_network(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(globals(), {'STATE': Path(d)}), \
                mock.patch(__name__ + '.require_helper_tests'), \
                mock.patch(__name__ + '.helper_source_preflight', side_effect=ValueError('source fixture refused')), \
                mock.patch(__name__ + '.github_cli') as auth:
            with self.assertRaisesRegex(ValueError, 'source fixture refused'):
                prepare_public(False)
            auth.assert_not_called()

    def test_synthetic_prepare_reaches_review_with_actual_helper_bytes(self):
        """Exercise preparation through final scanning/plan writes; Git/network are mocked.

        This is a helper integration fixture, not a real Git clone, push or AWS replay.
        """
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            state = home / 'otto_public_sync_state'
            run = state / 'run-fixture'
            run.mkdir(parents=True)
            source = home / 'otto-recommender-system'
            (source / '.git').mkdir(parents=True)
            raw_readme = b'# OTTO fixture\n'
            (source / 'README.md').write_bytes(raw_readme)
            head = 'a' * 40
            calls = []

            def fake_command(args, **kwargs):
                self.assertIn('clone', args)
                clone = Path(args[-1])
                clone.mkdir()
                (clone / 'README.md').write_bytes(raw_readme)
                calls.append('clone')
                return ''

            def fake_git(repo, *args, **kwargs):
                calls.append(args)
                if args == ('rev-parse', 'HEAD') or args == ('rev-parse', 'origin/main'):
                    return head + '\n'
                if args == ('status', '--porcelain=v1'):
                    return ''
                if args and args[0] == 'fetch':
                    return ''
                if args and args[0] == 'merge-base':
                    return head + '\n'
                if args and args[0] == 'ls-tree':
                    return b'100644 blob ' + git_blob(raw_readme).encode() + b'\tREADME.md\0'
                if args == ('ls-files', '--cached', '--others', '--exclude-standard', '-z'):
                    return 'README.md\0'
                if args[:2] == ('config', '--local') or args == ('diff', '--check'):
                    return ''
                if args == ('diff', '--no-renames', '--name-only', '-z'):
                    return 'README.md\0'
                if args == ('ls-files', '--others', '--exclude-standard', '-z'):
                    clone = Path(repo)
                    paths = sorted(p.relative_to(clone).as_posix() for p in clone.rglob('*')
                                   if p.is_file() and p.name != 'README.md')
                    # The generated research/manual/README.md is also untracked.
                    paths += ['research/manual/README.md']
                    return ''.join(p + '\0' for p in sorted(set(paths)))
                self.fail('Unexpected Git operation in preparation fixture: ' + repr(args))

            environment = {'HOME': home, 'STATE': state, 'RUN': run, 'EVENTS': [], 'GH': 'gh'}
            with mock.patch.dict(globals(), environment), \
                    mock.patch(__name__ + '.command', side_effect=fake_command), \
                    mock.patch(__name__ + '.git', side_effect=fake_git), \
                    mock.patch(__name__ + '.github_cli', return_value={'login': OWNER, 'id': 1}), \
                    mock.patch(__name__ + '.api', side_effect=AssertionError('No GitHub API in fixture')), \
                    mock.patch.object(shutil, 'disk_usage', return_value=mock.Mock(free=20 * 1024**3)):
                atomic(state / 'self_test.json', encoded({
                    'status': 'PUBLIC_SYNC_SELF_TESTS_PASSED', 'script_sha256': file_digest(__file__)}))
                with mock.patch('sys.stdout', new=io.StringIO()):
                    prepare_public(False)
                saved_run, plan = latest()
                self.assertEqual(saved_run, run)
                self.assertEqual(plan['status'], 'PREPARED_PUBLIC')
                approved = {r['path']: r for r in plan['files']}
                self.assertIn('tools/otto_workspace_sync.py', approved)
                self.assertEqual(approved['tools/otto_workspace_sync.py']['sha256'], file_digest(__file__))
                self.assertIsNone(inspect_bytes('tools/otto_workspace_sync.py',
                                                (run / 'checkout/tools/otto_workspace_sync.py').read_bytes()))
                self.assertTrue((run / 'REVIEW.md').is_file())
                self.assertEqual((source / 'README.md').read_bytes(), raw_readme)
                self.assertEqual([e['event'] for e in EVENTS],
                                 ['PUBLIC_HELPER_SOURCE_SCAN_PASSED', 'PUBLIC_PREPARED_NOT_PUSHED'])
                self.assertFalse(any(isinstance(c, tuple) and c[0] in ('commit', 'push') for c in calls))


class GitPublicationRegression(unittest.TestCase):
    """Owner-run real local Git fixtures; no network or project repositories."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='otto-publication-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.state = self.home / 'otto_public_sync_state'
        self.run = self.state / 'run-fixture'
        self.run.mkdir(parents=True)
        self.clone = self.run / 'checkout'
        self.clone.mkdir()
        self.environment = mock.patch.dict(globals(), {
            'HOME': self.home, 'STATE': self.state, 'RUN': self.run,
            'GH': None, 'EVENTS': []})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # Isolate fixture Git configuration, hooks and commit identities. These
        # settings apply only to tests, never to the real publication checkout.
        self.env = mock.patch.dict(os.environ, {
            'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_AUTHOR_NAME': 'OTTO fixture', 'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
            'GIT_COMMITTER_NAME': 'OTTO fixture', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.auth = mock.patch(__name__ + '.github_cli', side_effect=AssertionError('Unexpected network/authentication'))
        self.auth.start()
        self.addCleanup(self.auth.stop)
        git(self.clone, 'init', '--quiet')
        git(self.clone, 'config', '--local', 'user.name', 'OTTO fixture')
        git(self.clone, 'config', '--local', 'user.email', 'fixture@example.invalid')
        git(self.clone, 'config', '--local', 'commit.gpgsign', 'false')
        git(self.clone, 'config', '--local', 'core.hooksPath', '/dev/null')
        atomic(self.clone / 'README.md', b'# Fixture\n')
        git(self.clone, 'add', '--', 'README.md')
        git(self.clone, 'commit', '--quiet', '-m', 'Fixture baseline')
        self.base = git(self.clone, 'rev-parse', 'HEAD').strip()
        git(self.clone, 'remote', 'add', 'origin', 'https://github.com/' + TARGET + '.git')

    def add_file(self, relative, data, stage=False):
        atomic(safe_path(self.clone, relative), data)
        if stage:
            git(self.clone, '--literal-pathspecs', 'add', '--', relative)
        return {'path': relative, 'sha256': digest(data)}

    def test_real_git_all_staged_files_remain_in_delta(self):
        self.add_file('a.py', b'x = 1\n', stage=True)
        self.assertEqual(publication_delta(self.clone), {'a.py'})
        self.assertEqual(git(self.clone, 'diff', '--name-only').strip(), '')

    def test_real_git_partially_staged_files_remain_in_delta(self):
        self.add_file('a.py', b'x = 1\n', stage=True)
        self.add_file('b.py', b'y = 2\n')
        self.assertEqual(publication_delta(self.clone), {'a.py', 'b.py'})

    def test_real_git_stage_replay_is_idempotent(self):
        rows = [self.add_file('a.py', b'x = 1\n', stage=True)]
        with mock.patch('sys.stdout', new=io.StringIO()):
            first = stage_reviewed_files(self.run, self.clone, rows)
            second = stage_reviewed_files(self.run, self.clone, rows)
        self.assertEqual(first, second)
        self.assertEqual(git(self.clone, 'rev-parse', 'HEAD').strip(), self.base)

    def test_real_git_unreviewed_staged_file_stops(self):
        rows = [self.add_file('a.py', b'x = 1\n')]
        self.add_file('unexpected.py', b'z = 3\n', stage=True)
        with self.assertRaisesRegex(ValueError, 'differ from the reviewed plan'):
            stage_reviewed_files(self.run, self.clone, rows)

    def test_real_git_expected_eof_is_notice_without_source_rewrite(self):
        path = 'research/manual/otto_feature_round11/source_contract.py'
        data = b'x = 1\n\n'
        row = self.add_file(path, data, stage=True)
        with mock.patch.dict(globals(), {'EOF_SNAPSHOTS': {path: digest(data)}}), \
                mock.patch('sys.stdout', new=io.StringIO()):
            result = checked_public_diff(self.clone, [row])
        self.assertNotEqual(result['exit_code'], 0)
        self.assertEqual(len(result['archival_eof_notices']), 1)
        self.assertEqual((self.clone / path).read_bytes(), data)

    def test_real_git_eof_in_nonarchival_source_is_blocked(self):
        row = self.add_file('src/new.py', b'x = 1\n\n', stage=True)
        with self.assertRaisesRegex(ValueError, 'PUBLIC_DIFF_CHECK_BLOCKED'):
            checked_public_diff(self.clone, [row])

    def test_real_git_archival_filename_without_exact_hash_is_blocked(self):
        path = next(iter(EOF_SNAPSHOTS))
        row = self.add_file(path, b'x = 1\n\n', stage=True)
        with self.assertRaisesRegex(ValueError, 'PUBLIC_DIFF_CHECK_BLOCKED'):
            checked_public_diff(self.clone, [row])

    def test_real_git_trailing_space_is_not_an_eof_exception(self):
        path = 'research/manual/otto_feature_round11/source_contract.py'
        data = b'x = 1 \n\n'
        row = self.add_file(path, data, stage=True)
        with mock.patch.dict(globals(), {'EOF_SNAPSHOTS': {path: digest(data)}}):
            with self.assertRaisesRegex(ValueError, 'trailing whitespace'):
                checked_public_diff(self.clone, [row])

    def test_real_git_conflict_marker_is_still_blocked(self):
        row = self.add_file('conflicted.py', b'<<<<<<< ours\nx = 1\n=======\nx = 2\n>>>>>>> theirs\n', stage=True)
        with self.assertRaisesRegex(ValueError, 'conflict marker'):
            checked_public_diff(self.clone, [row])

    def test_real_git_eof_plus_unrelated_error_is_not_accepted(self):
        path = 'research/manual/otto_feature_round11/source_contract.py'
        data = b'x = 1\n\n'
        rows = [self.add_file(path, data, stage=True),
                self.add_file('bad.py', b'y = 2 \n', stage=True)]
        with mock.patch.dict(globals(), {'EOF_SNAPSHOTS': {path: digest(data)}}):
            with self.assertRaisesRegex(ValueError, 'bad.py'):
                checked_public_diff(self.clone, rows)

    def test_real_command_stdout_diagnostic_is_not_lost(self):
        with self.assertRaises(RuntimeError) as caught:
            command([sys.executable, '-c', 'import sys; print("explanation on stdout"); sys.exit(2)'])
        self.assertIn('explanation on stdout', str(caught.exception))
        self.assertIn('stdout', str(caught.exception))
        self.assertIn('exited 2', str(caught.exception))

    def test_real_command_stderr_diagnostic_is_retained(self):
        with self.assertRaises(RuntimeError) as caught:
            command([sys.executable, '-c', 'import sys; print("stderr explanation", file=sys.stderr); sys.exit(2)'])
        self.assertIn('stderr explanation', str(caught.exception))

    def test_real_command_empty_streams_name_the_command(self):
        with self.assertRaises(RuntimeError) as caught:
            command([sys.executable, '-c', 'raise SystemExit(2)'])
        self.assertIn('stdout: <empty>', str(caught.exception))
        self.assertIn('stderr: <empty>', str(caught.exception))
        self.assertIn('-c', str(caught.exception))

    def test_real_command_result_object_preserves_both_streams(self):
        result = command([sys.executable, '-c',
                          'import sys; print("out"); print("err", file=sys.stderr); sys.exit(2)'],
                         allowed=(2,), result_object=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout.strip(), b'out')
        self.assertEqual(result.stderr.strip(), b'err')

    def legacy_fixture(self):
        tool_data = b'# Synthetic old controller\n'
        path = 'research/manual/otto_feature_round11/source_contract.py'
        data = b'x = 1\n\n'
        rows = [self.add_file('tools/otto_workspace_sync.py', tool_data, stage=True),
                self.add_file(path, data, stage=True)]
        plan = {'target': TARGET, 'status': 'PREPARED_PUBLIC', 'base': self.base,
                'checkout': 'checkout', 'files': rows, 'script_sha256': digest(tool_data),
                'source_checkout_changed': False, 'selected_files': 1}
        raw = encoded(plan)
        atomic(self.run / 'plan.json', raw)
        atomic(self.state / 'latest.json', encoded({'run': self.run.name}))
        atomic(self.state / 'self_test.json', encoded({
            'status': 'PUBLIC_SYNC_SELF_TESTS_PASSED', 'script_sha256': file_digest(__file__)}))
        return path, data, raw, {'LEGACY_HELPER_SHA': digest(tool_data),
                                  'LEGACY_PLAN_SHA': digest(raw),
                                  'EOF_SNAPSHOTS': {path: digest(data)}}

    def test_real_recovery_preserves_prepared_source_and_original_plan(self):
        path, data, raw, settings = self.legacy_fixture()
        with mock.patch.dict(globals(), settings), mock.patch('sys.stdout', new=io.StringIO()):
            resume_public()
        _, plan = latest()
        self.assertEqual(plan['script_sha256'], file_digest(__file__))
        self.assertEqual((self.clone / path).read_bytes(), data)
        self.assertEqual((self.run / 'publisher_recovery/plan.before.json').read_bytes(), raw)
        self.assertEqual(plan['status'], 'PREPARED_PUBLIC')
        self.assertEqual(git(self.clone, 'rev-parse', 'HEAD').strip(), self.base)
        self.assertEqual(len(plan['files']), 3)
        self.assertTrue((self.run / 'RESUME_REVIEW.md').is_file())

    def test_real_recovery_can_be_repeated_without_new_plan_or_clone(self):
        _, _, _, settings = self.legacy_fixture()
        with mock.patch.dict(globals(), settings), mock.patch('sys.stdout', new=io.StringIO()):
            resume_public()
            before = (self.run / 'plan.json').read_bytes()
            resume_public()
        self.assertEqual((self.run / 'plan.json').read_bytes(), before)
        self.assertEqual(read(self.state / 'latest.json')['run'], self.run.name)

    def test_real_recovery_unknown_plan_stops_before_changes(self):
        _, _, _, settings = self.legacy_fixture()
        settings['LEGACY_PLAN_SHA'] = '0' * 64
        original_tool = (self.clone / 'tools/otto_workspace_sync.py').read_bytes()
        with mock.patch.dict(globals(), settings):
            with self.assertRaisesRegex(ValueError, 'exact returned prepared plan'):
                resume_public()
        self.assertEqual((self.clone / 'tools/otto_workspace_sync.py').read_bytes(), original_tool)
        self.assertFalse((self.clone / PUBLIC_CHECK_NOTES).exists())

    def test_real_recovery_modified_source_is_preserved_and_stops(self):
        path, _, _, settings = self.legacy_fixture()
        modified = b'x = 999\n'
        atomic(self.clone / path, modified)
        with mock.patch.dict(globals(), settings):
            with self.assertRaisesRegex(ValueError, 'Prepared source changed'):
                resume_public()
        self.assertEqual((self.clone / path).read_bytes(), modified)

    def test_real_recovery_resumes_after_intent_before_plan_update(self):
        _, _, raw, settings = self.legacy_fixture()
        original_atomic = atomic
        def fail_once(path, data):
            if Path(path) == self.run / 'plan.json' and data != raw:
                raise RuntimeError('Synthetic interruption before plan update')
            return original_atomic(path, data)
        with mock.patch.dict(globals(), settings), mock.patch('sys.stdout', new=io.StringIO()):
            with mock.patch(__name__ + '.atomic', side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
                    resume_public()
            resume_public()
        self.assertEqual(latest()[1]['script_sha256'], file_digest(__file__))

    def test_parse_diagnostics_omits_added_source_content(self):
        rows, other = parse_diff_check(b'a.py:1: trailing whitespace.\n+potentially sensitive content \n')
        self.assertEqual(rows, [{'path': 'a.py', 'line': 1, 'issue': 'trailing whitespace.'}])
        self.assertEqual(other, [])

    def test_archival_policy_has_no_global_or_filename_only_exception(self):
        self.assertEqual(len(EOF_SNAPSHOTS), 13)
        self.assertTrue(all(p.startswith('research/manual/otto_feature_round') for p in EOF_SNAPSHOTS))
        self.assertTrue(all(re.fullmatch(r'[a-f0-9]{64}', h) for h in EOF_SNAPSHOTS.values()))


def self_test():
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(PublicationContracts),
        unittest.defaultTestLoader.loadTestsFromTestCase(GitPublicationRegression)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Public-sync helper tests failed; no network work was started')
    receipt = {'status': 'PUBLIC_SYNC_SELF_TESTS_PASSED', 'tests_run': result.testsRun,
               'utc': utc(), 'script_sha256': file_digest(__file__), 'project_model_fits': 0,
               'network_calls': 0, 'project_repository_git_operations': 0,
               'temporary_local_git_fixtures_executed': True}
    atomic(STATE / 'self_test.json', encoded(receipt))
    record('PUBLIC_SYNC_SELF_TESTS_PASSED', tests_run=result.testsRun)


def main():
    global RUN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('self-test', 'prepare-public', 'publish-public',
                                      'finish-public', 'resume-public', 'repair', 'bundle'))
    p.add_argument('--approve-public', action='store_true')
    p.add_argument('--install-gh-if-needed', action='store_true')
    args = p.parse_args()
    if STATE.is_symlink():
        raise ValueError('State-directory symlink refused')
    STATE.mkdir(exist_ok=True)
    STATE.chmod(0o700)
    RUN = STATE / ('run-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid()))
    RUN.mkdir(mode=0o700)
    code = 0
    try:
        if platform.system() != 'Linux':
            raise ValueError('Run in your existing SageMaker JupyterLab terminal, not Windows PowerShell.')
        with (HOME / '.otto_rounds08_09.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'self-test': self_test()
            elif args.action == 'resume-public': resume_public()
            elif args.action == 'prepare-public': prepare_public(args.install_gh_if_needed)
            elif args.action == 'publish-public': publish_public(args.approve_public, args.install_gh_if_needed)
            elif args.action == 'finish-public': finish_public(args.install_gh_if_needed)
            elif args.action == 'repair': repair()
    except PublicationNotPrepared as exc:
        code = 2
        message = redact(exc)
        atomic(RUN / 'error.json', encoded({'status': 'PUBLIC_NOT_PREPARED', 'reason': message,
                                          'target': TARGET, 'project_model_fits': 0}))
        print('PUBLIC_NOT_PREPARED: ' + message, flush=True)
    except Exception as exc:
        code = 2
        message = redact(exc)
        atomic(RUN / 'error.json', encoded({'status': 'STOPPED_WITH_EVIDENCE', 'reason': message,
                                          'target': TARGET, 'project_model_fits': 0}))
        print('STOPPED_WITH_EVIDENCE: ' + message, flush=True)
    finally:
        bundle()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
