"""Manual notebook/terminal entry point. Bounds processes; never launches cloud compute."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
CAPS={'tests':60,'index':320,'features':320,'screen':260,'report':60}


def package_check():
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        expected,rel=line.split('  ',1)
        if rel.endswith('.ipynb'):continue
        p=ROOT/rel
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Unsafe package file')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Package file changed: '+rel)


def collect():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    index_receipts=[]
    for rp in sorted((out/'index').glob('part-*.json')):
        r=json.loads(rp.read_text());index_receipts.append(r)
    if index_receipts:
        (out/'INDEX_PROGRESS.json').write_text(json.dumps({'completed_partitions':len(index_receipts),
            'total_partitions':16,'partitions':index_receipts},indent=2)+'\n')
    files=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(out)
        if rel.parts[0] in ('features','models','index'):continue
        if p.suffix in ('.json','.md','.log','.npz','.html'):files.append(p)
    # Include executed notebooks as an audit of code-cell completion; not model/data chunks.
    files.extend(p for p in sorted(ROOT.glob('*.ipynb')) if p.is_file())
    for rel in ('protocol.json','feature_catalog.json','core.py','experiment.py','intent_features.py',
                'report.py','launch.py','notebook_view.py','run_tests.py','MANIFEST.sha256',
                'evidence/ROUND02_REFERENCE.json','evidence/ROUND02_CONTROL_STATISTICS.npz',
                'evidence/ROUND02_AUDIT.json','transition_features.py','transition_index.py','backend_smoke.py','analysis_utils.py',
                'evidence/ROUND04_AUDIT.json','evidence/ROUND04_RESULT.json',
                'outputs/index/contract.json','outputs/index/source_check.json',
                'outputs/index/index_manifest.json'):
        candidate=ROOT/rel
        if candidate.is_file():files.append(candidate)
    size=sum(p.stat().st_size for p in files)
    if size>25*1024**2:raise ValueError('Unexpected large return bundle; local evidence preserved')
    manifest=[];target=ROOT/'otto_round05_return.zip';tmp=target.with_suffix('.assembling')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            b=p.read_bytes();rel=str(p.relative_to(ROOT));z.writestr(rel,b)
            manifest.append({'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
        z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    os.replace(tmp,target)
    print('RETURN_FILE: '+str(target),flush=True)
    return str(target)


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)


def run_stage(phase):
    if phase=='bundle':return collect()
    if phase not in CAPS:raise ValueError('Unknown stage')
    package_check()
    python=Path.home()/'otto-recommender-system/.venv/bin/python'
    if not python.is_file():raise RuntimeError('Project interpreter missing. Do not install or wipe anything')
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    if phase=='tests':cmd=[str(python),'-W','error',str(ROOT/'run_tests.py')]
    elif phase=='report':cmd=[str(python),'-u',str(ROOT/'report.py')]
    else:cmd=[str(python),'-u',str(ROOT/'experiment.py'),phase]
    env={**os.environ,'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'POLARS_MAX_THREADS':'4','PYTHONUNBUFFERED':'1'}
    log=out/(phase+'.log');start=time.monotonic();proc=None;reason=None;cap=CAPS[phase]
    print(f'RUNNING {phase}; process cap {cap}s. No AWS resource changes. Log: {log}',flush=True)
    try:
        with log.open('ab',buffering=0) as writer:
            offset=writer.tell()
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=writer,stderr=subprocess.STDOUT,
                                  env=env,start_new_session=True)
            beat=start
            with log.open('r',errors='replace') as reader:
                reader.seek(offset)
                while proc.poll() is None:
                    text=reader.read()
                    if text:print(text,end='',flush=True)
                    elapsed=time.monotonic()-start
                    if elapsed>cap:reason='Hard process time limit';stop_process(proc);break
                    try:
                        rssline=next(s for s in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if s.startswith('VmRSS:'))
                        rss=int(rssline.split()[1])/1024**2
                        if rss>26:reason='26 GiB worker memory limit';stop_process(proc);break
                    except (FileNotFoundError,StopIteration):pass
                    if time.monotonic()-beat>=15:
                        print(f'{dt.datetime.now(dt.timezone.utc).isoformat()} LAUNCHER_HEARTBEAT phase={phase} seconds={elapsed:.1f}',flush=True)
                        beat=time.monotonic()
                    time.sleep(.3)
                text=reader.read()
                if text:print(text,end='',flush=True)
        if reason:raise RuntimeError(reason+'; checkpoints preserved. Return the report before retrying.')
        if proc.returncode==75:
            raise RuntimeError('PAUSED_CHECKPOINTED: a planned work boundary was reached, not a model failure. Completed partitions are saved. Return the ZIP; no automatic retry.')
        if proc.returncode:raise RuntimeError(f'{phase} failed with exit {proc.returncode}. Stop and return ZIP/log.')
        print(f'FINISHED {phase}: {time.monotonic()-start:.1f} seconds',flush=True)
        return {'phase':phase,'exit_code':0}
    finally:
        if proc is not None:stop_process(proc)
        (out/'runs').mkdir(exist_ok=True)
        receipt={'phase':phase,'elapsed_seconds':time.monotonic()-start,
                 'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                 'exit_code':None if proc is None else proc.returncode,'reason':reason,
                 'application_stopped':False}
        (out/'runs'/f'launcher-{phase}-{time.time_ns()}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        collect()


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python launch.py tests|index|features|screen|report|bundle')
    run_stage(sys.argv[1])
