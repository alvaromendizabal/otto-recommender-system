"""Bounded manual launcher: use the already installed OTTO project interpreter."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def verify_package():
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        if relative.endswith('.ipynb'):
            continue  # Executing notebooks legitimately changes their saved output bytes.
        p = ROOT/relative
        if not p.resolve().is_relative_to(ROOT.resolve()) or p.is_symlink():
            raise RuntimeError('Invalid package path')
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Changed package file: '+relative)


def run_stage(phase):
    if phase not in ('features', 'screen', 'report', 'bundle', 'tests'):
        raise ValueError('Unknown stage')
    verify_package()
    py = Path.home()/'otto-recommender-system/.venv/bin/python'
    if not py.is_file():
        raise RuntimeError('Existing project interpreter missing; do not install or wipe anything')
    out = ROOT/'outputs';out.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '4', 'POLARS_MAX_THREADS': '4', 'AWS_MAX_ATTEMPTS': '1'}
    if phase == 'tests':
        cmd = [str(py), '-W', 'error', '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-v']
    else:
        cmd = [str(py), '-u', str(ROOT/'experiment.py'), phase]
    cap = {'features':315, 'screen':255, 'report':75, 'bundle':30, 'tests':45}[phase]
    log_path = out/(phase+'.log')
    print(f'RUNNING {phase}; hard process limit {cap} seconds. Log: {log_path}', flush=True)
    start = time.monotonic()
    # File-backed output prevents a silent child's pipe from defeating the timeout.
    with log_path.open('ab', buffering=0) as log:
        read_from = log_path.stat().st_size
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            with log_path.open('r', errors='replace') as reader:
                reader.seek(read_from)
                while proc.poll() is None:
                    text = reader.read()
                    if text: print(text, end='', flush=True)
                    if time.monotonic()-start > cap:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL);proc.wait(timeout=5)
                        raise TimeoutError(f'{phase} hard time limit; existing checkpoints retained')
                    time.sleep(.3)
                text = reader.read()
                if text: print(text, end='', flush=True)
            if proc.returncode:
                raise RuntimeError(f'{phase} stopped, exit={proc.returncode}. Return the ZIP/log; do not retry blindly.')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL);proc.wait(timeout=5)
            if phase not in ('bundle', 'tests'):
                # Report-only recovery never resumes training after failure.
                subprocess.run([str(py), str(ROOT/'experiment.py'), 'bundle'], cwd=ROOT,
                               env=env, timeout=40, check=False)
    print(f'FINISHED {phase}: {time.monotonic()-start:.1f} seconds', flush=True)
    return {'phase':phase, 'exit_code':proc.returncode, 'log':str(log_path)}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python launch.py tests|features|screen|report|bundle')
    run_stage(sys.argv[1])
