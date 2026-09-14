"""Send saved Plotly JSON, not auto-selected HTML renderers. No model imports or fitting."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
START=time.monotonic()
STATE={'chart_payloads_emitted':[],'new_model_fits':0,'feature_rebuilds':0,
       'renderer':'application/vnd.plotly.v1+json','notebook_python':sys.executable,
       'ml_python':str(Path.home()/'otto-recommender-system/.venv/bin/python'),
       'browser_render_verified':False}


def persist(status):
    STATE.update(status=status,elapsed_seconds=time.monotonic()-START,
                 utc=dt.datetime.now(dt.timezone.utc).isoformat())
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/NOTEBOOK_SESSION.json').write_text(json.dumps(STATE,indent=2)+'\n')


def start():
    from IPython.display import display, Markdown
    result=json.loads((ROOT/'outputs/result.json').read_text())
    receipt=json.loads((ROOT/'outputs/report_receipt.json').read_text())
    raw=(ROOT/'outputs/result.json').read_bytes()
    if (receipt['status']!='ROUND05_REPORT_READY' or hashlib.sha256(raw).hexdigest()!=receipt['result_sha256']):
        raise ValueError('Run the report phase successfully first')
    if result['status']!='ROUND05_SCREEN_COMPLETED':raise ValueError('Research comparison not complete')
    STATE['result_sha256']=receipt['result_sha256'];persist('SAVED_RESULT_REVIEW_STARTED')
    print('KERNEL_READY — saved results only, no training. Notebook Python:',sys.executable,flush=True)
    display(Markdown('**Decision:** '+result['comparisons']['action_pairs_minus_control_shared']['decision']+'\n\n'
                    '**Action-pair primary gain versus matched control:** '+f"{result['comparisons']['action_pairs_minus_control_shared']['gain']:+.6f}"+
                    '\n\nReused fitting-cohort exploratory feature screen. Not a Kaggle score; no feature promotion.'))
    return result


def show_chart(number):
    from IPython.display import display
    if number not in range(1,8):raise ValueError('Chart number must be 1..7')
    path=ROOT/'outputs/plots'/f'{number:02d}.json'
    receipt=json.loads((ROOT/'outputs/report_receipt.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['plot_hashes'][path.name]:raise ValueError('Plot checksum mismatch')
    spec=json.loads(path.read_text())
    display({'application/vnd.plotly.v1+json':spec,'text/plain':f'Round 05 chart {number} (interactive Plotly)'},raw=True)
    if number not in STATE['chart_payloads_emitted']:STATE['chart_payloads_emitted'].append(number)
    persist('CHART_PAYLOAD_SENT')
    print(f'CHART_{number}_PAYLOAD_SENT — Python display returned.',flush=True)


def finish():
    if STATE['chart_payloads_emitted']!=list(range(1,8)):raise ValueError('Run all seven saved-chart cells')
    persist('ROUND05_NOTEBOOK_REVIEW_COMPLETE')
    from launch import collect
    print('ROUND05_NOTEBOOK_REVIEW_COMPLETE',flush=True)
    return collect()
