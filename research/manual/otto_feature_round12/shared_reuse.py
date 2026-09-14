"""Read-only lineage adapter. Original workspace, graphs and code remain unchanged."""
import importlib.util
import json
from pathlib import Path
from core import checked, sha
from shared_data import prior_chunks
ROOT=Path(__file__).resolve().parent
FROZEN=('model_params','rounds','negative_budget','negative_sampling_seed','sessions','candidate_budget',
        'control_features','added_features','ablation_features','maximum_new_models','embargo_hours',
        'source_commit','source_cutoff_ms')
def context(home):
 home=Path(home);root=home/'otto_feature_round08'
 ref=json.loads((ROOT/'evidence/ROUND08_REUSE_CONTRACT.json').read_text())
 for name,digest in ref['source_files'].items():checked(root/name,digest)
 for name,digest in ref['outputs'].items():checked(root/'outputs'/name,digest)
 p=json.loads((ROOT/'protocol.json').read_text());old=json.loads((root/'protocol.json').read_text())
 for key in FROZEN:
  if p[key]!=old[key]:raise ValueError('Frozen control comparison changed: '+key)
 reference=json.loads((ROOT/'evidence/COMPLETED_ROUNDS09_10.json').read_text())
 for number,entries in reference.items():
  for name,digest in entries.items():checked(home/f'otto_feature_round{int(number):02d}'/'outputs'/name,digest)
 if p['round']==12:
  out=home/'otto_feature_round11/outputs'
  receipt=json.loads((out/'report_receipt.json').read_text())
  if receipt.get('status')!='ROUND11_REPORT_READY':raise ValueError('Finish Round11 through its report, even if negative. Do not proceed past failure.')
  checked(out/'result.json',receipt['result_sha256'])
 spec=importlib.util.spec_from_file_location('_otto_round08_shared_readonly',root/'shared_data.py')
 if spec is None or spec.loader is None:raise ValueError('Verified shared reader unavailable')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 ctx=module.context(home);module.require(ctx);ctx['protocol']=p;ctx['_shared_reader']=module
 return ctx
def require(ctx):return ctx['_shared_reader'].require(ctx)
