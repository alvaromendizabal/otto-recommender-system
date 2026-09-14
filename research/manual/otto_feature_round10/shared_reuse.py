"""Read-only adapter for the completed Round08 cache; never rebuilds its data.

Loading the old reader preserves its exact source-code identity rather than
pretending that newly edited code produced an old checkpoint. All executable
files, old receipts and actual cached binaries are checked before reuse.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from core import checked, sha
from shared_data import prior_chunks

ROOT = Path(__file__).resolve().parent
FROZEN = ('model_params','rounds','negative_budget','negative_sampling_seed',
          'sessions','candidate_budget','control_features','added_features','ablation_features',
          'maximum_new_models','embargo_hours','source_commit','source_cutoff_ms')


def validate_protocol(current:dict, original:dict)->None:
    for key in FROZEN:
        if current[key] != original[key]:
            raise ValueError('Frozen comparison changed: '+key)
    if (current['round'] != 10 or current['new_arm_names'] != ['ordered_funnel','unordered_funnel_ablation']
        or current['primary_arm'] != 'ordered_funnel' or current['funnel_pairs'] != [[0,1],[1,2],[0,2]]
        or current['funnel_max_elapsed_ms'] != 1800000 or current['funnel_decay_ms'] != 300000
        or current['source_support_shrinkage'] != 20 or current['session_summary_lru_entries'] != 32768
        or any(current[k] for k in ('selection_access','evaluation_access','competition_test_access','algorithm_search'))
        or current['raw_json_scans'] != 0 or current['new_indices'] != 0
        or current['maximum_shared_additional_parquet_scans'] != 0):
        raise ValueError('Round10 preregistered feature contract changed')


def context(home:Path)->dict:
    home=Path(home);root=home/'otto_feature_round08'
    ref=json.loads((ROOT/'evidence/ROUND08_REUSE_CONTRACT.json').read_text())
    for name,digest in ref['source_files'].items():checked(root/name,digest)
    for name,digest in ref['outputs'].items():checked(root/'outputs'/name,digest)
    p=json.loads((ROOT/'protocol.json').read_text())
    old=json.loads((root/'protocol.json').read_text());validate_protocol(p,old)
    # The next hypothesis is fixed without seeing Round09's outcomes, but do not
    # interleave experiments or proceed past operational failures.
    result_path=home/'otto_feature_round09/outputs/result.json'
    receipt_path=home/'otto_feature_round09/outputs/report_receipt.json'
    if not result_path.is_file() or not receipt_path.is_file():
        raise ValueError('Finish pending Round09 through report before Round10. Do not rerun Round08.')
    result=json.loads(result_path.read_text());receipt=json.loads(receipt_path.read_text())
    if result.get('status')!='ROUND09_SCREEN_COMPLETED' or receipt.get('status')!='ROUND09_REPORT_READY':
        raise ValueError('Round09 has not completed. Bundle evidence; no downstream work.')
    checked(result_path,receipt['result_sha256'])
    spec=importlib.util.spec_from_file_location('_otto_round08_shared_readonly',root/'shared_data.py')
    if spec is None or spec.loader is None:raise ValueError('Verified shared reader unavailable')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ctx=module.context(home);module.require(ctx)
    # Same source commit, corpus, historical cutoff and all original numerical
    # controls. Changing the study's round label does not mutate shared records.
    ctx['protocol']=p;ctx['_shared_reader']=module
    return ctx


def reader(ctx:dict)->ModuleType:
    module=ctx.get('_shared_reader')
    if not isinstance(module,ModuleType):raise ValueError('Missing verified read-only shared reader')
    return module


def require(ctx:dict):
    return reader(ctx).require(ctx)


def evidence(ctx:dict):
    return reader(ctx).evidence(ctx)
