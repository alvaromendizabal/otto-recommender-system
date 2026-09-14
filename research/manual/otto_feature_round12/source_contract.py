"""Read-only original lineage checks; private data only through verified prior helpers."""
from __future__ import annotations
import importlib.metadata, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from core import checked,sha
ROOT=Path(__file__).resolve().parent
COMMIT='2638faa34afa427ed4ac4e92bba04deda57c688e'
HISTORY_END,FIT_END=1660687200000,1660946400000
SIGNALS=('graph_time_all_n1_uniform_sum','graph_cart_all_n5_uniform_max',
'domain_norm_symmetric_time_row_recent_unique_mean','domain_norm_forward_time_row_recent_unique_mean',
'domain_norm_symmetric_order_row_purchase_unique_mean','domain_norm_symmetric_order_row_last_mean',
'domain_funnel_full_count_share')
def read(p):return json.loads(Path(p).read_text())
def safe(root,rel):
 p=root/rel
 if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe source path')
 return p
def source_context(home: Path):
    p={};previous=home/'otto_feature_round02';old=previous/'outputs'
    ref=read(ROOT/'evidence/ROUND02_REFERENCE.json')
    for name,digest in ref['round02_files'].items():checked(safe(previous,name),digest)
    from legacy_imports import load_replication
    prior=load_replication(previous, ref['round02_files'])
    paths=prior.paths_for(home);repo=paths['repo']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,timeout=15).strip()
    if git('rev-parse','HEAD')!=COMMIT or git('branch','--show-current')!='main' or git('status','--porcelain'):
        raise ValueError('Reviewed clean main required; preserve local changes, never reset')
    if sys.version_info[:2]!=(3,13):raise ValueError('Use launch.py and the existing project Python 3.13')
    ready=read(home/'otto_manual_workspace/reports/readiness.json')
    if ready.get('readiness')!='SOURCE_AND_INPUTS_VERIFIED':raise ValueError('Verified setup receipt missing')
    for row in ready['environment']['packages']:
        if importlib.metadata.version(row['package'])!=row['expected']:raise ValueError('Existing environment changed: '+row['package'])
    if shutil.disk_usage(home).free<10*1024**3:raise ValueError('Need 10 GiB free; do not delete prior data')
    sys.path.insert(0,str(repo/'src'))
    feature=read(old/'feature_contract.json');manifest=read(old/'feature_manifest.json');co=read(old/'cohort.json')
    if manifest['contract_id']!=sha(old/'feature_contract.json'):raise ValueError('Original feature lineage changed')
    ids=np.asarray(co['ids'],np.int64)
    if len(ids)!=4096 or np.unique(ids).size!=4096 or not np.array_equal(ids,np.sort(ids)):raise ValueError('Original cohort changed')
    names=feature['names'][:134]
    if len(names)!=134 or not set(SIGNALS)<=set(names):raise ValueError('Cached graph affinity schema differs')
    for name,dst in [('result.json','ROUND06_RESULT.json'),('candidate_coverage.json','ROUND06_COVERAGE.json'),('evaluation_statistics.npz','ROUND06_STATISTICS.npz')]:
        checked(home/'otto_feature_round06/outputs'/name,sha(ROOT/'evidence'/dst))
    cm=read(paths['corpus']/'manifest.json');checked(paths['corpus']/'manifest.json',feature['corpus_manifest_sha256'])
    if cm['protocol']['history_end']!=HISTORY_END or cm['protocol']['fit_end']!=FIT_END:raise ValueError('Wrong temporal corpus')
    for name in ('observed.parquet','queries.parquet'):checked(paths['corpus']/name,cm['files'][name])
    rm=read(paths['retrieval']/'manifest.json');checked(paths['retrieval']/'manifest.json',feature['retrieval_manifest_sha256'])
    if rm['status']!='passed' or rm['history_end']!=HISTORY_END or rm['query_labels_used'] is not False or rm['observed_history_max_ts']>=HISTORY_END:
        raise ValueError('Historical retrieval source cutoff/provenance failed')
    ix=read(ROOT/'evidence/ROUND05_INDEX_CONTRACT.json');sc=read(ROOT/'evidence/ROUND05_SOURCE_CHECK.json')
    checked(home/'otto_feature_round05/outputs/index/contract.json',sha(ROOT/'evidence/ROUND05_INDEX_CONTRACT.json'))
    checked(home/'otto_feature_round05/outputs/index/source_check.json',sha(ROOT/'evidence/ROUND05_SOURCE_CHECK.json'))
    if sc['contract_id']!=sha(ROOT/'evidence/ROUND05_INDEX_CONTRACT.json') or sc['max_ts']>=HISTORY_END or ix['history_tail_sha256']!=rm['files']['history_tail.parquet']:
        raise ValueError('Certified historical tail source differs')
    excluded=np.asarray(ix['excluded_sessions'],np.int64)
    if np.unique(excluded).size!=5120 or not set(map(int,ids))<=set(map(int,excluded)):raise ValueError('Study-session exclusions differ')
    return dict(paths=paths,prior=prior,old=old,cohort=co,ids=ids,names=names,manifest=manifest,
                excluded=excluded,history=paths['retrieval']/'history_tail.parquet',history_sha=rm['files']['history_tail.parquet'],protocol=p)

