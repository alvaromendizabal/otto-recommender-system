"""Manual timing replication on new fitting-session IDs; no downloads or cloud writes.

Source, raw files, existing graphs and previous models are read-only. New evidence
is stored in this package's outputs directory with immutable chunk/model receipts.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from core import (OBJECTIVES, WEIGHTS, array_digest, checked, choose_cohort, decision,
                  encode, forward_folds, hits_at20, paired_interval, pooled, save_arrays,
                  save_json, sha, target_arrays, validate_prefix, write_once)
from intent_features import NAMES, TIMING, FAMILIES, transform

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
COMMIT='2638faa34afa427ed4ac4e92bba04deda57c688e'
HISTORY_END=1660687200000
FIT_END=1660946400000
TIMING_NAMES=tuple(NAMES[i] for i in TIMING)
PROGRESS={'stage':'start','completed':0,'total':0}
DEADLINE=float('inf')


class PausedCheckpointed(Exception):
    pass


def log(event,**details):
    print(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'event':event,**details},
                     sort_keys=True),flush=True)


def before_unit():
    if time.monotonic() > DEADLINE-20:
        raise PausedCheckpointed('Reached safe boundary near phase budget; completed units preserved')


def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ValueError('Cannot load pinned source module')
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def paths_for(home):
    root=home/'otto-artifacts/shared-feature-smoke/early-smoke-91d3dcd1'
    return {'repo':home/'otto-recommender-system', 'corpus':root/'inputs/corpus',
            'retrieval':root/'inputs/retrieval','graphs':root/'output/graphs',
            'scale':home/'otto-artifacts/shared-feature-scale/early-scale-29439ec2/output',
            'round01':home/'otto_feature_round01'}


def check_runtime(home):
    import shutil
    paths=paths_for(home); repo=paths['repo']
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True,timeout=15).strip()
    if git('rev-parse','HEAD')!=COMMIT or git('branch','--show-current')!='main' or git('status','--porcelain'):
        raise ValueError('Require clean reviewed main 2638faa; preserve other local work, do not reset')
    if sys.version_info[:2]!=(3,13):
        raise ValueError('ML stage must use the existing project .venv Python 3.13')
    readiness=json.loads((home/'otto_manual_workspace/reports/readiness.json').read_text())
    if readiness.get('readiness')!='SOURCE_AND_INPUTS_VERIFIED':
        raise ValueError('Verified source/input prerequisite is missing')
    for row in readiness['environment']['packages']:
        if importlib.metadata.version(row['package']) != row['expected']:
            raise ValueError('Existing environment changed: '+row['package'])
    notebook=json.loads((paths['round01']/'outputs/notebook_support/notebook_session.json').read_text())
    if (notebook['status']!='NOTEBOOK_REVIEW_COMPLETE' or notebook['new_model_fits']!=0
            or notebook['chart_payloads_emitted']!=list(range(1,8))):
        raise ValueError('Completed notebook/kernel receipt not found')
    ref=json.loads((ROOT/'evidence/ROUND01_REFERENCE.json').read_text())
    checked(paths['round01']/'outputs/result.json',ref['round01_result_sha256'])
    checked(ROOT/'intent_features.py',ref['timing_code_sha256'])
    if sha(paths['round01']/'intent_features.py')!=ref['timing_code_sha256']:
        raise ValueError('Original timing implementation changed; frozen formulas must match')
    free=shutil.disk_usage(home).free/1024**3
    if free<6: raise ValueError(f'Need 6 GiB free for new evidence; available {free:.2f}. Do not delete data')
    sys.path.insert(0,str(repo/'src'))
    log('source_runtime_ready',commit=COMMIT,free_GiB=round(free,2),installs=0)
    return paths,ref


def protocol():
    p=json.loads((ROOT/'protocol.json').read_text())
    if (p['sessions']!=4096 or p['candidate_budget']!=400 or p['timing_features']!=24
            or p['maximum_new_models']!=12 or p['source_commit']!=COMMIT
            or p['selection_access'] is not False or p['evaluation_access'] is not False
            or p['competition_test_access'] is not False or p['role']!='fit'):
        raise ValueError('Preregistered protocol changed')
    return p


def checked_manifest(path):
    return json.loads(Path(path).read_text())


def source_contract(paths,ref):
    corpus=checked_manifest(paths['corpus']/'manifest.json')
    if (corpus['status']!='passed' or corpus['protocol']['history_end']!=HISTORY_END
            or corpus['protocol']['fit_end']!=FIT_END):
        raise ValueError('Wrong temporal corpus')
    for name in ('observed.parquet','queries.parquet'):
        checked(paths['corpus']/name,corpus['files'][name])
    cfg_path=paths['repo']/'configs/shared_feature_validation.json'
    cfg=checked_manifest(cfg_path)
    names=cfg['baseline_features']+cfg['added_features']
    if len(cfg['baseline_features'])!=102 or len(cfg['added_features'])!=32 or len(set(names))!=134:
        raise ValueError('Frozen shared schema is not 102+32')
    if set(names).intersection(TIMING_NAMES): raise ValueError('Timing names collide with control')
    smoke=checked_manifest(paths['repo']/'reports/research/early_shared_feature_smoke.json')
    gm={}
    for family in FAMILIES:
        path=paths['graphs']/family/'manifest.json'; m=checked_manifest(path)
        if (m['input_id']!=smoke['graphs'][family]['input_id'] or m['status']!='passed'
                or m['history_end']!=HISTORY_END or m['query_labels_used'] is not False
                or m['observed_history_max_ts']>=HISTORY_END or m['completed_parts']!=64):
            raise ValueError('Historical wide graph identity/cutoff mismatch')
        gm[family]=sha(path)
    m=checked_manifest(paths['retrieval']/'manifest.json')
    if (m['status']!='passed' or m['history_end']!=HISTORY_END or m['query_labels_used'] is not False
            or m['observed_history_max_ts']>=HISTORY_END):
        raise ValueError('Baseline retrieval temporal contract mismatch')
    contract={'commit':COMMIT,'protocol_sha256':sha(ROOT/'protocol.json'),
              'timing_sha256':sha(ROOT/'intent_features.py'),
              'runner_sha256':sha(Path(__file__)),'core_sha256':sha(ROOT/'core.py'),
              'corpus_manifest_sha256':sha(paths['corpus']/'manifest.json'),
              'query_sha256':corpus['files']['queries.parquet'],
              'observed_sha256':corpus['files']['observed.parquet'],
              'schema_sha256':sha(cfg_path),'graph_manifests':gm,
              'retrieval_manifest_sha256':sha(paths['retrieval']/'manifest.json'),
              'excluded_ids_sha256':array_digest(np.asarray(ref['excluded_session_ids'],dtype=np.int64)),
              'names':names+list(TIMING_NAMES)}
    return contract,cfg,corpus


def load_cohort(paths,ref,corpus):
    import polars as pl
    ledger=(pl.scan_parquet(paths['corpus']/'queries.parquet').filter(pl.col('split_role')=='fit').collect())
    click=ledger.filter(pl.col('objective')=='clicks').sort('session')
    all_ids=click['session'].to_numpy().astype(np.int64)
    old=np.asarray(ref['excluded_session_ids'],dtype=np.int64)
    old_manifest=checked_manifest(paths['repo']/'reports/research/early_shared_feature_scale.json')
    checked(paths['scale']/'fit_query_ledger.parquet',old_manifest['outputs']['query_ledger']['sha256'])
    actual_old=pl.read_parquet(paths['scale']/'fit_query_ledger.parquet')['session'].sort().to_numpy()
    if not np.array_equal(old,actual_old) or len(old)!=1024:
        raise ValueError('Exclusion set differs from prior cohort')
    ids=choose_cohort(all_ids,old)
    # No target-derived eligibility. Outcomes are consulted only after ID selection is frozen.
    sub=ledger.filter(pl.col('session').is_in(ids.tolist()))
    rows={}
    for row in sub.iter_rows(named=True): rows.setdefault(int(row['session']),[]).append(row)
    if len(rows)!=4096 or sub.height!=4096*3: raise ValueError('Incomplete objective ledger')
    q=[]; first=[]; count=[]; last=[]; tc=[]; den=[]
    common=('query_ts','first_ts','observed_events','observed_last_index','period_end')
    for s in ids:
        rr=rows[int(s)]
        if len(rr)!=3 or {r['objective'] for r in rr}!=set(OBJECTIVES):
            raise ValueError('Duplicate/missing objective metadata')
        by={r['objective']:r for r in rr}; a=by['clicks']
        if any(any(r[key]!=a[key] for key in common) for r in rr):
            raise ValueError('Inconsistent query metadata across objectives')
        if not HISTORY_END<=a['first_ts']<=a['query_ts']<FIT_END or a['period_end']!=FIT_END:
            raise ValueError('Chosen session crosses fitting role')
        truth=[by[o]['true_items'] for o in OBJECTIVES]
        denom=[by[o]['recall_denominator'] for o in OBJECTIVES]
        if truth[0]>1 or any(t<0 for t in truth) or denom!=[min(20,t) for t in truth]:
            raise ValueError('Incomplete target denominators')
        q.append(a['query_ts']); first.append(a['first_ts']); count.append(a['observed_events'])
        last.append(a['observed_last_index']);tc.append(truth);den.append(denom)
    payload={'ids':ids.tolist(),'query_ts':q,'first_ts':first,'observed_events':count,
             'observed_last_index':last,'true_counts':tc,'denominator':den,
             'period_end':FIT_END,'excluded_overlap':0,'selection':'ID hash only'}
    save_json(OUT/'cohort.json',payload)
    return payload,ledger


def observed_prefixes(paths,ledger,ids):
    import polars as pl
    from otto_recsys.research.features import Prefix
    df=(pl.scan_parquet(paths['corpus']/'observed.parquet')
        .filter((pl.col('split_role')=='fit') & pl.col('session').is_in(list(map(int,ids))))
        .sort('session','event_index').collect())
    meta={int(r['session']):r for r in ledger.filter(
        (pl.col('objective')=='clicks')&pl.col('session').is_in(list(map(int,ids)))).iter_rows(named=True)}
    result={}
    for g in df.partition_by('session',maintain_order=True):
        s=int(g['session'][0]); a=g['aid'].to_numpy().astype(np.int64)
        t=g['ts'].to_numpy().astype(np.int64); k=g['event_type'].to_numpy().astype(np.int64)
        validate_prefix(a,t,k,g['event_index'].to_numpy(),meta[s],HISTORY_END,FIT_END)
        result[s]=Prefix(s,a,t,k)
    if set(result)!=set(map(int,ids)): raise ValueError('Missing observed prefix')
    return result


def verify_chunks(manifest,contract,ids):
    cid=sha(OUT/'feature_contract.json')
    if manifest['contract_id']!=cid or manifest['sessions']!=4096:
        raise ValueError('Feature manifest lineage mismatch')
    start=0
    for row in manifest['parts']:
        if row['start']!=start or row['end']!=start+64: raise ValueError('Feature chunk discontinuity')
        path=OUT/row['path']
        if path.parent!=OUT/'features': raise ValueError('Unsafe feature chunk path')
        checked(path,row['sha256'])
        with np.load(path,allow_pickle=False) as z:
            if (not np.array_equal(z['ids'],ids[start:start+64]) or z['x'].shape!=(64,400,158)
                    or z['aids'].shape!=(64,400) or not np.isfinite(z['x']).all()):
                raise ValueError('Feature chunk arrays invalid')
        start+=64
    if start!=4096 or manifest['names']!=contract['names']: raise ValueError('Feature chunks incomplete')


def build_features(paths,ref):
    import polars as pl
    contract,cfg,corpus=source_contract(paths,ref)
    cohort,ledger=load_cohort(paths,ref,corpus)
    ids=np.asarray(cohort['ids'],dtype=np.int64)
    contract['cohort_sha256']=sha(OUT/'cohort.json')
    save_json(OUT/'feature_contract.json',contract)
    cid=sha(OUT/'feature_contract.json')
    if (OUT/'feature_manifest.json').exists():
        m=checked_manifest(OUT/'feature_manifest.json');verify_chunks(m,contract,ids)
        log('features_verified_reused',sessions=4096,graphs_loaded=0,new_features_materialized=0)
        return {'status':'ROUND02_FEATURES_READY','new_sessions':0,'reused_sessions':4096}
    PROGRESS.update(stage='load_certified_graphs',completed=0,total=4096)
    helper=module_from_file('otto_scale_pinned',paths['repo']/'scripts/run_shared_feature_scale.py')
    from otto_recsys.research.features import FeatureEngine
    from otto_recsys.research.graph_signals import GraphSignals
    from otto_recsys.research.domain_features import NormalizedGraphSignals
    objects={f:GraphSignals(paths['graphs']/f,f) for f in FAMILIES}
    graphs={f:g.graphs for f,g in objects.items()}
    normalized=NormalizedGraphSignals(objects); engine=FeatureEngine(paths['retrieval'])
    if engine.cutoff!=HISTORY_END: raise ValueError('Unexpected feature engine cutoff')
    original_ids=np.asarray(ref['excluded_session_ids'][:8],dtype=np.int64)
    prefixes=observed_prefixes(paths,ledger,np.r_[ids,original_ids])
    base_names=tuple(cfg['baseline_features']); added_names=tuple(cfg['added_features'])
    def one(session):
        prefix=prefixes[int(session)]
        candidates,base,_,_=helper.feature_matrix(engine,normalized,prefix,base_names,added_names,None)
        if candidates.aid.shape!=(400,) or np.unique(candidates.aid).size!=400:
            raise ValueError('Candidate pool is not 400 unique items')
        extra=transform(prefix.aid,prefix.ts,prefix.kind,candidates.aid,graphs,HISTORY_END)[:,list(TIMING)]
        full=np.column_stack((base,extra)).astype(np.float32)
        if full.shape!=(400,158) or not np.isfinite(full).all(): raise ValueError('Invalid reconstructed features')
        return candidates.aid.astype(np.int64),full
    PROGRESS['stage']='8_prior_session_compatibility'
    old_report=checked_manifest(paths['repo']/'reports/research/early_shared_feature_scale.json')
    part=old_report['outputs']['parts'][0]
    checked(paths['scale']/part['part'],part['sha256'])
    oldframe=pl.read_parquet(paths['scale']/part['part']).filter(pl.col('session').is_in(original_ids.tolist()))
    fm=checked_manifest(paths['round01']/'outputs/feature_manifest.json')
    oldpiece=fm['parts'][0];op=paths['round01']/'outputs'/oldpiece['path'];checked(op,oldpiece['sha256'])
    with np.load(op,allow_pickle=False) as z:
        oldids=z['ids']; oldtiming=z['x'][:,:,list(TIMING)].copy()
    for s in original_ids:
        a,x=one(s); row=oldframe.filter(pl.col('session')==int(s)).sort('candidate_position')
        j=np.flatnonzero(oldids==s)
        if row.height!=400 or len(j)!=1: raise ValueError('Old smoke IDs not available')
        if (not np.array_equal(a,row['aid'].to_numpy())
                or not np.array_equal(x[:,:134],row.select(contract['names'][:134]).to_numpy())
                or not np.array_equal(x[:,134:],oldtiming[int(j[0])])):
            raise ValueError('Reconstructed old control/timing changed; stop before new experiment')
    save_json(OUT/'compatibility.json',{'old_sessions':8,'control_and_timing_exact':True,
              'control_features':134,'timing_features':24,'contract_id':cid,'used_for_training':False})
    parts=[];new=0;reuse=0
    PROGRESS.update(stage='build_new_sessions',completed=0,total=4096)
    for begin in range(0,4096,64):
        before_unit();stop=begin+64
        path=OUT/'features'/f'part-{begin//64:03d}.npz';receipt_path=path.with_suffix('.json')
        if receipt_path.exists():
            r=checked_manifest(receipt_path)
            if r['contract_id']!=cid or r['start']!=begin or r['end']!=stop:
                raise ValueError('Feature checkpoint contract mismatch')
            checked(path,r['sha256']);reuse+=64
        else:
            if path.exists():raise ValueError('Orphan feature file preserved; return report')
            tick=time.monotonic(); values=[];aids=[]
            for i in range(begin,stop):
                a,x=one(ids[i])
                if i<16:
                    aa,xx=one(ids[i])
                    if not np.array_equal(a,aa) or not np.array_equal(x,xx):
                        raise ValueError('New-session numerical replay differs')
                aids.append(a);values.append(x)
            digest=save_arrays(path,ids=ids[begin:stop],aids=np.stack(aids),x=np.stack(values))
            r={'contract_id':cid,'start':begin,'end':stop,'sha256':digest,
               'path':str(path.relative_to(OUT)),'rows':64*400,'columns':158}
            save_json(receipt_path,r);new+=64
            log('feature_chunk_saved',completed=stop,total=4096,seconds=round(time.monotonic()-tick,3))
        parts.append(r); PROGRESS['completed']=stop
    m={'status':'ROUND02_FEATURES_READY','contract_id':cid,'sessions':4096,'rows':4096*400,
       'names':contract['names'],'parts':parts,'new_timing_features':24,
       'prior_session_numerical_parity':8,'new_session_numerical_replay':16,
       'cohort_overlap_with_round01':0,'graph_rebuilds':0,'candidate_policy_changed':False,
       'selection_access':False,'evaluation_access':False,'model_fits':0}
    save_json(OUT/'feature_manifest.json',m)
    return {'status':m['status'],'new_sessions':new,'reused_sessions':reuse}


def labels_for(paths,ids):
    import polars as pl
    m=checked_manifest(paths['corpus']/'manifest.json')
    checked(paths['corpus']/'labels.parquet',m['files']['labels.parquet'])
    df=(pl.scan_parquet(paths['corpus']/'labels.parquet').filter(
        (pl.col('split_role')=='fit')&pl.col('session').is_in(ids.tolist())).collect())
    labels={}
    for s,a,o,t,ix in df.select('session','aid','objective','label_ts','label_event_index').iter_rows():
        if o not in OBJECTIVES: raise ValueError('Unknown label objective')
        labels.setdefault(int(s),[]).append((int(a),OBJECTIVES.index(o),int(t),int(ix)))
    return labels,m['files']['labels.parquet']


def model_or_reuse(directory,contract,x,y,groups,valid,names,params,rounds=150):
    import lightgbm as lgb
    model_path=directory/'model.txt';receipt=directory/'receipt.json'
    if receipt.exists():
        r=checked_manifest(receipt)
        if r['contract']!=contract:raise ValueError('Model checkpoint does not match new replication')
        checked(model_path,r['model_sha256']);model=lgb.Booster(model_file=str(model_path));new=0
    else:
        if model_path.exists():raise ValueError('Orphan native model preserved; no automatic refit')
        if sum(groups)!=len(y) or len(y)!=len(x) or int(y.sum())<5:
            raise ValueError('Insufficient positives or malformed ranking groups')
        model=lgb.train(params,lgb.Dataset(x,label=y,group=groups,feature_name=names),num_boost_round=rounds)
        write_once(model_path,model.model_to_string().encode())
        save_json(receipt,{'contract':contract,'model_sha256':sha(model_path)});new=1
    if model.feature_name()!=names:raise ValueError('Native feature order mismatch')
    p=np.asarray(model.predict(valid,num_threads=4),dtype=np.float64)
    loaded=lgb.Booster(model_file=str(model_path))
    if not np.array_equal(p,np.asarray(loaded.predict(valid,num_threads=4),dtype=np.float64)):
        raise ValueError('Native reload changes predictions')
    return p,new


def screen(paths,ref):
    p=protocol(); c,cfg,cm=source_contract(paths,ref)
    cohort=checked_manifest(OUT/'cohort.json');c['cohort_sha256']=sha(OUT/'cohort.json')
    if checked_manifest(OUT/'feature_contract.json')!=c:raise ValueError('Feature sources changed')
    ids=np.asarray(cohort['ids'],dtype=np.int64)
    if len(ids)!=4096 or np.intersect1d(ids,ref['excluded_session_ids']).size:
        raise ValueError('New cohort overlaps pilot')
    m=checked_manifest(OUT/'feature_manifest.json');verify_chunks(m,c,ids)
    if (OUT/'result.json').exists():
        saved_study=checked_manifest(OUT/'screen_contract.json')
        checked(paths['corpus']/'labels.parquet',saved_study['labels_sha256'])
        from report import validate_saved_result
        validate_saved_result(OUT)
        existing=checked_manifest(OUT/'result.json')
        for relative,digest in existing['model_inventory'].items():checked(OUT/relative,digest)
        if len(existing['model_inventory'])!=12:raise ValueError('Completed model inventory incomplete')
        log('completed_screen_reused',new_model_fits=0)
        return {'status':'ROUND02_SCREEN_COMPLETED','new_model_fits':0,'models_reused':12}
    arrays=[];aa=[]
    for part in m['parts']:
        with np.load(OUT/part['path'],allow_pickle=False) as z:
            arrays.append(z['x']);aa.append(z['aids'])
    x=np.concatenate(arrays);aids=np.concatenate(aa);del arrays,aa;gc.collect()
    q=np.asarray(cohort['query_ts'],dtype=np.int64);last=np.asarray(cohort['observed_last_index'],dtype=np.int64)
    tc=np.asarray(cohort['true_counts'],dtype=np.int64);den=np.asarray(cohort['denominator'],dtype=np.int64)
    if not np.array_equal(den,np.minimum(tc,20)):raise ValueError('Target denominator changed')
    labels,labelsha=labels_for(paths,ids)
    truth=target_arrays(ids,aids,q,last,tc,labels,FIT_END)
    folds=forward_folds(ids,q)
    for fold in folds:
        if den[fold['valid'],2].sum()<p['minimum_validation_order_denominator_per_fold']:
            raise ValueError('Insufficient validation order support; no label-driven cohort resampling')
    study={'feature_contract_id':m['contract_id'],'labels_sha256':labelsha,
           'params':p['model_params'],'rounds':p['rounds'],'model_seed':p['model_seed'],
           'protocol_sha256':sha(ROOT/'protocol.json'),'cohort_sha256':sha(OUT/'cohort.json'),
           'runner_sha256':sha(Path(__file__)),'core_sha256':sha(ROOT/'core.py')}
    save_json(OUT/'screen_contract.json',study)
    from otto_recsys.research.dataset import sampled_rows
    schemas={'control_shared':c['names'][:134],'plus_timing':c['names']}
    hit_rows={name:[] for name in schemas};all_den=[];fids=[];eval_ids=[];records=[];diagnostics=[]
    counts={'new_model_fits':0,'models_reused':0}
    PROGRESS.update(stage='matched_control_and_timing',completed=0,total=12)
    for fi,fold in enumerate(folds):
        tr,va=fold['train'],fold['valid'];d=den[va]
        train_truth=target_arrays(ids[tr],aids[tr],q[tr],last[tr],tc[tr],labels,FIT_END,cutoff=fold['cutoff'])
        selected=[sampled_rows(train_truth[j],aids[i],int(ids[i]),60,20260908) for j,i in enumerate(tr)]
        groups=[len(s) for s in selected]
        xt=np.concatenate([x[i,s] for i,s in zip(tr,selected,strict=True)])
        yy=np.concatenate([train_truth[j,s] for j,s in enumerate(selected)])
        if (yy.sum(axis=0)<5).any():raise ValueError('Insufficient positive training examples')
        fold_record={'fold':fi,'train_sessions':len(tr),'valid_sessions':len(va),
                     'purged_sessions':int(fold['purged']),'cutoff_ms':fold['cutoff'],
                     'latest_training_query_ms':int(q[tr].max()),'embargo_hours':6,
                     'censored_targets_before_negative_sampling':True,
                     'positive_training_rows':yy.sum(axis=0).tolist(),
                     'training_ids':ids[tr].tolist(),'validation_ids':ids[va].tolist(),
                     'validation_denominators':d.sum(axis=0).tolist(),
                     'sampled_row_ids_sha256':array_digest(np.concatenate(selected)),
                     'shared_between_arms':True}
        save_json(OUT/'folds'/f'fold-{fi}.json',fold_record)
        sample=np.sort(np.random.default_rng(20260911).choice(tr,min(64,len(tr)),False))
        dx=x[sample].reshape(-1,158).astype(np.float64);std=dx.std(axis=0)
        centered=(dx-dx.mean(axis=0))/np.where(std>0,std,1)
        corr=np.abs(centered[:,134:].T@centered[:,:134]/len(centered))
        for j,name in enumerate(TIMING_NAMES):
            diagnostics.append({'fold':fi,'name':name,'training_sessions_sampled':len(sample),
                    'nonzero_fraction':float((dx[:,134+j]!=0).mean()),'std':float(std[134+j]),
                    'max_abs_corr_control':float(corr[j].max()),'automatic_selection':False})
        del dx,centered,corr
        validation=x[va].reshape(-1,158)
        for arm,names in schemas.items():
            hits=np.zeros((len(va),3),dtype=np.int64);width=len(names)
            for j,o in enumerate(OBJECTIVES):
                before_unit()
                contract={'study':study,'fold':fi,'arm':arm,'objective':o,'names':names,
                          'fold_sha256':sha(OUT/'folds'/f'fold-{fi}.json'),
                          'training_x_sha256':array_digest(xt[:,:width]),
                          'training_y_sha256':array_digest(yy[:,j]),
                          'validation_x_sha256':array_digest(validation[:,:width]),
                          'groups_sha256':array_digest(np.asarray(groups,dtype=np.int64))}
                prediction,n=model_or_reuse(OUT/'models'/f'fold-{fi}'/arm/o,contract,
                                      xt[:,:width],yy[:,j],groups,validation[:,:width],names,p['model_params'])
                counts['new_model_fits']+=n;counts['models_reused']+=1-n
                hits[:,j]=hits_at20(prediction.reshape(-1,400),aids[va],truth[va,:,j])
                PROGRESS['completed']+=1
                log('model_checkpoint',fold=fi,arm=arm,objective=o,new_fit=n,completed=PROGRESS['completed'],total=12)
            stats=pooled(hits,d);record={'fold':fi,'arm':arm,'features':width,**stats}
            save_json(OUT/'arm_reports'/f'fold-{fi}-{arm}.json',{
                'session_ids':ids[va].tolist(),'hits_by_session':hits.tolist(),'denominators':d.tolist(),'metrics':stats})
            records.append(record);hit_rows[arm].append(hits)
        all_den.append(d);eval_ids.append(ids[va]);fids.extend([fi]*len(va))
        del xt,yy,validation,train_truth;gc.collect()
    h={name:np.concatenate(v) for name,v in hit_rows.items()}
    d=np.concatenate(all_den);fold_ids=np.asarray(fids,dtype=np.int8)
    aggregate={name:pooled(v,d) for name,v in h.items()}
    gain=aggregate['plus_timing']['weighted_recall_at_20']-aggregate['control_shared']['weighted_recall_at_20']
    fg=[pooled(hit_rows['plus_timing'][i],all_den[i])['weighted_recall_at_20']-
        pooled(hit_rows['control_shared'][i],all_den[i])['weighted_recall_at_20'] for i in range(2)]
    order_gain=aggregate['plus_timing']['recall']['orders']-aggregate['control_shared']['recall']['orders']
    ci=paired_interval(h['plus_timing']-h['control_shared'],d,fold_ids)
    save_arrays(OUT/'evaluation_statistics.npz',ids=np.concatenate(eval_ids),folds=fold_ids,
                denominator=d,**h)
    save_json(OUT/'diagnostics.json',diagnostics)
    inventory={str(path.relative_to(OUT)):sha(path) for path in sorted((OUT/'models').rglob('model.txt'))}
    if len(inventory)!=12:raise ValueError('Expected exactly 12 native models')
    total=d.sum(axis=0);hit_delta=h['plus_timing'].sum(axis=0)-h['control_shared'].sum(axis=0)
    result={'status':'ROUND02_SCREEN_COMPLETED','source_commit':COMMIT,'sessions':4096,
          'validation_sessions':2048,'round01_session_overlap':0,'control_features':134,
          'timing_features':24,'total_timing_arm_features':158,'arms':aggregate,'fold_results':records,
          'comparison':{'gain':gain,'fold_gains':fg,'order_gain':order_gain,**ci,
                       'net_extra_hits':dict(zip(OBJECTIVES,map(int,hit_delta),strict=True)),
                       'weighted_contributions':dict(zip(OBJECTIVES,map(float,hit_delta/total*WEIGHTS),strict=True)),
                       'one_order_hit_weight':float(.6/total[2]),
                       'gain_less_one_order_hit':float(gain-.6/total[2]),
                       'decision':decision(gain,fg,order_gain,ci['descriptive_95_interval'])},
          'statistics_sha256':sha(OUT/'evaluation_statistics.npz'),'model_inventory':inventory,
          'native_reload_prediction_parity':True,'model_count':12,'feature_retention_decisions':0,
          'selection_access':False,'evaluation_access':False,'competition_test_access':False,
          'graph_rebuilds':0,'candidate_policy_changed':False,'algorithm_search':False,
          'feature_engineering_complete':False,'limitations':p['limitations'],
          'metric':'pooled weighted Recall@20 on new fitting-session cohort; NOT a Kaggle submission score'}
    save_json(OUT/'result.json',result)
    return {'status':result['status'],**counts,'gain':gain}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['features','screen'])
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    cap=protocol()['work_seconds'][args.phase]
    global DEADLINE
    start=time.monotonic();DEADLINE=start+cap
    stop=threading.Event()
    def beat():
        while not stop.wait(15):log('heartbeat',elapsed_seconds=round(time.monotonic()-start,1),**PROGRESS)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    # Signal catches long Python work; launcher also terminates long native calls.
    def expired(signum,frame):
        del signum,frame
        raise TimeoutError('Phase work deadline reached; completed checkpoints preserved')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cap)
    code=0;details={}
    try:
        with (OUT/'.experiment.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            paths,ref=check_runtime(Path.home())
            details=build_features(paths,ref) if args.phase=='features' else screen(paths,ref)
    except PausedCheckpointed as e:
        code=75;details={'status':'PAUSED_CHECKPOINTED','reason':str(e),**PROGRESS}
    except Exception as e:
        import traceback
        code=2;details={'status':'STOPPED_REVIEW_REQUIRED','error':f'{type(e).__name__}: {e}',
                        'traceback':traceback.format_exc(limit=8),**PROGRESS}
    finally:
        signal.alarm(0);stop.set();thread.join(timeout=1)
        details.update(phase=args.phase,exit_code=code,elapsed_seconds=time.monotonic()-start,
                       peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save_json(OUT/'runs'/f'{args.phase}-{time.time_ns()}.json',details)
    log('phase_finished',**details)
    print('RESULT: '+details['status'],flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
