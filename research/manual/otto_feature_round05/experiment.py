"""Manual matched feature experiment. Existing data/models and Git checkout are read-only."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from core import (OBJECTIVES, WEIGHTS, array_digest, checked, encode, forward_folds,
                  hits_at20, paired_interval, pooled, save_arrays, save_json, sha, target_arrays)
from transition_features import NAMES, COLLAPSED_NAMES, anchors, request_keys, transform
from transition_index import build as build_index, load as load_index, IndexPause
FAMILIES = ('symmetric', 'forward')

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'
COMMIT = '2638faa34afa427ed4ac4e92bba04deda57c688e'
HISTORY_END, FIT_END = 1660687200000, 1660946400000
PROGRESS = {'stage':'preflight', 'completed':0, 'total':0}
DEADLINE = float('inf')
ARMS = ('control_shared', 'action_pairs', 'collapsed_action_ablation')

class PausedCheckpointed(Exception):
    pass

def log(event, **values):
    print(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'event':event, **values}, sort_keys=True), flush=True)

def before_unit():
    if time.monotonic() > DEADLINE-20:
        raise PausedCheckpointed('Work budget nearly reached; completed units preserved')

def load_json(path):
    return json.loads(Path(path).read_text())

def safe_path(root, rel):
    p = root/rel
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe evidence path')
    return p

def load_prior_module(path):
    spec = importlib.util.spec_from_file_location('round02_reused_helpers', path)
    if spec is None or spec.loader is None:
        raise ValueError('Previous helper unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def protocol():
    p = load_json(ROOT/'protocol.json')
    expected = {'sessions':4096, 'candidate_budget':400, 'added_features':27,
                'maximum_new_models':12, 'control_models_refit':0,
                'source_commit':COMMIT, 'role':'fit', 'selection_access':False,
                'evaluation_access':False, 'competition_test_access':False}
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('transition_source_partitions')!=16 or p.get('transition_maximum_gap_ms')!=1800000 or p.get('source_cutoff_ms')!=HISTORY_END or p.get('smoothing')!=20.0 or p.get('ablation_features')!=9:
        raise ValueError('Frozen Round05 protocol differs')
    return p

def prepare_context(home):
    import shutil
    previous=home/'otto_feature_round02'; ref=load_json(ROOT/'evidence/ROUND02_REFERENCE.json')
    for rel,digest in ref['round02_files'].items():
        checked(safe_path(previous,rel),digest)
    prior=load_prior_module(previous/'replication.py')
    paths=prior.paths_for(home); repo=paths['repo']
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],timeout=15,text=True).strip()
    if git('rev-parse','HEAD')!=COMMIT or git('branch','--show-current')!='main' or git('status','--porcelain'):
        raise ValueError('Require the unchanged clean reviewed main; preserve changes, do not reset')
    if sys.version_info[:2]!=(3,13):
        raise ValueError('Use the existing ML .venv Python 3.13 through launch.py')
    ready=load_json(home/'otto_manual_workspace/reports/readiness.json')
    if ready.get('readiness')!='SOURCE_AND_INPUTS_VERIFIED':
        raise ValueError('Verified setup receipt missing')
    for row in ready['environment']['packages']:
        if importlib.metadata.version(row['package'])!=row['expected']:
            raise ValueError('Existing package version changed: '+row['package'])
    if shutil.disk_usage(home).free < 10*1024**3:
        raise ValueError('Need 10 GiB free for checkpoints; do not delete data')
    sys.path.insert(0,str(repo/'src'))
    old=previous/'outputs'; feature=load_json(old/'feature_contract.json')
    cohort=load_json(old/'cohort.json'); ids=np.asarray(cohort['ids'],dtype=np.int64)
    if len(ids)!=4096 or np.unique(ids).size!=4096 or not np.array_equal(ids,np.sort(ids)):
        raise ValueError('Saved fitting cohort changed')
    p=protocol(); oldp=load_json(previous/'protocol.json')
    if p['model_params']!=oldp['model_params'] or p['rounds']!=oldp['rounds']:
        raise ValueError('Not a matched model configuration')
    if p['negative_budget']!=oldp['negative_budget'] or p['negative_sampling_seed']!=oldp['negative_sampling_seed']:
        raise ValueError('Not a matched negative-sampling policy')
    cm=load_json(paths['corpus']/'manifest.json')
    checked(paths['corpus']/'manifest.json',feature['corpus_manifest_sha256'])
    if cm['protocol']['history_end']!=HISTORY_END or cm['protocol']['fit_end']!=FIT_END:
        raise ValueError('Wrong temporal input')
    for name in ('observed.parquet','queries.parquet'):
        checked(paths['corpus']/name,cm['files'][name])
    for family in FAMILIES:
        mp=paths['graphs']/family/'manifest.json';checked(mp,feature['graph_manifests'][family]);m=load_json(mp)
        if (m['status']!='passed' or m['history_end']!=HISTORY_END or m['query_labels_used'] is not False
                or m['observed_history_max_ts']>=HISTORY_END or m['completed_parts']!=64):
            raise ValueError('Historical graph provenance mismatch')
    manifest=load_json(old/'feature_manifest.json')
    if manifest['contract_id']!=sha(old/'feature_contract.json') or manifest['names']!=feature['names']:
        raise ValueError('Saved feature contract mismatch')
    base_names=feature['names'][:134]
    if len(base_names)!=134 or len(set(base_names+list(NAMES)))!=161:
        raise ValueError('Control/added feature schema mismatch')
    controls={k:v for k,v in ref['result']['model_inventory'].items() if '/control_shared/' in k}
    if len(controls)!=6:raise ValueError('Six previously fitted control models required')
    for rel,digest in controls.items():checked(safe_path(old,rel),digest)
    return {'paths':paths,'old':old,'prior':prior,'ref':ref,'cohort':cohort,'ids':ids,
            'manifest':manifest,'base_names':base_names,'feature':feature,'protocol':p,'controls':controls}

def prior_chunks(ctx):
    start=0
    for row in ctx['manifest']['parts']:
        if row['start']!=start or row['end']!=start+64:
            raise ValueError('Prior feature partition gap')
        path=safe_path(ctx['old'],row['path']);checked(path,row['sha256'])
        with np.load(path,allow_pickle=False) as z:
            x=z['x'];ids=z['ids'];aids=z['aids']
        if (x.shape!=(64,400,158) or x.dtype!=np.float32 or aids.shape!=(64,400)
                or not np.isfinite(x).all() or not np.array_equal(ids,ctx['ids'][start:start+64])
                or any(len(np.unique(a))!=400 for a in aids)):
            raise ValueError('Previous complete feature/candidate chunk invalid')
        yield row,ids,aids,x[:,:,:134].copy()
        start+=64
    if start!=4096:raise ValueError('Incomplete prior feature cohort')

def query_anchors(ctx):
    import polars as pl
    from core import validate_prefix
    ids=ctx['ids'];co=ctx['cohort']
    path=ctx['paths']['corpus']/'observed.parquet'
    frame=(pl.scan_parquet(path).filter((pl.col('split_role')=='fit') & pl.col('session').is_in(ids.tolist()))
           .select('session','aid','ts','event_type','event_index').collect().sort('session','event_index'))
    groups={}
    for key,g in frame.partition_by('session',as_dict=True).items():
        sid=int(key[0] if isinstance(key,tuple) else key);groups[sid]=g
    if set(groups)!=set(map(int,ids)):raise ValueError('Complete observed fitting sessions differ')
    aa=[]
    for i,sid in enumerate(ids):
        g=groups[int(sid)];a=g['aid'].to_numpy();ts=g['ts'].to_numpy();k=g['event_type'].to_numpy();ix=g['event_index'].to_numpy()
        ledger={key:co[key][i] for key in ('first_ts','query_ts','observed_events','observed_last_index')}
        ledger['period_end']=FIT_END
        validate_prefix(a,ts,k,ix,ledger,HISTORY_END,FIT_END)
        aa.append(anchors(a,k))
    return np.stack(aa)


def index_phase(ctx):
    import polars as pl
    from backend_smoke import run as smoke_backend
    backend = smoke_backend()
    log('installed_backend_smoke_passed', **backend)
    previous=ctx['old'];paths=ctx['paths'];retrieval=paths['retrieval']
    mp=retrieval/'manifest.json';checked(mp,ctx['feature']['retrieval_manifest_sha256']);m=load_json(mp)
    if m['status']!='passed' or m['history_end']!=HISTORY_END or m['query_labels_used'] is not False or m['observed_history_max_ts']>=HISTORY_END:
        raise ValueError('Wrong historical retrieval source')
    tail=retrieval/'history_tail.parquet';PROGRESS.update(stage='verify_historical_tail_hash',completed=0,total=16)
    checked(tail,m['files']['history_tail.parquet'])
    aa=query_anchors(ctx);requests=[]
    for row,ids,cc,_ in prior_chunks(ctx):
        requests.append(request_keys(aa[row['start']:row['end']],cc))
    keys=np.unique(np.concatenate(requests));del requests
    if len(keys)>10000000:raise ValueError('Pair-request resource cap exceeded; do not raise the cap')
    legacy=paths['scale']/'fit_query_ledger.parquet'
    reference=load_json(paths['repo']/'reports/research/early_shared_feature_scale.json')
    checked(legacy,reference['outputs']['query_ledger']['sha256'])
    oldids=pl.read_parquet(legacy,columns=['session'])['session'].to_numpy().astype(np.int64)
    excluded=np.unique(np.concatenate([ctx['ids'],oldids]))
    if len(oldids)!=1024 or len(excluded)!=5120:raise ValueError('Source-study exclusion identity differs')
    contract={'history_tail_sha256':m['files']['history_tail.parquet'],
        'retrieval_manifest_sha256':sha(mp),'source_cutoff_ms':HISTORY_END,
        'previous_feature_manifest_sha256':sha(previous/'feature_manifest.json'),
        'cohort_sha256':sha(previous/'cohort.json'),'request_sha256':array_digest(keys),
        'anchor_sha256':array_digest(aa),'excluded_sessions':excluded.tolist(),
        'index_code_sha256':sha(ROOT/'transition_index.py'),'formula_sha256':sha(ROOT/'transition_features.py'),
        'protocol_sha256':sha(ROOT/'protocol.json'),'backend_duckdb':importlib.metadata.version('duckdb'),
        'adjacent_event_index_only':True,'maximum_gap_ms':1800000,'session_unique_support':True,
        'outgoing_denominators_before_candidate_pair_filter':True,'source_is_retained_historical_tail':True}
    result=build_index(tail,OUT/'index',keys,excluded,contract,deadline=DEADLINE,progress=PROGRESS,log=log)
    summary={k:v for k,v in result.items() if k!='contract'}
    summary['index_manifest_sha256']=sha(OUT/'index/index_manifest.json')
    save_json(OUT/'index_summary.json',summary)
    return {'status':'ROUND05_INDEX_READY','completed_source_partitions':16,
            'request_pairs':len(keys),'study_sessions_excluded':5120,'new_model_fits':0}


def feature_identity(ctx):
    return {'previous_feature_manifest_sha256':sha(ctx['old']/'feature_manifest.json'),
            'cohort_sha256':sha(ctx['old']/'cohort.json'),
            'previous_feature_contract_sha256':sha(ctx['old']/'feature_contract.json'),
            'source_commit':COMMIT,'formula_sha256':sha(ROOT/'transition_features.py'),
            'index_manifest_sha256':sha(OUT/'index/index_manifest.json'),
            'runner_sha256':sha(Path(__file__)),'core_sha256':sha(ROOT/'core.py'),
            'protocol_sha256':sha(ROOT/'protocol.json'),'names':list(NAMES),
            'ablation_names':list(COLLAPSED_NAMES),
            'control_features_rebuilt':False,'previous_failed_additions_included':False}


def verify_new_chunk(path,ids,aids):
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids) or not np.array_equal(z['aids'],aids):
            raise ValueError('New transition features not aligned with certified candidates')
        typed,collapsed=z['typed'],z['collapsed']
    for a,n in ((typed,27),(collapsed,9)):
        if a.shape!=(len(ids),400,n) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('New transition feature chunk invalid')
    return typed,collapsed


def features(ctx):
    summary=load_json(OUT/'index_summary.json')
    checked(OUT/'index/index_manifest.json',summary['index_manifest_sha256'])
    contract=feature_identity(ctx);save_json(OUT/'feature_contract.json',contract);cid=sha(OUT/'feature_contract.json')
    if (OUT/'feature_manifest.json').exists():
        m=load_json(OUT/'feature_manifest.json')
        if m['contract_id']!=cid or len(m['parts'])!=64:raise ValueError('Incomplete transition feature manifest')
        for row,(_,ids,aids,_) in zip(m['parts'],prior_chunks(ctx),strict=True):
            checked(OUT/row['path'],row['sha256']);verify_new_chunk(OUT/row['path'],ids,aids)
        return {'status':'ROUND05_FEATURES_READY','new_sessions':0,'reused_sessions':4096}
    aa=query_anchors(ctx);PROGRESS.update(stage='load_transition_index',completed=0,total=4096)
    index=load_index(OUT/'index');parts=[];new_count=reused=0
    for i,(old,ids,aids,base) in enumerate(prior_chunks(ctx)):
        before_unit();path=safe_path(OUT,f'features/part-{i:03d}.npz');receipt=path.with_suffix('.json')
        unit={'contract_id':cid,'source_chunk_sha256':old['sha256'],'start':old['start'],'end':old['end']}
        if receipt.exists():
            saved=load_json(receipt)
            if saved['unit']!=unit:raise ValueError('Chunk contract mismatch')
            checked(path,saved['sha256']);verify_new_chunk(path,ids,aids);reused+=64
        else:
            if path.exists():raise ValueError('Orphan feature chunk preserved; return diagnostics')
            tt=[];cc=[]
            for j,s in enumerate(ids):
                anchor=aa[old['start']+j];x,y=transform(index,anchor,aids[j])
                if old['start']+j<16:
                    xx,yy=transform(index,anchor,aids[j])
                    if not np.array_equal(x,xx) or not np.array_equal(y,yy):raise ValueError('Numerical transition replay differs')
                tt.append(x);cc.append(y)
            typed=np.stack(tt);collapsed=np.stack(cc)
            hh=save_arrays(path,ids=ids,aids=aids,typed=typed,collapsed=collapsed)
            save_json(receipt,{'unit':unit,'sha256':hh});new_count+=64
        parts.append({'path':str(path.relative_to(OUT)),'sha256':sha(path),'start':old['start'],'end':old['end']})
        PROGRESS.update(stage='action_pair_features',completed=old['end'],total=4096)
        if (i+1)%4==0:log('feature_checkpoints',**PROGRESS)
    save_json(OUT/'feature_manifest.json',{'status':'ROUND05_FEATURES_READY','contract_id':cid,
        'sessions':4096,'rows':1638400,'names':list(NAMES),'ablation_names':list(COLLAPSED_NAMES),'parts':parts,
        'numeric_replay_sessions':16,'control_rebuilds':0,'old_graph_rebuilds':0,'model_fits':0})
    return {'status':'ROUND05_FEATURES_READY','new_sessions':new_count,'reused_sessions':reused}


def control_prediction(directory, expected_contract, xvalid, expected_hash, names):
    import lightgbm as lgb
    receipt=load_json(directory/'receipt.json')
    if receipt['contract']!=expected_contract:raise ValueError('Saved control was trained on a different contract')
    checked(directory/'model.txt',expected_hash)
    if receipt['model_sha256']!=expected_hash:raise ValueError('Control model receipt mismatch')
    model=lgb.Booster(model_file=str(directory/'model.txt'))
    if model.feature_name()!=names:raise ValueError('Control native feature order mismatch')
    prediction=np.asarray(model.predict(xvalid,num_threads=4),dtype=np.float64)
    if not np.isfinite(prediction).all():raise ValueError('Invalid control predictions')
    return prediction

def screen(ctx):
    if load_json(OUT/'feature_contract.json')!=feature_identity(ctx):raise ValueError('New feature sources changed')
    fm=load_json(OUT/'feature_manifest.json')
    if fm['contract_id']!=sha(OUT/'feature_contract.json') or len(fm['parts'])!=64:
        raise ValueError('New feature stage incomplete')
    p=ctx['protocol'];ids=ctx['ids'];co=ctx['cohort'];old=ctx['old'];names=ctx['base_names']
    base=np.empty((4096,400,134),dtype=np.float32);typed_features=np.empty((4096,400,27),dtype=np.float32);collapsed_features=np.empty((4096,400,9),dtype=np.float32)
    aids=np.empty((4096,400),dtype=np.int64)
    for new,(oldrow,chunk_ids,aa,xx) in zip(fm['parts'],prior_chunks(ctx),strict=True):
        s,e=oldrow['start'],oldrow['end']
        if new['start']!=s or new['end']!=e:raise ValueError('Chunk boundaries differ')
        path=safe_path(OUT,new['path']);checked(path,new['sha256'])
        base[s:e]=xx;aids[s:e]=aa;typed_features[s:e],collapsed_features[s:e]=verify_new_chunk(path,chunk_ids,aa)
    if (OUT/'result.json').exists():
        from report import validate_result
        result=validate_result(OUT)
        for rel,digest in result['model_inventory'].items():checked(safe_path(OUT,rel),digest)
        if len(result['model_inventory'])!=12:raise ValueError('Native inventory incomplete')
        return {'status':'ROUND05_SCREEN_COMPLETED','new_model_fits':0,'models_reused':12}
    q=np.asarray(co['query_ts'],np.int64);last=np.asarray(co['observed_last_index'],np.int64)
    tc=np.asarray(co['true_counts'],np.int64);den=np.asarray(co['denominator'],np.int64)
    if not np.array_equal(den,np.minimum(tc,20)):raise ValueError('Complete denominators changed')
    labels,labelsha=ctx['prior'].labels_for(ctx['paths'],ids)
    truth=target_arrays(ids,aids,q,last,tc,labels,FIT_END)
    folds=forward_folds(ids,q);study={'feature_contract_id':fm['contract_id'],'labels_sha256':labelsha,
        'prior_screen_contract_sha256':sha(old/'screen_contract.json'),'params':p['model_params'],
        'rounds':p['rounds'],'cohort_sha256':sha(old/'cohort.json'),'control_width':134}
    prior_study=load_json(old/'screen_contract.json')
    if prior_study['labels_sha256']!=labelsha:raise ValueError('Labels differ from control')
    save_json(OUT/'screen_contract.json',study)
    from otto_recsys.research.dataset import sampled_rows
    hit_lists={a:[] for a in ARMS};den_lists=[];id_lists=[];fold_labels=[];records=[];diags=[]
    counts={'new_model_fits':0,'models_reused':0,'control_models_replayed':0}
    PROGRESS.update(stage='verify_all_saved_controls',completed=0,total=6)
    prepared=[]
    for fi,fold in enumerate(folds):
        tr,va=fold['train'],fold['valid'];d=den[va]
        train_truth=target_arrays(ids[tr],aids[tr],q[tr],last[tr],tc[tr],labels,FIT_END,cutoff=fold['cutoff'])
        selected=[sampled_rows(train_truth[j],aids[i],int(ids[i]),60,20260908) for j,i in enumerate(tr)]
        groups=[len(s) for s in selected]
        yy=np.concatenate([train_truth[j,s] for j,s in enumerate(selected)])
        xt=np.concatenate([base[i,s] for i,s in zip(tr,selected,strict=True)])
        mt=np.concatenate([typed_features[i,s] for i,s in zip(tr,selected,strict=True)])
        st=np.concatenate([collapsed_features[i,s] for i,s in zip(tr,selected,strict=True)])
        xv=base[va].reshape(-1,134);mv=typed_features[va].reshape(-1,27);sv=collapsed_features[va].reshape(-1,9)
        previous_fold=load_json(old/'folds'/f'fold-{fi}.json')
        if (previous_fold['training_ids']!=ids[tr].tolist() or previous_fold['validation_ids']!=ids[va].tolist()
                or previous_fold['cutoff_ms']!=fold['cutoff']
                or previous_fold['sampled_row_ids_sha256']!=array_digest(np.concatenate(selected))):
            raise ValueError('Training, validation, embargo or negative rows differ from control')
        save_json(OUT/'folds'/f'fold-{fi}.json',previous_fold)
        control_hits=np.empty((len(va),3),dtype=np.int64)
        for j,o in enumerate(OBJECTIVES):
            before_unit()
            contract={'study':prior_study,'fold':fi,'arm':'control_shared','objective':o,'names':names,
                      'fold_sha256':sha(old/'folds'/f'fold-{fi}.json'),
                      'training_x_sha256':array_digest(xt),'training_y_sha256':array_digest(yy[:,j]),
                      'validation_x_sha256':array_digest(xv),'groups_sha256':array_digest(np.asarray(groups,dtype=np.int64))}
            rel=f'models/fold-{fi}/control_shared/{o}/model.txt'
            pred=control_prediction((old/rel).parent,contract,xv,ctx['controls'][rel],names)
            control_hits[:,j]=hits_at20(pred.reshape(-1,400),aids[va],truth[va,:,j])
            counts['control_models_replayed']+=1
            PROGRESS['completed']=counts['control_models_replayed']
        saved=load_json(old/'arm_reports'/f'fold-{fi}-control_shared.json')
        if (saved['session_ids']!=ids[va].tolist() or saved['hits_by_session']!=control_hits.tolist()
                or saved['denominators']!=d.tolist()):raise ValueError('Native control does not replay prior hits exactly')
        hit_lists['control_shared'].append(control_hits)
        records.append({'fold':fi,'arm':'control_shared','features':134,**pooled(control_hits,d)})
        save_json(OUT/'arm_reports'/f'fold-{fi}-control_shared.json',saved)
        prepared.append((fi,fold,tr,va,d,groups,yy,xt,mt,st,xv,mv,sv))
    # No challenger fit occurs until all six existing controls have passed replay.
    if counts['control_models_replayed']!=6:raise ValueError('Complete control replay gate required')
    log('all_six_controls_verified_before_new_fits')
    del train_truth
    PROGRESS.update(stage='matched_action_pair_screen',completed=0,total=12)
    for fi,fold,tr,va,d,groups,yy,xt,mt,st,xv,mv,sv in prepared:
        # Diagnostics sample only training sessions and retain every formula for the comparison.
        sample=np.sort(np.random.default_rng(20260911).choice(tr,min(64,len(tr)),False))
        dx=typed_features[sample].reshape(-1,27).astype(np.float64);bx=base[sample].reshape(-1,134).astype(np.float64)
        ds=dx.std(0);bs=bx.std(0)
        dz=(dx-dx.mean(0))/np.where(ds>0,ds,1);bz=(bx-bx.mean(0))/np.where(bs>0,bs,1)
        corr=np.abs(dz.T@bz/len(dx));mx=np.abs(dz.T@dz/len(dx));np.fill_diagonal(mx,0)
        for j,name in enumerate(NAMES):
            diags.append({'fold':fi,'name':name,'training_sessions_sampled':len(sample),
                'support_fraction':float((dx[:,j]!=0).mean()),'recent_positive_fraction':float((dx[:,j]>0).mean()),
                'std':float(ds[j]),'max_abs_corr_control':float(corr[j].max()),
                'max_abs_corr_added':float(mx[j].max()),'automatic_selection':False})
        del dx,bx,dz,bz,corr,mx
        for arm in ARMS[1:]:
            xtrain=np.concatenate([xt,mt if arm=='action_pairs' else st],axis=1)
            validation=np.concatenate([xv,mv if arm=='action_pairs' else sv],axis=1)
            allnames=names+list(NAMES if arm=='action_pairs' else COLLAPSED_NAMES);hits=np.zeros((len(va),3),dtype=np.int64)
            for j,o in enumerate(OBJECTIVES):
                before_unit()
                contract={'study':study,'fold':fi,'arm':arm,'objective':o,'names':allnames,
                          'fold_sha256':sha(OUT/'folds'/f'fold-{fi}.json'),
                          'training_x_sha256':array_digest(xtrain),'training_y_sha256':array_digest(yy[:,j]),
                          'validation_x_sha256':array_digest(validation),'groups_sha256':array_digest(np.asarray(groups,np.int64))}
                pred,n=ctx['prior'].model_or_reuse(OUT/'models'/f'fold-{fi}'/arm/o,contract,
                              xtrain,yy[:,j],groups,validation,allnames,p['model_params'],rounds=150)
                counts['new_model_fits']+=n;counts['models_reused']+=1-n
                hits[:,j]=hits_at20(pred.reshape(-1,400),aids[va],truth[va,:,j])
                PROGRESS['completed']+=1;log('model_checkpoint',fold=fi,arm=arm,objective=o,new_fit=n,**PROGRESS)
            metrics=pooled(hits,d)
            save_json(OUT/'arm_reports'/f'fold-{fi}-{arm}.json',{'session_ids':ids[va].tolist(),
                       'hits_by_session':hits.tolist(),'denominators':d.tolist(),'metrics':metrics})
            records.append({'fold':fi,'arm':arm,'features':len(allnames),**metrics});hit_lists[arm].append(hits)
            del xtrain,validation
        den_lists.append(d);id_lists.append(ids[va]);fold_labels.extend([fi]*len(va))
        prepared[fi]=None
        del xt,mt,st,xv,mv,sv,yy;gc.collect()
    h={a:np.concatenate(rows) for a,rows in hit_lists.items()};d=np.concatenate(den_lists)
    fs=np.asarray(fold_labels,dtype=np.int8);aggregate={a:pooled(v,d) for a,v in h.items()}
    comparisons={}
    from analysis_utils import compare
    for left,right in [('action_pairs','control_shared'),('collapsed_action_ablation','control_shared'),
                       ('action_pairs','collapsed_action_ablation')]:
        comparisons[left+'_minus_'+right]=compare(h[left],h[right],d,fs)
    save_arrays(OUT/'evaluation_statistics.npz',ids=np.concatenate(id_lists),folds=fs,denominator=d,**h)
    save_json(OUT/'diagnostics.json',diags)
    inv={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'models').rglob('model.txt'))}
    if len(inv)!=12:raise ValueError('Exactly 12 new experiment models required')
    result={'status':'ROUND05_SCREEN_COMPLETED','source_commit':COMMIT,'sessions':4096,'validation_sessions':2048,
            'control_features':134,'added_features':27,'model_count':12,'control_models_refit':0,
            'control_prediction_replay_exact':True,'native_reload_prediction_parity':True,
            'arms':aggregate,'fold_results':records,'comparisons':comparisons,
            'primary_comparison':'action_pairs_minus_control_shared','secondary_comparisons_are_not_advancement_gates':True,'model_inventory':inv,
            'statistics_sha256':sha(OUT/'evaluation_statistics.npz'),'feature_retention_decisions':0,
            'selection_access':False,'evaluation_access':False,'competition_test_access':False,
            'original_graph_rebuilds':0,'algorithm_search':False,'round02_timing_included':False,'round03_mass_included':False,
            'historical_transition_cutoff_ms':HISTORY_END,'study_sessions_excluded':5120,'full_outgoing_denominators':True,'old_graphs_rebuilt':0,'new_transition_index':True,
            'feature_engineering_complete':False,'control_recipe_unchanged':True,
            'limitations':['Reused Round02 fitting cohort; exploratory new-feature screen, not untouched holdout.',
              'Historical adjacent-event transitions are counted only inside the previously retained session tails before Aug16. This does not test all nonadjacent action-pair relationships.',
              'All current and earlier study sessions are excluded. Outgoing denominators include noncandidate destination items; no targets enter transition construction.',
              'Original graphs and candidates are unchanged. Primary adds 27 typed features; the action-collapsed ablation adds nine. This removes information and changes feature dimension.',
              'Candidate policy and model capacity held fixed; negative findings apply to this protocol.',
              'No new Kaggle predictions, submission, leaderboard score or feature promotion.']}
    save_json(OUT/'result.json',result)
    return {'status':result['status'],**counts,'primary_gain':comparisons[result['primary_comparison']]['gain']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['index','features','screen'])
    args=parser.parse_args();OUT.mkdir(exist_ok=True);cap=protocol()['work_seconds'][args.phase]
    global DEADLINE
    start=time.monotonic();DEADLINE=start+cap;stop=threading.Event()
    def beat():
        while not stop.wait(15):log('heartbeat',elapsed_seconds=round(time.monotonic()-start,1),**PROGRESS)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    def expired(signum,frame):
        raise TimeoutError('Phase deadline reached; completed checkpoints preserved')
    signal.signal(signal.SIGALRM,expired);signal.alarm(cap);code=0;details={}
    try:
        with (OUT/'.experiment.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            ctx=prepare_context(Path.home())
            details={'index':index_phase,'features':features,'screen':screen}[args.phase](ctx)
    except (PausedCheckpointed, IndexPause) as e:
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
    log('phase_finished',**details);print('RESULT: '+details['status'],flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
