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
from relative_features import NAMES, GLOBAL_NAMES, SIGNALS, SEEN_SIGNAL, transform, coverage_statistics
FAMILIES = ('symmetric', 'forward')

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'
COMMIT = '2638faa34afa427ed4ac4e92bba04deda57c688e'
HISTORY_END, FIT_END = 1660687200000, 1660946400000
PROGRESS = {'stage':'preflight', 'completed':0, 'total':0}
DEADLINE = float('inf')
ARMS = ('control_shared', 'repeat_context', 'global_context_ablation')

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
    expected = {'sessions':4096, 'candidate_budget':400, 'added_features':12,
                'ablation_features':12, 'maximum_new_models':12, 'control_models_refit':0,
                'source_commit':COMMIT, 'role':'fit', 'selection_access':False,
                'evaluation_access':False, 'competition_test_access':False,
                'new_indices':0, 'raw_json_scans':0, 'primary_arm':'repeat_context',
                'new_arm_names':['repeat_context','global_context_ablation']}
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen Round06 protocol differs')
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
    if len(base_names)!=134 or len(set(base_names+list(NAMES)))!=146:
        raise ValueError('Control/added feature schema mismatch')
    for name in ('result.json','diagnostics.json','evaluation_statistics.npz'):
        ev={'result.json':'ROUND05_RESULT.json','diagnostics.json':'ROUND05_DIAGNOSTICS.json',
            'evaluation_statistics.npz':'ROUND05_STATISTICS.npz'}[name]
        checked(home/'otto_feature_round05/outputs'/name,sha(ROOT/'evidence'/ev))
    if not set(SIGNALS + (SEEN_SIGNAL,)).issubset(base_names):
        raise ValueError('Six frozen affinity inputs and complete-prefix seen status required')
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
    if start!=len(ctx['ids']):raise ValueError('Incomplete prior feature cohort')

def feature_identity(ctx):
    return {'previous_feature_contract_sha256':sha(ctx['old']/'feature_contract.json'),
        'previous_feature_manifest_sha256':sha(ctx['old']/'feature_manifest.json'),
        'cohort_sha256':sha(ctx['old']/'cohort.json'),'names':list(NAMES),'ablation_names':list(GLOBAL_NAMES),
        'source_signals':list(SIGNALS),'repeat_membership_signal':SEEN_SIGNAL,'protocol_sha256':sha(ROOT/'protocol.json'),
        'runner_sha256':sha(ROOT/'experiment.py'),'formula_sha256':sha(ROOT/'relative_features.py'),
        'core_sha256':sha(ROOT/'core.py'),'source_commit':COMMIT,
        'control_features_rebuilt':False,'full_candidate_set_before_negative_sampling':True,
        'source_cutoff_ms':HISTORY_END,'round05_result_sha256':sha(ROOT/'evidence/ROUND05_RESULT.json')}


def verify_new_chunk(path,ids,aids):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'ids','aids','relative','global_context'}:
            raise ValueError('Wrong feature checkpoint schema')
        if not np.array_equal(z['ids'],ids) or not np.array_equal(z['aids'],aids):
            raise ValueError('Candidate/session alignment changed')
        xx, yy = z['relative'], z['global_context']
    if any(v.shape!=(len(ids),400,12) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in (xx,yy)):
        raise ValueError('Invalid new feature matrix')
    if (xx<0).any() or (xx>1).any() or (yy<0).any() or (yy>1).any():
        raise ValueError('Context transform is out of range')
    return xx,yy


def features(ctx):
    contract=feature_identity(ctx);save_json(OUT/'feature_contract.json',contract);cid=sha(OUT/'feature_contract.json')
    if (OUT/'feature_manifest.json').exists():
        m=load_json(OUT/'feature_manifest.json')
        if m['contract_id']!=cid or len(m['parts'])!=len(ctx['ids'])//64:raise ValueError('Incomplete feature manifest')
        for row,(_,ids,aids,_) in zip(m['parts'],prior_chunks(ctx),strict=True):
            checked(OUT/row['path'],row['sha256']);verify_new_chunk(OUT/row['path'],ids,aids)
        return {'status':'ROUND06_FEATURES_READY','new_sessions':0,'reused_sessions':len(ctx['ids'])}
    parts=[];new_count=reused=0
    for i,(old,ids,aids,base) in enumerate(prior_chunks(ctx)):
        before_unit();path=safe_path(OUT,f'features/part-{i:03d}.npz');receipt=path.with_suffix('.json')
        unit={'contract_id':cid,'source_chunk_sha256':old['sha256'],'start':old['start'],'end':old['end']}
        if receipt.exists():
            saved=load_json(receipt)
            if saved['unit']!=unit:raise ValueError('Feature chunk contract mismatch')
            checked(path,saved['sha256']);verify_new_chunk(path,ids,aids);reused+=64
        else:
            if path.exists():raise ValueError('Orphan checkpoint preserved; return diagnostics')
            tt=[];cc=[]
            for j,s in enumerate(ids):
                x,y=transform(base[j],ctx['base_names'],aids[j])
                if old['start']+j<16:
                    reverse=np.arange(399,-1,-1)
                    xx,yy=transform(base[j,reverse],ctx['base_names'],aids[j,reverse])
                    if not np.array_equal(x,xx[reverse]) or not np.array_equal(y,yy[reverse]):
                        raise ValueError('Candidate-order invariance failed')
                tt.append(x);cc.append(y)
            hh=save_arrays(path,ids=ids,aids=aids,relative=np.stack(tt),global_context=np.stack(cc))
            save_json(receipt,{'unit':unit,'sha256':hh});new_count+=64
        parts.append({'path':str(path.relative_to(OUT)),'sha256':sha(path),'start':old['start'],'end':old['end']})
        PROGRESS.update(stage='candidate_repeat_context',completed=old['end'],total=4096)
        if (i+1)%4==0:log('feature_checkpoints',**PROGRESS)
    save_json(OUT/'feature_manifest.json',{'status':'ROUND06_FEATURES_READY','contract_id':cid,
        'sessions':len(ctx['ids']),'rows':len(ctx['ids'])*400,'names':list(NAMES),'ablation_names':list(GLOBAL_NAMES),'parts':parts,
        'candidate_permutation_replay_sessions':16,'control_rebuilds':0,'raw_source_scans':0,'model_fits':0})
    return {'status':'ROUND06_FEATURES_READY','new_sessions':new_count,'reused_sessions':reused}


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
    if fm['contract_id']!=sha(OUT/'feature_contract.json') or len(fm['parts'])!=len(ctx['ids'])//64:
        raise ValueError('New feature stage incomplete')
    p=ctx['protocol'];ids=ctx['ids'];co=ctx['cohort'];old=ctx['old'];names=ctx['base_names']
    base=np.empty((len(ids),400,134),dtype=np.float32);relative_features=np.empty((len(ids),400,12),dtype=np.float32);global_features=np.empty((len(ids),400,12),dtype=np.float32)
    aids=np.empty((len(ids),400),dtype=np.int64)
    for new,(oldrow,chunk_ids,aa,xx) in zip(fm['parts'],prior_chunks(ctx),strict=True):
        s,e=oldrow['start'],oldrow['end']
        if new['start']!=s or new['end']!=e:raise ValueError('Chunk boundaries differ')
        path=safe_path(OUT,new['path']);checked(path,new['sha256'])
        base[s:e]=xx;aids[s:e]=aa;relative_features[s:e],global_features[s:e]=verify_new_chunk(path,chunk_ids,aa)
    if (OUT/'result.json').exists():
        from report import validate_result
        result=validate_result(OUT)
        for rel,digest in result['model_inventory'].items():checked(safe_path(OUT,rel),digest)
        if len(result['model_inventory'])!=12:raise ValueError('Native inventory incomplete')
        return {'status':'ROUND06_SCREEN_COMPLETED','new_model_fits':0,'models_reused':12}
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
    hit_lists={a:[] for a in ARMS};den_lists=[];id_lists=[];fold_labels=[];records=[];diags=[];coverage_rows=[];support_rows=[];context_rows=[]
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
        mt=np.concatenate([relative_features[i,s] for i,s in zip(tr,selected,strict=True)])
        st=np.concatenate([global_features[i,s] for i,s in zip(tr,selected,strict=True)])
        xv=base[va].reshape(-1,134);mv=relative_features[va].reshape(-1,12);sv=global_features[va].reshape(-1,12)
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
        coverage_rows.append({'fold':fi,**coverage_statistics(truth[va],d,control_hits)})
        hit_lists['control_shared'].append(control_hits)
        records.append({'fold':fi,'arm':'control_shared','features':134,**pooled(control_hits,d)})
        save_json(OUT/'arm_reports'/f'fold-{fi}-control_shared.json',saved)
        prepared.append((fi,fold,tr,va,d,groups,yy,xt,mt,st,xv,mv,sv))
    # No challenger fit occurs until all six existing controls have passed replay.
    if counts['control_models_replayed']!=6:raise ValueError('Complete control replay gate required')
    log('all_six_controls_verified_before_new_fits')
    del train_truth
    PROGRESS.update(stage='matched_candidate_context_screen',completed=0,total=12)
    for fi,fold,tr,va,d,groups,yy,xt,mt,st,xv,mv,sv in prepared:
        # Target-aware support uses TRAINING targets censored before validation cutoff.
        censored=target_arrays(ids[tr],aids[tr],q[tr],last[tr],tc[tr],labels,FIT_END,cutoff=fold['cutoff'])
        for j,name in enumerate(NAMES):
            values=relative_features[tr,:,j]
            positive=[]
            for o in range(3):
                mask=censored[:,:,o]>0
                positive.append(None if not mask.any() else float((values[mask]!=0).mean()))
            support_rows.append({'fold':fi,'name':name,'all_training_candidate_support':float((values!=0).mean()),
                'positive_training_candidate_support':dict(zip(OBJECTIVES,positive,strict=True)),
                'censored_targets':True,'label_based_feature_selection':False})
        seen = base[tr,:,names.index(SEEN_SIGNAL)] > 0
        context_rows.append({'fold':fi,'training_sessions':len(tr),
            'all_seen_queries':int(seen.all(axis=1).sum()),
            'all_unseen_queries':int((~seen).all(axis=1).sum()),
            'mean_seen_candidates':float(seen.sum(axis=1).mean()),
            'mean_abs_primary_minus_global':np.abs(relative_features[tr]-global_features[tr]).mean(axis=(0,1)).tolist(),
            'strata':{label:{'candidate_rows':int(mask.sum()),
                'censored_positive_counts':{o:int(censored[:,:,j][mask].sum()) for j,o in enumerate(OBJECTIVES)}}
                for label,mask in [('repeat',seen),('discovery',~seen)]},
            'training_only':True,'labels_used_to_define_strata':False})
        del censored,seen
        # Diagnostics sample only training sessions and retain every formula for the comparison.
        sample=np.sort(np.random.default_rng(20260911).choice(tr,min(64,len(tr)),False))
        dx=relative_features[sample].reshape(-1,12).astype(np.float64);bx=base[sample].reshape(-1,134).astype(np.float64)
        ds=dx.std(0);bs=bx.std(0)
        dz=(dx-dx.mean(0))/np.where(ds>0,ds,1);bz=(bx-bx.mean(0))/np.where(bs>0,bs,1)
        corr=np.abs(dz.T@bz/len(dx));mx=np.abs(dz.T@dz/len(dx));np.fill_diagonal(mx,0)
        for j,name in enumerate(NAMES):
            diags.append({'fold':fi,'name':name,'training_sessions_sampled':len(sample),
                'support_fraction':float((dx[:,j]!=0).mean()),'positive_fraction':float((dx[:,j]>0).mean()),
                'std':float(ds[j]),'max_abs_corr_control':float(corr[j].max()),
                'max_abs_corr_added':float(mx[j].max()),'automatic_selection':False})
        del dx,bx,dz,bz,corr,mx
        for arm in ARMS[1:]:
            xtrain=np.concatenate([xt,mt if arm=='repeat_context' else st],axis=1)
            validation=np.concatenate([xv,mv if arm=='repeat_context' else sv],axis=1)
            allnames=names+list(NAMES if arm=='repeat_context' else GLOBAL_NAMES);hits=np.zeros((len(va),3),dtype=np.int64)
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
    for left,right in [('repeat_context','control_shared'),('global_context_ablation','control_shared'),
                       ('repeat_context','global_context_ablation')]:
        comparisons[left+'_minus_'+right]=compare(h[left],h[right],d,fs)
    all_valid=np.concatenate([f['valid'] for f in folds])
    coverage={'folds':coverage_rows,'pooled':coverage_statistics(truth[all_valid],d,h['control_shared']),
        'exploratory_reused_fitting_cohort':True,'untouched_holdout':False}
    save_json(OUT/'candidate_coverage.json',coverage)
    save_json(OUT/'positive_support.json',support_rows)
    save_json(OUT/'context_diagnostics.json',context_rows)
    save_arrays(OUT/'evaluation_statistics.npz',ids=np.concatenate(id_lists),folds=fs,denominator=d,**h)
    save_json(OUT/'diagnostics.json',diags)
    inv={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'models').rglob('model.txt'))}
    if len(inv)!=12:raise ValueError('Exactly 12 new experiment models required')
    result={'status':'ROUND06_SCREEN_COMPLETED','source_commit':COMMIT,'sessions':len(ids),'validation_sessions':len(d),
            'control_features':134,'added_features':12,'model_count':12,'control_models_refit':0,
            'control_prediction_replay_exact':True,'native_reload_prediction_parity':True,
            'arms':aggregate,'fold_results':records,'comparisons':comparisons,
            'primary_comparison':'repeat_context_minus_control_shared','secondary_comparisons_are_not_advancement_gates':True,'model_inventory':inv,
            'statistics_sha256':sha(OUT/'evaluation_statistics.npz'),'feature_retention_decisions':0,
            'selection_access':False,'evaluation_access':False,'competition_test_access':False,
            'original_graph_rebuilds':0,'algorithm_search':False,'round02_timing_included':False,'round03_mass_included':False,
            'historical_source_cutoff_ms':HISTORY_END,'old_graphs_rebuilt':0,'new_indices':0,
            'feature_engineering_complete':False,'control_recipe_unchanged':True,
            'candidate_coverage_sha256':sha(OUT/'candidate_coverage.json'),
            'positive_support_sha256':sha(OUT/'positive_support.json'),
            'context_diagnostics_sha256':sha(OUT/'context_diagnostics.json'),
            'candidate_context_before_label_dependent_sampling':True,
            'limitations':['Reused Round02 fitting cohort; exploratory, not untouched confirmation.',
              'Six cached graph-affinity inputs with complete-prefix repeat status; no failed demand or transition additions retained.',
              'The source window is unchanged. Relative context does not fix source staleness or missing candidates.',
              'Both challengers add 12 columns. Global ablation removes repeat-stratum conditioning, not all candidate context.',
              'Conditional support uses censored training targets only; no automatic feature filtering.',
              'Candidate policy and model capacity held fixed; neither is proved optimal.',
              'No new Kaggle predictions, submission, leaderboard score or feature promotion.']}
    save_json(OUT/'result.json',result)
    return {'status':result['status'],**counts,'primary_gain':comparisons[result['primary_comparison']]['gain']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['features','screen'])
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
            details={'features':features,'screen':screen}[args.phase](ctx)
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
    log('phase_finished',**details);print('RESULT: '+details['status'],flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
