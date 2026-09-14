"""Bounded fixed-ranker comparison for one predeclared historical feature family."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,gc,json,resource,signal,threading,time
from pathlib import Path
import numpy as np
from core import (OBJECTIVES,WEIGHTS,array_digest,checked,encode,forward_folds,hits_at20,
                  paired_interval,pooled,save_arrays,save_json,sha,target_arrays)
from shared_reuse import context as prepare_context, prior_chunks, require
from source_contract import HISTORY_END,FIT_END,COMMIT
from latent_features import names, affinity_summaries
import latent_models
from coverage import coverage_statistics
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
ROUND=11
NAMES=names(ROUND);GLOBAL_NAMES=names(ROUND,True)
SEEN_SIGNAL='domain_funnel_full_count_share'
ARMS=('control_shared', 'idf_session_latent', 'unweighted_session_latent')
PROGRESS={'stage':'initializing','completed':0,'total':0}
DEADLINE=float('inf')
class PausedCheckpointed(Exception):pass
def log(event,**values):print(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'event':event,**values},sort_keys=True),flush=True)
def before_unit():
 if time.monotonic()>DEADLINE-15:raise PausedCheckpointed('Work cap near; committed units preserved')
def load_json(path):return json.loads(Path(path).read_text())
def safe_path(root,rel):
 p=root/rel
 if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe artifact path')
 return p
def protocol():return load_json(ROOT/'protocol.json')
def feature_identity(ctx):
 d=ctx['shared']; require(ctx)
 manifest=OUT/'representations/manifest.json'
 latent_models.load_embeddings(ROOT,OUT,ROUND)
 return {'round':ROUND,'source_commit':COMMIT,'protocol_sha256':sha(ROOT/'protocol.json'),
   'code_sha256':{n:sha(ROOT/n) for n in ('experiment.py','shared_reuse.py','latent_models.py','latent_features.py','core.py')},
   'shared_input_contract_sha256':sha(d/'input_contract.json'),'shared_cohort_sha256':sha(d/'cohort.npz'),
   'shared_history_sha256':load_json(d/'history.json')['sha256'],'representation_manifest_sha256':sha(manifest),
   'names':list(NAMES),'ablation_names':list(GLOBAL_NAMES),'query_labels_read':False,
   'control_features_rebuilt':False,'candidate_budget':400,'historical_cutoff_ms':HISTORY_END}

def verify_new_chunk(path,ids,aids):
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!={'ids','aids','primary','ablation'}:raise ValueError('New chunk schema differs')
  if not np.array_equal(z['ids'],ids) or not np.array_equal(z['aids'],aids):raise ValueError('New chunk identity differs')
  xx=z['primary'];yy=z['ablation']
 if any(x.shape!=(len(ids),400,24) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in (xx,yy)):
  raise ValueError('Invalid 24-feature latent representation')
 return xx,yy

def features(ctx):
 contract=feature_identity(ctx);save_json(OUT/'feature_contract.json',contract);cid=sha(OUT/'feature_contract.json')
 a=require(ctx);vocabulary,embeddings=latent_models.load_embeddings(ROOT,OUT,ROUND)
 parts=[];new_count=reused=0
 for old,ids,candidates,base in prior_chunks(ctx):
  before_unit();s,e=old['start'],old['end'];path=OUT/'features'/f'part-{s//64:03d}.npz';receipt=path.with_suffix('.json')
  if not np.array_equal(a['ids'][s:e],ids) or not np.array_equal(a['candidates'][s:e],candidates):raise ValueError('Shared versus control alignment differs')
  unit={'contract_id':cid,'source_chunk_sha256':old['sha256'],'start':s,'end':e}
  if receipt.exists():
   rr=load_json(receipt)
   if rr['unit']!=unit:raise ValueError('Existing feature contract differs')
   checked(path,rr['sha256']);verify_new_chunk(path,ids,candidates);reused+=len(ids)
  else:
   if path.exists():raise ValueError('Orphan feature checkpoint preserved; return evidence')
   values={key:[] for key in ('primary','ablation')}
   for j,sid in enumerate(ids):
    anchors=a['anchors'][s+j];anchors=anchors[anchors>=0]
    for arm,(q,t) in embeddings.items():
     x=affinity_summaries(anchors,candidates[j],vocabulary,q,t)
     if s+j<8:
      rev=affinity_summaries(anchors,candidates[j,::-1],vocabulary,q,t)
      if not np.allclose(rev[::-1],x,rtol=1e-6,atol=1e-6):raise ValueError('Candidate permutation invariant failed')
     values[arm].append(x)
   digest=save_arrays(path,ids=ids,aids=candidates,primary=np.stack(values['primary']),ablation=np.stack(values['ablation']))
   save_json(receipt,{'unit':unit,'sha256':digest});new_count+=len(ids)
  parts.append({'path':str(path.relative_to(OUT)),'sha256':sha(path),'start':s,'end':e})
  PROGRESS.update(stage='latent_feature_chunks',completed=e,total=len(ctx['ids']));log('feature_checkpoint',**PROGRESS)
 save_json(OUT/'feature_manifest.json',{'status':f'ROUND{ROUND:02d}_FEATURES_READY','contract_id':cid,'sessions':len(ctx['ids']),
    'names':list(NAMES),'ablation_names':list(GLOBAL_NAMES),'parts':parts,'query_labels_read':False,'model_fits':0,'candidate_policy_changed':False})
 return {'status':f'ROUND{ROUND:02d}_FEATURES_READY','new_sessions':new_count,'reused_sessions':reused}

def prepare(ctx):
 return latent_models.prepare(ctx,ROOT,OUT,ROUND,before_unit,lambda **kw:PROGRESS.update(kw))

def representations(ctx):
 return latent_models.learn(ctx,ROOT,OUT,ROUND,before_unit,lambda **kw:PROGRESS.update(kw))
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
    base=np.empty((len(ids),400,134),dtype=np.float32);relative_features=np.empty((len(ids),400,24),dtype=np.float32);global_features=np.empty((len(ids),400,24),dtype=np.float32)
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
        return {'status':'ROUND11_SCREEN_COMPLETED','new_model_fits':0,'models_reused':12}
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
        xv=base[va].reshape(-1,134);mv=relative_features[va].reshape(-1,24);sv=global_features[va].reshape(-1,24)
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
    PROGRESS.update(stage='matched_historical_feature_screen',completed=0,total=12)
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
            'mean_abs_primary_minus_ablation':np.abs(relative_features[tr]-global_features[tr]).mean(axis=(0,1)).tolist(),
            'strata':{label:{'candidate_rows':int(mask.sum()),
                'censored_positive_counts':{o:int(censored[:,:,j][mask].sum()) for j,o in enumerate(OBJECTIVES)}}
                for label,mask in [('repeat',seen),('discovery',~seen)]},
            'training_only':True,'labels_used_to_define_strata':False})
        del censored,seen
        # Diagnostics sample only training sessions and retain every formula for the comparison.
        sample=np.sort(np.random.default_rng(20260911).choice(tr,min(64,len(tr)),False))
        dx=relative_features[sample].reshape(-1,24).astype(np.float64);bx=base[sample].reshape(-1,134).astype(np.float64)
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
            xtrain=np.concatenate([xt,mt if arm=='idf_session_latent' else st],axis=1)
            validation=np.concatenate([xv,mv if arm=='idf_session_latent' else sv],axis=1)
            allnames=names+list(NAMES if arm=='idf_session_latent' else GLOBAL_NAMES);hits=np.zeros((len(va),3),dtype=np.int64)
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
    for left,right in [('idf_session_latent','control_shared'),('unweighted_session_latent','control_shared'),
                       ('idf_session_latent','unweighted_session_latent')]:
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
    result={'status':'ROUND11_SCREEN_COMPLETED','source_commit':COMMIT,'sessions':len(ids),'validation_sessions':len(d),
            'control_features':134,'added_features':24,'model_count':12,'control_models_refit':0,
            'control_prediction_replay_exact':True,'native_reload_prediction_parity':True,
            'arms':aggregate,'fold_results':records,'comparisons':comparisons,
            'primary_comparison':'idf_session_latent_minus_control_shared','secondary_comparisons_are_not_advancement_gates':True,'model_inventory':inv,
            'statistics_sha256':sha(OUT/'evaluation_statistics.npz'),'feature_retention_decisions':0,
            'selection_access':False,'evaluation_access':False,'competition_test_access':False,
            'original_graph_rebuilds':0,'shared_historical_cache_reused':True,'algorithm_search':False,'round02_timing_included':False,'round03_mass_included':False,
            'historical_source_cutoff_ms':HISTORY_END,'old_graphs_rebuilt':0,'new_indices':0,
            'feature_engineering_complete':False,'control_recipe_unchanged':True,
            'candidate_coverage_sha256':sha(OUT/'candidate_coverage.json'),
            'positive_support_sha256':sha(OUT/'positive_support.json'),
            'context_diagnostics_sha256':sha(OUT/'context_diagnostics.json'),
            'features_committed_before_label_access':True,
            'limitations':['Reused Round02 fitting cohort; adaptive exploration, not independent validation.',
              'Rounds11 and12 are independent predeclared latent-feature comparisons against the immutable control.',
              'All original study sessions excluded from certified retained historical tails; source not full raw sessions.',
              'Historical cache was retrieved using up to64 sessions per anchor. New factors use only history selected by earliest-fold training anchors, not later validation anchors or full raw data. Vocabulary is capped at60000; compression and coverage limits remain.',
              'Candidate pool frozen at400. Candidate expansion is not tested or applied.',
              'All target-aware diagnostics use censored training targets only; no automatic feature filtering.',
              'Model capacity and data scale held fixed to isolate features; not proved optimal.',
              'Intervals are descriptive and do not adjust for adaptive hypothesis selection or temporal dependence.',
              'No Kaggle submission, achieved leaderboard score or automatic feature promotion.']}
    save_json(OUT/'result.json',result)
    return {'status':result['status'],**counts,'primary_gain':comparisons[result['primary_comparison']]['gain']}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','representations','features','screen'])
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
            details={'prepare':prepare,'representations':representations,'features':features,'screen':screen}[args.phase](ctx)
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
