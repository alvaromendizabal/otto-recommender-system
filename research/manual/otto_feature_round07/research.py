"""Bounded manual-only historical-session feature feasibility study; zero model fits."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
from core import checked,sha,save_json,save_arrays,forward_folds,validate_prefix,array_digest
from neighbors import NAMES,ABLATION_NAMES,query_anchors,group_history,build_query
from history_io import (connect_history,ids_table,posting_sql,history_sql,collect_rows,
                        MAX_POSTINGS,MAX_HISTORY_ROWS,backend_smoke)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
COMMIT='2638faa34afa427ed4ac4e92bba04deda57c688e'
HISTORY_END=1660687200000
FIT_END=1660946400000
PILOT=256
SIGNALS=('graph_time_all_n1_uniform_sum','graph_cart_all_n5_uniform_max',
'domain_norm_symmetric_time_row_recent_unique_mean','domain_norm_forward_time_row_recent_unique_mean',
'domain_norm_symmetric_order_row_purchase_unique_mean','domain_norm_symmetric_order_row_last_mean',
'domain_funnel_full_count_share')
DEADLINE=float('inf')
PROGRESS={'stage':'initializing','completed':0,'total':0}

class CheckpointPause(Exception): pass

def log(event,**kwargs):
    print(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'event':event,**kwargs},sort_keys=True),flush=True)

def before_unit():
    if time.monotonic()>DEADLINE-12:raise CheckpointPause('Useful-work cap reached; committed stages preserved')

def read(path):return json.loads(Path(path).read_text())

def safe(root,rel):
    p=root/rel
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe source path')
    return p

def protocol():
    p=read(ROOT/'protocol.json')
    expected={'round':7,'pilot_sessions':256,'max_query_anchors':4,'postings_per_anchor':64,'neighbors_per_query':64,
              'proposals_per_objective':100,'new_model_fits':0,'candidate_budget':400,'source_commit':COMMIT,
              'selection_access':False,'evaluation_access':False,'competition_test_access':False}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen pilot protocol differs')
    return p

def source_context(home: Path):
    p=protocol();previous=home/'otto_feature_round02';old=previous/'outputs'
    ref=read(ROOT/'evidence/ROUND02_REFERENCE.json')
    for name,digest in ref['round02_files'].items():checked(safe(previous,name),digest)
    spec=importlib.util.spec_from_file_location('round02_certified_helpers',previous/'replication.py')
    if spec is None or spec.loader is None:raise ValueError('Saved Round02 helpers missing')
    prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
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


def identity(ctx):
    return {'source_commit':COMMIT,'protocol_sha256':sha(ROOT/'protocol.json'),
       'code_sha256':{n:sha(ROOT/n) for n in ('research.py','neighbors.py','history_io.py','metrics.py','core.py')},
       'round02_cohort_sha256':sha(ctx['old']/'cohort.json'),'round02_feature_manifest_sha256':sha(ctx['old']/'feature_manifest.json'),
       'round06_result_sha256':sha(ROOT/'evidence/ROUND06_RESULT.json'),'history_tail_sha256':ctx['history_sha'],
       'study_exclusions_sha256':array_digest(ctx['excluded']),'history_end':HISTORY_END,
       'source_is_retained_tail_not_full_raw_sessions':True,'new_model_fits':0}


def require(ctx):
    if read(OUT/'input_contract.json')!=identity(ctx):raise ValueError('Pilot input or code contract changed')
    m=read(OUT/'pilot.json');checked(OUT/'pilot.npz',m['sha256'])
    with np.load(OUT/'pilot.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if len(a['ids'])!=PILOT or a['candidates'].shape!=(PILOT,400) or a['anchors'].shape!=(PILOT,4):raise ValueError('Pilot matrix shape differs')
    return m,a


def prepare(ctx):
    import polars as pl
    checked(ctx['history'],ctx['history_sha'])
    co=ctx['cohort'];ids=ctx['ids'];q=np.asarray(co['query_ts'],np.int64)
    first=forward_folds(ids,q)[0];saved=read(ctx['old']/'folds/fold-0.json')
    eligible=ids[first['train']]
    if eligible.tolist()!=saved['training_ids'] or first['cutoff']!=saved['cutoff_ms']:raise ValueError('Earliest-fold training ledger changed')
    with np.errstate(over='ignore'):key=(eligible.astype(np.uint64)^np.uint64(20260912))*np.uint64(11400714819323198485)
    chosen=np.sort(eligible[np.argsort(key,kind='stable')[:PILOT]])
    if len(chosen)!=PILOT:raise ValueError('Insufficient early training sessions')
    rows=np.searchsorted(ids,chosen)
    frame=(pl.scan_parquet(ctx['paths']['corpus']/'observed.parquet')
           .filter((pl.col('split_role')=='fit') & pl.col('session').is_in(chosen.tolist()))
           .select('session','aid','ts','event_type','event_index').collect().sort('session','event_index'))
    groups={int(k[0] if isinstance(k,tuple) else k):g for k,g in frame.partition_by('session',as_dict=True).items()}
    if set(groups)!=set(map(int,chosen)):raise ValueError('Complete observed pilot prefixes missing')
    anchors=np.full((PILOT,4),-1,np.int64)
    for j,(sid,i) in enumerate(zip(chosen,rows,strict=True)):
        g=groups[int(sid)];a=g['aid'].to_numpy();t=g['ts'].to_numpy();k=g['event_type'].to_numpy();ix=g['event_index'].to_numpy()
        ledger={n:co[n][i] for n in ('first_ts','query_ts','observed_events','observed_last_index')};ledger['period_end']=FIT_END
        validate_prefix(a,t,k,ix,ledger,HISTORY_END,FIT_END);aa=query_anchors(a);anchors[j,:len(aa)]=aa
    candidates=np.empty((PILOT,400),np.int64);base=np.empty((PILOT,400,7),np.float32);covered=set()
    for part in ctx['manifest']['parts']:
        before_unit();s,e=part['start'],part['end'];jj=np.flatnonzero((rows>=s)&(rows<e))
        if not len(jj):continue
        path=safe(ctx['old'],part['path']);checked(path,part['sha256'])
        with np.load(path,allow_pickle=False) as z:
            if z['x'].shape!=(64,400,158) or z['aids'].shape!=(64,400) or not np.array_equal(z['ids'],ids[s:e]):raise ValueError('Original chunk schema changed')
            for j in jj:
                local=rows[j]-s;candidates[j]=z['aids'][local];base[j]=z['x'][local][:,[ctx['names'].index(n) for n in SIGNALS]]
                covered.add(int(j))
        PROGRESS.update(stage='pilot_cached_candidates',completed=len(covered),total=PILOT)
    if len(covered)!=PILOT or not np.isfinite(base).all() or any(len(np.unique(a))!=400 for a in candidates):raise ValueError('Invalid pilot candidates')
    save_json(OUT/'input_contract.json',identity(ctx))
    digest=save_arrays(OUT/'pilot.npz',ids=chosen,candidates=candidates,anchors=anchors,baseline_signals=base,
               query_ts=q[rows],last_index=np.asarray(co['observed_last_index'],np.int64)[rows],
               true_counts=np.asarray(co['true_counts'],np.int64)[rows])
    save_json(OUT/'pilot.json',{'status':'ROUND07_PILOT_READY','sha256':digest,'sessions':PILOT,
        'source_eligible_training_sessions':len(eligible),'cutoff_ms':first['cutoff'],
        'selection':'ID hash seed 20260912; only earliest-fold training IDs; no labels or outcomes used',
        'training_only':True,'labels_read':False,'baseline_signals':list(SIGNALS),'new_model_fits':0})
    return {'status':'ROUND07_PILOT_READY','sessions':PILOT}


def postings(ctx):
    _,p=require(ctx);checked(ctx['history'],ctx['history_sha']);before_unit()
    # This real-backend gate runs even when terminal users omit the tests command.
    smoke=backend_smoke();log('backend_smoke_passed',**smoke)
    if (OUT/'postings.json').exists():
        m=read(OUT/'postings.json');checked(OUT/'postings.npz',m['sha256']);return {'status':'ROUND07_POSTINGS_READY','reused':True}
    anchors=np.unique(p['anchors'][p['anchors']>=0]);con=connect_history(ctx['history'],OUT/'working/postings')
    try:
        ids_table(con,'wanted','anchor',anchors);ids_table(con,'excluded','session',ctx['excluded'])
        PROGRESS.update(stage='scan_historical_anchor_postings',completed=0,total=len(anchors))
        a=collect_rows(con.execute(posting_sql()),4,MAX_POSTINGS,PROGRESS)
    finally:con.close()
    if len(a) and (not set(map(int,a[:,0]))<=set(map(int,anchors)) or np.unique(a[:,:2],axis=0).shape[0]!=len(a)):
        raise ValueError('Invalid historical postings')
    digest=save_arrays(OUT/'postings.npz',postings=a)
    save_json(OUT/'postings.json',{'status':'ROUND07_POSTINGS_READY','sha256':digest,'rows':len(a),
        'requested_anchors':len(anchors),'supported_anchors':int(np.unique(a[:,0]).size) if len(a) else 0,
        'anchors_over_cap':int(len(set(map(int,a[a[:,3]>64,0])))) if len(a) else 0,
        'source_sessions':int(np.unique(a[:,1]).size) if len(a) else 0,'per_anchor_cap':64,
        'cap_policy':'64 most recent matching historical sessions per anchor, tie by session ID',
        'history_tail_sha256':ctx['history_sha'],'backend_smoke':smoke,'new_model_fits':0})
    return {'status':'ROUND07_POSTINGS_READY','postings':len(a)}


def histories(ctx):
    require(ctx);pm=read(OUT/'postings.json');checked(OUT/'postings.npz',pm['sha256']);before_unit()
    if (OUT/'history.json').exists():
        m=read(OUT/'history.json');checked(OUT/'history.npz',m['sha256']);return {'status':'ROUND07_HISTORY_READY','reused':True}
    checked(ctx['history'],ctx['history_sha'])
    with np.load(OUT/'postings.npz',allow_pickle=False) as z:selected=np.unique(z['postings'][:,1])
    if len(selected)>65536:raise ValueError('Historical session cap exceeded')
    if len(selected):
        con=connect_history(ctx['history'],OUT/'working/histories')
        try:
            ids_table(con,'selected','session',selected);ids_table(con,'excluded','session',ctx['excluded'])
            PROGRESS.update(stage='scan_selected_historical_sessions',completed=0,total=len(selected))
            rows=collect_rows(con.execute(history_sql()),5,MAX_HISTORY_ROWS,PROGRESS)
        finally:con.close()
    else:rows=np.empty((0,5),np.int64)
    grouped=group_history(rows,HISTORY_END,set(map(int,ctx['excluded'])))
    if set(grouped)!=set(map(int,selected)):raise ValueError('Selected source histories missing')
    digest=save_arrays(OUT/'history.npz',events=rows)
    save_json(OUT/'history.json',{'status':'ROUND07_HISTORY_READY','sha256':digest,'source_sessions':len(grouped),'retained_events':len(rows),
        'source_tail_not_complete_original_sessions':True,'new_model_fits':0,'postings_sha256':pm['sha256'],'history_tail_sha256':ctx['history_sha']})
    return {'status':'ROUND07_HISTORY_READY','sessions':len(grouped),'events':len(rows)}


def features(ctx):
    _,p=require(ctx);hm=read(OUT/'history.json');pm=read(OUT/'postings.json')
    checked(OUT/'history.npz',hm['sha256']);checked(OUT/'postings.npz',pm['sha256'])
    with np.load(OUT/'history.npz',allow_pickle=False) as z:hist=group_history(z['events'],HISTORY_END,set(map(int,ctx['excluded'])))
    with np.load(OUT/'postings.npz',allow_pickle=False) as z:posts=z['postings']
    lookup={}
    for a,s,_,_ in posts:lookup.setdefault(int(a),set()).add(int(s))
    parts=[]
    for start in range(0,PILOT,32):
        before_unit();end=start+32;path=OUT/'features'/f'part-{start//32:02d}.npz';receipt=path.with_suffix('.json')
        contract={'start':start,'end':end,'input_contract_sha256':sha(OUT/'input_contract.json'),
                  'pilot_sha256':sha(OUT/'pilot.npz'),'history_sha256':hm['sha256'],'postings_sha256':pm['sha256']}
        if receipt.exists():
            rr=read(receipt)
            if rr['contract']!=contract:raise ValueError('Existing feature checkpoint contract changed')
            checked(path,rr['sha256'])
        else:
            if path.exists():raise ValueError('Orphan feature checkpoint preserved; return evidence')
            values={k:[] for k in ('primary','ablation','primary_proposals','ablation_proposals','primary_scores','ablation_scores','neighbor_counts','multi_neighbor_counts','source_counts')}
            for i in range(start,end):
                aa=p['anchors'][i];aa=aa[aa>=0];source_ids=sorted(set().union(*(lookup.get(int(a),set()) for a in aa)))
                r=build_query(aa,p['candidates'][i],source_ids,hist);left=r['session_context'];right=r['last_anchor']
                if i<8:
                    alt=build_query(aa,p['candidates'][i,::-1],list(reversed(source_ids)),hist)
                    for arm in ('session_context','last_anchor'):
                        if not np.array_equal(r[arm]['features'],alt[arm]['features'][::-1]) or not np.array_equal(r[arm]['proposal_ids'],alt[arm]['proposal_ids']):
                            raise ValueError('Candidate/source order invariance failed')
                for key,val in [('primary',left['features']),('ablation',right['features']),
                    ('primary_proposals',left['proposal_ids']),('ablation_proposals',right['proposal_ids']),
                    ('primary_scores',left['proposal_scores']),('ablation_scores',right['proposal_scores']),
                    ('neighbor_counts',[left['neighbor_count'],right['neighbor_count']]),
                    ('multi_neighbor_counts',[left['multi_overlap_neighbors'],right['multi_overlap_neighbors']]),('source_counts',len(source_ids))]:values[key].append(val)
            digest=save_arrays(path,ids=p['ids'][start:end],candidates=p['candidates'][start:end],**{k:np.asarray(v) for k,v in values.items()})
            save_json(receipt,{'contract':contract,'sha256':digest})
        parts.append({'path':str(path.relative_to(OUT)),'sha256':sha(path),'start':start,'end':end})
        PROGRESS.update(stage='session_neighbor_features',completed=end,total=PILOT);log('feature_chunk_committed',**PROGRESS)
    save_json(OUT/'feature_manifest.json',{'status':'ROUND07_FEATURES_READY','sessions':PILOT,'names':list(NAMES),'ablation_names':list(ABLATION_NAMES),
        'parts':parts,'input_contract_sha256':sha(OUT/'input_contract.json'),'labels_read':False,'model_fits':0})
    return {'status':'ROUND07_FEATURES_READY','sessions':PILOT,'new_model_fits':0}


def audit(ctx):
    from metrics import censored_truth,audit_arrays
    meta,p=require(ctx);fm=read(OUT/'feature_manifest.json')
    if fm['input_contract_sha256']!=sha(OUT/'input_contract.json'):raise ValueError('Feature inputs changed')
    chunks=[]
    for r in fm['parts']:
        path=safe(OUT,r['path']);checked(path,r['sha256'])
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if not np.array_equal(a['ids'],p['ids'][r['start']:r['end']]) or not np.array_equal(a['candidates'],p['candidates'][r['start']:r['end']]):raise ValueError('Feature alignment differs')
        chunks.append(a)
    data={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
    if not np.array_equal(data['ids'],p['ids']):raise ValueError('Feature chunks not complete or ordered')
    # This is the FIRST stage allowed to read pilot targets. All feature/proposal files already exist.
    labels,labelsha=ctx['prior'].labels_for(ctx['paths'],p['ids'])
    truths=censored_truth(p['ids'],p['query_ts'],p['last_index'],p['true_counts'],labels,FIT_END,meta['cutoff_ms'])
    result,stats=audit_arrays(p['candidates'],data['primary'],data['ablation'],data['primary_proposals'],data['ablation_proposals'],truths)
    dx=data['primary'].reshape(-1,12).astype(np.float64);bx=p['baseline_signals'].reshape(-1,7).astype(np.float64)
    std=dx.std(0);bs=bx.std(0);dz=(dx-dx.mean(0))/np.where(std>0,std,1);bz=(bx-bx.mean(0))/np.where(bs>0,bs,1)
    corr=np.abs(dz.T@bz/len(dx));diags=[]
    for j,n in enumerate(NAMES):
        diags.append({'name':n,'support_fraction':float((dx[:,j]>0).mean()),'std':float(std[j]),
                      'max_abs_corr_seven_cached_signals':float(corr[j].max()),'compared_control_signals':list(SIGNALS)})
    neighbors={'queries':PILOT,'query_anchor_counts':(p['anchors']>=0).sum(1).tolist(),
        'neighbor_counts':data['neighbor_counts'].tolist(),'multi_overlap_neighbor_counts':data['multi_neighbor_counts'].tolist(),
        'candidate_rows_with_primary_support':(data['primary'].sum(2)>0).sum(1).tolist(),
        'mean_abs_primary_minus_ablation':np.abs(data['primary']-data['ablation']).mean(axis=(0,1)).tolist(),
        'queries_with_two_or_more_anchor_evidence':int((data['multi_neighbor_counts'][:,0]>0).sum())}
    seen_positive=result['arms']['session_context']['supported_baseline_positive_items'];new_hits=result['arms']['session_context']['new_distinct_target_hits']
    informative=sum(seen_positive[1:])>=20 and (neighbors['queries_with_two_or_more_anchor_evidence']>=26 or sum(new_hits[1:])>=3)
    result.update(status='ROUND07_AUDIT_READY',sessions=PILOT,features_per_arm=12,
        decision='REVIEW_FOR_MATCHED_RANKER_SCREEN_NOT_PROMOTION' if informative else 'REVIEW_SUPPORT_LIMITS_BEFORE_MORE_COMPUTE',
        heuristic_triage_only=True,insufficient_denominator_objectives=[k for k,v in zip(('clicks','carts','orders'),result['baseline_candidate_oracle']['denominators']) if v<20],
        feature_engineering_complete=False,labels_sha256=labelsha,pilot_manifest_sha256=sha(OUT/'pilot.json'),
        feature_manifest_sha256=sha(OUT/'feature_manifest.json'),source=read(OUT/'history.json'),postings=read(OUT/'postings.json'),
        limitations=['Reused fitting data; 256-query early-training feasibility pilot, not independent validation.',
        'Source uses retained historical tails and the 64 most recent matching sessions per last-four anchor; not exhaustive nearest-neighbor search.',
        'All labels used only after committed features/proposals; pilot labels censored before earliest validation cutoff.',
        'Baseline400 union top100 proposals per objective has at most500 items; its oracle gains are not achieved ranking gains.',
        'Last-anchor arm removes older anchors; neighbor composition and similarity change together. No causal isolation claimed.',
        'No new ranker fits, leaderboard submission or feature promotion.'])
    save_json(OUT/'diagnostics.json',diags);save_json(OUT/'neighbor_diagnostics.json',neighbors)
    save_arrays(OUT/'audit_statistics.npz',ids=p['ids'],**stats)
    save_json(OUT/'result.json',result)
    from report import render
    render(OUT)
    return {'status':'ROUND07_AUDIT_READY','decision':result['decision'],'new_model_fits':0}


def main():
    global DEADLINE
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','postings','histories','features','audit']);args=parser.parse_args()
    p=protocol();OUT.mkdir(exist_ok=True);start=time.monotonic();DEADLINE=start+p['work_seconds'][args.phase]
    details={};code=0
    try:
        ctx=source_context(Path.home());before_unit();details=globals()[args.phase](ctx)
    except CheckpointPause as exc:details={'status':'PAUSED_CHECKPOINTED','error':str(exc)};code=75
    except Exception as exc:
        import traceback;traceback.print_exc();details={'status':'STOPPED_REVIEW_REQUIRED','error':str(exc),'error_type':type(exc).__name__};code=1
    finally:
        details.update(phase=args.phase,elapsed_seconds=time.monotonic()-start,
          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,new_model_fits=0)
        save_json(OUT/'runs'/f'{args.phase}-{time.time_ns()}.json',details)
    log('phase_complete',**details);print('RESULT: '+details['status'],flush=True);return code

if __name__=='__main__':raise SystemExit(main())
