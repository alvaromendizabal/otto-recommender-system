"""Two additional Parquet scans at most, shared by the two fixed-cohort experiments.

Reuses every verified Round07 posting and retained historical event. Only new
anchors and missing source sessions are read. No labels are used in this module.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from core import checked,sha,save_arrays,save_json,validate_prefix,encode
from neighbors import query_anchors,group_history
from history_io import connect_history,ids_table,posting_sql,history_sql,collect_rows,backend_smoke
from source_contract import source_context,HISTORY_END,FIT_END

ROOT=Path(__file__).resolve().parent
PROGRESS={}

def read(p):return json.loads(Path(p).read_text())
def data_root(home):return Path(home)/'otto_feature_round08/outputs/shared'

def context(home):
    ctx=source_context(home);p=read(ROOT/'protocol.json')
    if (p['round'] not in (8,9) or p['sessions']!=4096 or p['added_features']!=24 or p['ablation_features']!=24
        or p['candidate_budget']!=400 or p['maximum_new_models']!=12 or p['rounds']!=150
        or any(p[k] for k in ('selection_access','evaluation_access','competition_test_access'))):
        raise ValueError('Frozen two-round protocol changed')
    oldp=read(Path(home)/'otto_feature_round02/protocol.json')
    for name in ('model_params','rounds','negative_budget','negative_sampling_seed'):
        if p[name]!=oldp[name]:raise ValueError('Ranker/negative-sampling settings changed')
    prev=Path(home)/'otto_feature_round07/outputs'
    for n in ('result.json','input_contract.json','pilot.json','postings.json','history.json','feature_manifest.json'):
        checked(prev/n,sha(ROOT/'evidence'/('ROUND07_'+n.upper())))
    for n,receipt in (('postings.npz','postings.json'),('history.npz','history.json'),('pilot.npz','pilot.json')):
        checked(prev/n,read(prev/receipt)['sha256'])
    if read(prev/'result.json')['status']!='ROUND07_AUDIT_READY':raise ValueError('Round07 did not complete')
    ref=read(ROOT/'evidence/ROUND02_REFERENCE.json')
    controls={k:v for k,v in ref['result']['model_inventory'].items() if '/control_shared/' in k}
    if len(controls)!=6:raise ValueError('Six controls required')
    for name,digest in controls.items():checked(ctx['old']/name,digest)
    ctx.update(home=Path(home),shared=data_root(home),round07=prev,base_names=ctx['names'],
               protocol=p,controls=controls,ref=ref)
    return ctx

def identity(ctx):
    return {'source_commit':ctx['protocol']['source_commit'],
      'history_tail_sha256':ctx['history_sha'],'history_end':HISTORY_END,
      'cohort_sha256':sha(ctx['old']/'cohort.json'),
      'feature_manifest_sha256':sha(ctx['old']/'feature_manifest.json'),
      'round07_input_contract_sha256':sha(ctx['round07']/'input_contract.json'),
      'round07_postings_sha256':read(ctx['round07']/'postings.json')['sha256'],
      'round07_history_sha256':read(ctx['round07']/'history.json')['sha256'],
      'source_code':{n:sha(ROOT/n) for n in ('shared_data.py','source_contract.py','history_io.py','neighbors.py','core.py')},
      'queries':len(ctx['ids']),'max_anchors':4,'postings_per_anchor':64,
      'historical_session_cap':400000,'historical_event_cap':10000000,
      'query_labels_read':False,'raw_json_scans':0}

def load_arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def require(ctx):
    d=ctx['shared']
    if read(d/'input_contract.json')!=identity(ctx):raise ValueError('Shared source/code contract differs')
    m=read(d/'cohort.json');checked(d/'cohort.npz',m['sha256']);a=load_arrays(d/'cohort.npz')
    if not np.array_equal(a['ids'],ctx['ids']) or a['anchors'].shape!=(len(ctx['ids']),4) or a['candidates'].shape!=(len(ctx['ids']),400):
        raise ValueError('Shared cohort/candidates changed')
    return a

def prior_chunks(ctx):
    start=0
    for row in ctx['manifest']['parts']:
        if row['start']!=start or row['end']!=start+64:raise ValueError('Original partition gap')
        p=ctx['old']/row['path'];checked(p,row['sha256']);z=load_arrays(p)
        if (z['x'].shape!=(64,400,158) or z['x'].dtype!=np.float32 or not np.isfinite(z['x']).all()
            or z['aids'].shape!=(64,400) or not np.array_equal(z['ids'],ctx['ids'][start:start+64])
            or any(len(np.unique(a))!=400 for a in z['aids'])):raise ValueError('Cached original data schema differs')
        yield row,z['ids'],z['aids'],z['x'][:,:,:134].copy()
        start+=64
    if start!=len(ctx['ids']):raise ValueError('Incomplete original cohort')

def prepare(ctx,before=lambda:None):
    import polars as pl
    d=ctx['shared'];d.mkdir(parents=True,exist_ok=True)
    save_json(d/'input_contract.json',identity(ctx))
    if (d/'cohort.json').exists():
        require(ctx);return {'status':'SHARED_COHORT_READY','new_queries':0,'reused_queries':len(ctx['ids'])}
    if (d/'cohort.npz').exists():raise ValueError('Orphan shared cohort preserved')
    before();checked(ctx['history'],ctx['history_sha'])
    ids=ctx['ids'];co=ctx['cohort'];n=len(ids)
    frame=(pl.scan_parquet(ctx['paths']['corpus']/'observed.parquet')
      .filter((pl.col('split_role')=='fit')&pl.col('session').is_in(ids.tolist()))
      .select('session','aid','ts','event_type','event_index').collect().sort('session','event_index'))
    groups={int(k[0] if isinstance(k,tuple) else k):g for k,g in frame.partition_by('session',as_dict=True).items()}
    if set(groups)!=set(map(int,ids)):raise ValueError('Observed cohort incomplete')
    anchors=np.full((n,4),-1,np.int64);candidates=np.empty((n,400),np.int64)
    for i,sid in enumerate(ids):
        if i%64==0:before()
        g=groups[int(sid)];a,t,k,ix=(g[c].to_numpy() for c in ('aid','ts','event_type','event_index'))
        ledger={name:co[name][i] for name in ('first_ts','query_ts','observed_events','observed_last_index')};ledger['period_end']=FIT_END
        validate_prefix(a,t,k,ix,ledger,HISTORY_END,FIT_END);q=query_anchors(a);anchors[i,:len(q)]=q
    for row,_,a,_ in prior_chunks(ctx):before();candidates[row['start']:row['end']]=a
    old=load_arrays(ctx['round07']/'pilot.npz');ix=np.searchsorted(ids,old['ids'])
    if not np.array_equal(ids[ix],old['ids']) or not np.array_equal(anchors[ix],old['anchors']) or not np.array_equal(candidates[ix],old['candidates']):
        raise ValueError('Round07 pilot does not embed exactly in full cohort')
    digest=save_arrays(d/'cohort.npz',ids=ids,anchors=anchors,candidates=candidates)
    save_json(d/'cohort.json',{'status':'SHARED_COHORT_READY','sessions':n,'sha256':digest,
      'input_contract_sha256':sha(d/'input_contract.json'),'query_labels_read':False,'round07_pilot_replayed':len(ix)})
    return {'status':'SHARED_COHORT_READY','sessions':n}

def validate_postings(rows,requested,excluded):
    if rows.ndim!=2 or rows.shape[1]!=4 or rows.dtype.kind not in 'iu':raise ValueError('Posting schema')
    if len(rows):
        if not set(rows[:,0])<=set(requested) or set(rows[:,1])&set(excluded) or (rows[:,2]<0).any() or (rows[:,2]>=HISTORY_END).any():raise ValueError('Posting source leakage')
        if len(set(map(tuple,rows[:,:2])))!=len(rows):raise ValueError('Duplicate anchor/session')
        ordered=rows[np.argsort(rows[:,0],kind='stable')]
        _,starts,counts=np.unique(ordered[:,0],return_index=True,return_counts=True)
        for start,count in zip(starts,counts,strict=True):
            group=ordered[start:start+count]
            if len(group)>64 or len(np.unique(group[:,3]))!=1 or group[0,3]<len(group):raise ValueError('Uncapped frequency or posting cap mismatch')

def postings(ctx,before=lambda:None):
    d=ctx['shared'];a=require(ctx);wanted=np.unique(a['anchors'][a['anchors']>=0]);before()
    if (d/'postings.json').exists():
        checked(d/'postings.npz',read(d/'postings.json')['sha256']);validate_postings(load_arrays(d/'postings.npz')['postings'],wanted,ctx['excluded'])
        return {'status':'SHARED_POSTINGS_READY','reused':True}
    if (d/'postings.npz').exists():raise ValueError('Orphan shared postings preserved')
    smoke=backend_smoke()
    old=load_arrays(ctx['round07']/'postings.npz')['postings'];pilot=load_arrays(ctx['round07']/'pilot.npz')
    known=np.unique(pilot['anchors'][pilot['anchors']>=0]);missing=np.setdiff1d(wanted,known)
    if len(missing):
        con=connect_history(ctx['history'],d/'working/postings')
        try:
            ids_table(con,'wanted','anchor',missing);ids_table(con,'excluded','session',ctx['excluded'])
            new=collect_rows(con.execute(posting_sql()),4,4*len(ctx['ids'])*64,PROGRESS)
        finally:con.close()
    else:new=np.empty((0,4),np.int64)
    rows=np.concatenate([old,new]);rows=rows[np.lexsort((rows[:,1],rows[:,0]))]
    validate_postings(rows,wanted,ctx['excluded'])
    sessions=np.unique(rows[:,1])
    if len(sessions)>400000:raise ValueError('400000 historical-session cap; no trimming')
    digest=save_arrays(d/'postings.npz',postings=rows)
    save_json(d/'postings.json',{'status':'SHARED_POSTINGS_READY','sha256':digest,'requested_anchors':len(wanted),
      'new_anchors_scanned':len(missing),'round07_anchors_reused':len(known),'supported_anchors':len(np.unique(rows[:,0])),
      'source_sessions':len(sessions),'rows':len(rows),'backend_smoke':smoke,
      'anchors_over_cap':len(np.unique(rows[rows[:,3]>64,0])),
      'cohort_sha256':sha(d/'cohort.npz'),'additional_parquet_scans':int(bool(len(missing))),'query_labels_read':False})
    return {'status':'SHARED_POSTINGS_READY','new_anchors':len(missing),'reused_anchors':len(known),'historical_sessions':len(sessions)}

def histories(ctx,before=lambda:None):
    d=ctx['shared'];require(ctx);pm=read(d/'postings.json');checked(d/'postings.npz',pm['sha256']);before()
    if (d/'history.json').exists():
        checked(d/'history.npz',read(d/'history.json')['sha256']);return {'status':'SHARED_HISTORY_READY','reused':True}
    if (d/'history.npz').exists():raise ValueError('Orphan shared history preserved')
    posts=load_arrays(d/'postings.npz')['postings'];selected=np.unique(posts[:,1])
    old=load_arrays(ctx['round07']/'history.npz')['events'];known=np.unique(old[:,0]);missing=np.setdiff1d(selected,known)
    if len(selected)>400000:raise ValueError('Historical source count exceeded')
    if len(missing):
        con=connect_history(ctx['history'],d/'working/histories')
        try:
            ids_table(con,'selected','session',missing);ids_table(con,'excluded','session',ctx['excluded'])
            new=collect_rows(con.execute(history_sql()),5,10000000-len(old),PROGRESS)
        finally:con.close()
    else:new=np.empty((0,5),np.int64)
    rows=np.concatenate([old,new]);del new
    if len(rows)>10000000:raise ValueError('10000000 retained-event cap; no trimming')
    rows=rows[np.lexsort((rows[:,4],rows[:,0]))]
    h=group_history(rows,HISTORY_END,set(map(int,ctx['excluded'])))
    if set(h)!=set(map(int,selected)):raise ValueError('Missing source sessions')
    # Every posting must be backed by an actual retained event in its source session.
    for anchor,sid,last_ts,df in posts:
        hh=h[int(sid)];m=hh.aids==anchor
        if not m.any() or int(hh.timestamps[m].max())!=last_ts:raise ValueError('Posting/history alignment failure')
    digest=save_arrays(d/'history.npz',events=rows)
    save_json(d/'history.json',{'status':'SHARED_HISTORY_READY','sha256':digest,'source_sessions':len(h),'retained_events':len(rows),
      'round07_sessions_reused':len(known),'new_source_sessions':len(missing),'additional_parquet_scans':int(bool(len(missing))),
      'postings_sha256':pm['sha256'],'source_tail_not_complete_original_sessions':True,
      'excluded_study_sessions':len(ctx['excluded']),'history_end':HISTORY_END,'query_labels_read':False})
    return {'status':'SHARED_HISTORY_READY','sessions':len(h),'events':len(rows),'reused_round07_sessions':len(known)}

def evidence(ctx):
    d=ctx['shared'];a=require(ctx);hm=read(d/'history.json');pm=read(d/'postings.json')
    if hm['postings_sha256']!=pm['sha256'] or pm['cohort_sha256']!=sha(d/'cohort.npz'):raise ValueError('Shared stage lineage changed')
    checked(d/'history.npz',hm['sha256']);checked(d/'postings.npz',pm['sha256'])
    h=group_history(load_arrays(d/'history.npz')['events'],HISTORY_END,set(map(int,ctx['excluded'])))
    rows=load_arrays(d/'postings.npz')['postings'];validate_postings(rows,np.unique(a['anchors'][a['anchors']>=0]),ctx['excluded'])
    lookup={};freq={}
    for anchor,sid,_,df in rows:lookup.setdefault(int(anchor),set()).add(int(sid));freq[int(anchor)]=int(df)
    return a,h,lookup,freq
