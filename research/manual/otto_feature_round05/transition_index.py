"""Count only exact adjacent historical events; preserve each source partition.

SQL extraction accepts an `events` relation with the original event_index. Filtering
source IDs is applied after the adjacency join, never before a LEAD calculation.
The complete outgoing denominator includes targets outside requested candidate pools.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from core import checked, save_arrays, save_json, sha
from transition_features import pair_keys, split_keys, TransitionIndex

PARTITIONS=16
GAP_MS=30*60*1000
CUTOFF=1660687200000
MAX_REQUESTS=10_000_000

class IndexPause(Exception): pass


def extraction_sql(bucket, partitions=PARTITIONS, cutoff=CUTOFF, gap=GAP_MS):
    if (type(bucket) is not int or type(partitions) is not int or not 0<=bucket<partitions
            or type(cutoff) is not int or cutoff<=0 or type(gap) is not int or gap<0):
        raise ValueError('Invalid transition SQL bounds')
    return f'''CREATE TEMP TABLE edges AS
       SELECT DISTINCT a.session, a.aid AS source_aid, b.aid AS target_aid,
                       a.event_type AS source_kind, b.event_type AS target_kind
       FROM events a
       JOIN events b ON a.session=b.session AND b.event_index=CAST(a.event_index AS BIGINT)+1
       JOIN wanted_sources w ON a.aid=w.source_aid
       WHERE a.aid % {partitions} = {bucket}
         AND a.aid<>b.aid AND a.ts>=0 AND b.ts>=a.ts AND b.ts-a.ts<={gap}
         AND a.ts<{cutoff} AND b.ts<{cutoff}
         AND a.event_type IN (0,1,2) AND b.event_type IN (0,1,2)
         AND NOT EXISTS (SELECT 1 FROM excluded x WHERE x.session=a.session)'''


def extract_partition(connection,bucket,partitions=PARTITIONS,cutoff=CUTOFF,gap=GAP_MS):
    """Same relational queries exercised by the portable fixture and AWS DuckDB smoke."""
    connection.execute('DROP TABLE IF EXISTS edges')
    connection.execute('DROP TABLE IF EXISTS untyped_edges')
    connection.execute(extraction_sql(bucket,partitions,cutoff,gap))
    connection.execute('CREATE TEMP TABLE untyped_edges AS SELECT DISTINCT session,source_aid,target_aid FROM edges')
    typed=connection.execute('''SELECT e.source_aid,e.target_aid,e.source_kind,e.target_kind,count(*)
      FROM edges e JOIN requested r ON e.source_aid=r.source_aid AND e.target_aid=r.target_aid
      GROUP BY e.source_aid,e.target_aid,e.source_kind,e.target_kind
      ORDER BY e.source_aid,e.target_aid,e.source_kind,e.target_kind''').fetchall()
    pooled=connection.execute('''SELECT e.source_aid,e.target_aid,count(*)
      FROM untyped_edges e JOIN requested r ON e.source_aid=r.source_aid AND e.target_aid=r.target_aid
      GROUP BY e.source_aid,e.target_aid ORDER BY e.source_aid,e.target_aid''').fetchall()
    outgoing=connection.execute('''SELECT source_aid,source_kind,count(*) FROM edges
      GROUP BY source_aid,source_kind ORDER BY source_aid,source_kind''').fetchall()
    pooled_out=connection.execute('''SELECT source_aid,count(*) FROM untyped_edges
      GROUP BY source_aid ORDER BY source_aid''').fetchall()
    rows=int(connection.execute('SELECT count(*) FROM edges').fetchone()[0])
    return {'typed':typed,'pooled':pooled,'outgoing':outgoing,'pooled_out':pooled_out,'distinct_typed_support_units':rows}


def pack_partition(keys, source_ids, result):
    s,t=split_keys(keys);n=len(keys)
    counts=np.zeros((n,9),np.int64);pooled=np.zeros(n,np.int64)
    den=np.zeros((len(source_ids),3),np.int64);pd=np.zeros(len(source_ids),np.int64)
    for rows,kind in ((result['typed'],'typed'),(result['pooled'],'pooled')):
        if not rows: continue
        arr=np.asarray(rows,np.int64);pk=pair_keys(arr[:,0],arr[:,1]);ix=np.searchsorted(keys,pk)
        if (ix>=n).any() or not np.array_equal(keys[ix],pk):raise ValueError('Unexpected query pair in counts')
        if kind=='typed':counts[ix,arr[:,2]*3+arr[:,3]]=arr[:,4]
        else:pooled[ix]=arr[:,2]
    for rows,kind in ((result['outgoing'],'typed'),(result['pooled_out'],'pooled')):
        if not rows:continue
        arr=np.asarray(rows,np.int64);ix=np.searchsorted(source_ids,arr[:,0])
        if (ix>=len(source_ids)).any() or not np.array_equal(source_ids[ix],arr[:,0]):
            raise ValueError('Unexpected denominator source')
        if kind=='typed':den[ix,arr[:,1]]=arr[:,2]
        else:pd[ix]=arr[:,1]
    TransitionIndex(keys,counts,pooled,source_ids,den,pd)
    return dict(keys=keys,typed=counts,collapsed=pooled,source_ids=source_ids,outgoing=den,pooled_outgoing=pd)


def build(history, output, keys, excluded_ids, contract, *, deadline, progress, log):
    import duckdb
    import pandas as pd
    history=Path(history);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if keys.dtype!=np.uint64 or (keys[1:]<=keys[:-1]).any() or not len(keys) or len(keys)>MAX_REQUESTS:
        raise ValueError('Invalid or excessive transition requests; no silent trimming')
    save_json(output/'contract.json',contract);cid=sha(output/'contract.json')
    source,target=split_keys(keys);sources=np.unique(source)
    con=duckdb.connect()
    receipts=[]
    try:
        con.execute('SET threads=4');con.execute("SET memory_limit='6GB'")
        con.execute('SET preserve_insertion_order=false')
        work=output/'working';work.mkdir(exist_ok=True)
        con.execute("SET temp_directory='"+str(work).replace("'","''")+"'")
        con.execute("SET max_temp_directory_size='3GB'")
        path=str(history).replace("'","''")
        con.execute(f"CREATE TEMP VIEW events AS SELECT session,aid,ts,event_type,event_index FROM read_parquet('{path}')")
        excluded=pd.DataFrame({'session':np.asarray(excluded_ids,np.int64)})
        con.register('excluded',excluded)
        certificate=output/'source_check.json'
        if certificate.exists():
            audit=json.loads(certificate.read_text())
            if audit['contract_id']!=cid:raise ValueError('Source-check lineage differs')
        else:
            progress.update(stage='transition_source_check',completed=0,total=PARTITIONS)
            row=con.execute('''SELECT count(*),min(ts),max(ts),
              sum(CASE WHEN aid<0 OR aid>=2147483648 OR ts<0 OR event_index<0 OR event_type NOT IN (0,1,2)
               OR aid IS NULL OR ts IS NULL OR event_index IS NULL OR session IS NULL OR event_type IS NULL THEN 1 ELSE 0 END)
              FROM events''').fetchone()
            if not row or row[0]<=0 or row[1]<0 or row[2]>=CUTOFF or row[3]:
                raise ValueError('Historical tail schema or strict cutoff failed')
            audit={'contract_id':cid,'rows':int(row[0]),'min_ts':int(row[1]),'max_ts':int(row[2]),
                   'cutoff':CUTOFF,'historical_tail_not_full_raw_source':True}
            save_json(certificate,audit)
        for bucket in range(PARTITIONS):
            path=output/f'part-{bucket:02d}.npz';rp=path.with_suffix('.json')
            if rp.exists():
                r=json.loads(rp.read_text())
                if r['contract_id']!=cid or r['bucket']!=bucket:raise ValueError('Wrong index partition contract')
                checked(path,r['sha256']);receipts.append(r)
                progress.update(stage='transition_partition',completed=len(receipts),total=PARTITIONS)
                continue
            if time.monotonic()>deadline-35:
                raise IndexPause('Completed transition partitions preserved; return the report before further work')
            if path.exists():raise ValueError('Orphan index partition preserved; return report')
            tick=time.monotonic();mask=source%PARTITIONS==bucket;kk=keys[mask]
            ss=source[mask];tt=target[mask];source_ids=np.unique(ss)
            requests=pd.DataFrame({'source_aid':ss,'target_aid':tt})
            wanted=pd.DataFrame({'source_aid':source_ids})
            con.register('requested',requests);con.register('wanted_sources',wanted)
            result=extract_partition(con,bucket)
            arrays=pack_partition(kk,source_ids,result)
            digest=save_arrays(path,**arrays)
            r={'contract_id':cid,'bucket':bucket,'path':path.name,'sha256':digest,
               'request_pairs':len(kk),'sources':len(source_ids),
               'typed_support_units_including_nonrequested_targets':result['distinct_typed_support_units'],
               'supported_request_pairs':int((arrays['collapsed']>0).sum()),
               'elapsed_seconds':time.monotonic()-tick}
            save_json(rp,r);receipts.append(r)
            progress.update(stage='transition_partition',completed=len(receipts),total=PARTITIONS)
            log('transition_partition_committed',**r)
            con.execute('DROP TABLE edges');con.execute('DROP TABLE untyped_edges')
            con.unregister('requested');con.unregister('wanted_sources')
            del arrays,result,requests,wanted
        stable={'status':'ROUND05_INDEX_READY','contract_id':cid,'source_check':audit,
                'partitions':receipts,'request_pairs':len(keys),'source_count':len(sources),
                'old_graphs_rebuilt':0,'new_transition_index':True,'model_fits':0,
                'cutoff_ms':CUTOFF,'maximum_adjacent_gap_ms':GAP_MS,
                'study_sessions_excluded':len(excluded_ids),'full_outgoing_denominators':True}
        save_json(output/'index_manifest.json',stable)
        return stable
    finally:con.close()


def load(output):
    root=Path(output);m=json.loads((root/'index_manifest.json').read_text())
    if m['status']!='ROUND05_INDEX_READY' or len(m['partitions'])!=PARTITIONS:
        raise ValueError('Complete transition index required')
    aa=[]
    for j,row in enumerate(m['partitions']):
        p=root/row['path']
        if row['bucket']!=j or p.is_symlink() or p.parent!=root:raise ValueError('Invalid index path/order')
        checked(p,row['sha256'])
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        s,_=split_keys(a['keys'])
        if (s%PARTITIONS!=j).any():raise ValueError('Wrong source partition')
        TransitionIndex(**a);aa.append(a)
    combined={k:np.concatenate([x[k] for x in aa],axis=0) for k in aa[0]}
    order=np.argsort(combined['keys'],kind='stable')
    for k in ('keys','typed','collapsed'):combined[k]=combined[k][order]
    ix=np.argsort(combined['source_ids'],kind='stable')
    for k in ('source_ids','outgoing','pooled_outgoing'):combined[k]=combined[k][ix]
    return TransitionIndex(**combined)
