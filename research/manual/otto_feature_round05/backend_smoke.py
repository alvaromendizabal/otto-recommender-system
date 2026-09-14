"""Tiny, mandatory backend check before any real-data transition query; no cloud calls."""
from __future__ import annotations
from collections import Counter
from transition_index import extract_partition

# session, aid, timestamp, original event index, event type
EVENTS = [
 (1,1,100,0,0),(1,2,110,1,1),(1,1,120,2,0),(1,2,130,3,1),
 (1,1,140,4,1),(1,2,150,5,2),  # same item pair, different type, collapsed only once
 (2,1,100,0,0),(2,99,110,1,0),(2,3,120,2,2), # must not skip 99
 (3,1,100,0,0),(3,2,100,1,1), # equal timestamps, strictly later index
 (4,1,100,0,0),(4,2,150,1,1), # excluded whole session
 (5,1,190,0,0),(5,2,200,1,1), # exclusive cutoff
 (6,1,100,0,0),(6,2,111,1,1), # above a ten-ms gap
 (7,1,100,0,0),(7,1,101,1,2), # same item excluded
 (8,1,100,0,0),(8,2,101,2,1), # missing index is not an adjacent event
 (9,2,100,0,1),(9,1,110,1,0), # reverse direction
]
REQUESTS=[(1,2),(2,1),(1,3),(3,1)]
EXCLUDED=[4]

def reference(events,requests,excluded,cutoff,gap,bucket,parts):
    source={a for a,b in requests};lookup={}
    for s,a,t,i,k in events:
        if (s,i) in lookup:raise ValueError('duplicate original event index in fixture')
        lookup[(s,i)]=(a,t,k)
    edges=set()
    for (s,i),(a,t,k) in lookup.items():
        nxt=lookup.get((s,i+1))
        if nxt is None:continue
        b,u,j=nxt
        if s not in excluded and a in source and a%parts==bucket and a!=b and 0<=t<=u<cutoff and u-t<=gap:
            edges.add((s,a,b,k,j))
    collapsed={(s,a,b) for s,a,b,k,j in edges};req=set(requests)
    typed=Counter((a,b,k,j) for s,a,b,k,j in edges if (a,b) in req)
    pooled=Counter((a,b) for s,a,b in collapsed if (a,b) in req)
    out=Counter((a,k) for s,a,b,k,j in edges);pout=Counter(a for s,a,b in collapsed)
    return {'typed':[(*k,v) for k,v in sorted(typed.items())],
            'pooled':[(*k,v) for k,v in sorted(pooled.items())],
            'outgoing':[(*k,v) for k,v in sorted(out.items())],
            'pooled_out':[(k,v) for k,v in sorted(pout.items())],
            'distinct_typed_support_units':len(edges)}

def populate(con,events=EVENTS,requests=REQUESTS,excluded=EXCLUDED):
    con.execute('CREATE TABLE events(session BIGINT,aid BIGINT,ts BIGINT,event_index BIGINT,event_type BIGINT)')
    con.executemany('INSERT INTO events VALUES (?,?,?,?,?)',[(s,a,t,i,k) for s,a,t,i,k in events])
    con.execute('CREATE TABLE requested(source_aid BIGINT,target_aid BIGINT)')
    con.executemany('INSERT INTO requested VALUES (?,?)',requests)
    con.execute('CREATE TABLE wanted_sources(source_aid BIGINT)')
    con.executemany('INSERT INTO wanted_sources VALUES (?)',[(s,) for s in sorted({a for a,b in requests})])
    con.execute('CREATE TABLE excluded(session BIGINT)')
    if excluded:con.executemany('INSERT INTO excluded VALUES (?)',[(s,) for s in excluded])

def run():
    import duckdb
    con=duckdb.connect()
    try:
        populate(con)
        for bucket in (0,1):
            actual=extract_partition(con,bucket,partitions=2,cutoff=200,gap=10)
            expected=reference(EVENTS,REQUESTS,EXCLUDED,200,10,bucket,2)
            if actual!=expected:raise ValueError('DuckDB adjacency/type/denominator fixture differs from Python oracle')
        # Verify the exact Parquet loading interface separately on the synthetic events.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            p=(Path(td)/'events.parquet').as_posix().replace("'","''")
            con.execute(f"COPY events TO '{p}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            if con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]!=len(EVENTS):
                raise ValueError('DuckDB synthetic Parquet roundtrip failed')
        return {'status':'DUCKDB_TRANSITION_SMOKE_PASSED','duckdb':duckdb.__version__,'synthetic_events':len(EVENTS),'model_fits':0}
    finally:con.close()

if __name__=='__main__':
    import json
    print(json.dumps(run()),flush=True)
