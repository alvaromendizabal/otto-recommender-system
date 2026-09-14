"""Portable relational extraction; DuckDB is mandatory for real Parquet data.

Unit tests execute the IDENTICAL SQL in SQLite against an independent oracle.
backend_smoke() separately checks installed DuckDB and tiny Parquet files in AWS.
No fallback database is used for private data.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np

CUTOFF=1660687200000
MAX_POSTINGS=65536
MAX_HISTORY_ROWS=2_000_000


def posting_sql(cutoff: int=CUTOFF, cap: int=64) -> str:
    if type(cutoff) is not int or cutoff<=0 or type(cap) is not int or not 1<=cap<=64:
        raise ValueError('Invalid frozen SQL bounds')
    return f'''WITH unique_sessions AS (
      SELECT e.aid AS anchor, e.session, MAX(e.ts) AS last_ts
      FROM events e JOIN wanted w ON e.aid=w.anchor
      WHERE e.ts>=0 AND e.ts<{cutoff}
        AND NOT EXISTS (SELECT 1 FROM excluded x WHERE x.session=e.session)
      GROUP BY e.aid,e.session
    ), ranked AS (
      SELECT anchor,session,last_ts,
        COUNT(*) OVER (PARTITION BY anchor) AS available_sessions,
        ROW_NUMBER() OVER (PARTITION BY anchor ORDER BY last_ts DESC,session ASC) AS position
      FROM unique_sessions
    ) SELECT anchor,session,last_ts,available_sessions FROM ranked
      WHERE position<={cap} ORDER BY anchor,session'''


def history_sql(cutoff: int=CUTOFF) -> str:
    if type(cutoff) is not int or cutoff<=0:raise ValueError('Invalid history cutoff')
    return f'''SELECT e.session,e.aid,e.ts,e.event_type,e.event_index
      FROM events e JOIN selected s ON e.session=s.session
      WHERE e.ts>=0 AND e.ts<{cutoff}
        AND NOT EXISTS (SELECT 1 FROM excluded x WHERE x.session=e.session)
      ORDER BY e.session,e.event_index'''


def ids_table(connection, name: str, column: str, values) -> None:
    if (name,column) not in (('wanted','anchor'),('excluded','session'),('selected','session')):
        raise ValueError('Unsafe SQL table name')
    a=np.asarray(values)
    if a.ndim!=1 or (len(a) and (a.dtype.kind not in 'iu' or (a<0).any() or np.unique(a).size!=len(a))):
        raise ValueError('Distinct nonnegative SQL identifiers required')
    connection.execute(f'CREATE TEMP TABLE {name} ({column} BIGINT PRIMARY KEY)')
    if len(a):connection.executemany(f'INSERT INTO {name} VALUES (?)',[(int(v),) for v in a])


def collect_rows(cursor, columns: int, cap: int, progress=None) -> np.ndarray:
    chunks=[];count=0
    while True:
        rows=cursor.fetchmany(32768)
        if not rows:break
        count+=len(rows)
        if count>cap:raise ValueError('Historical row bound exceeded; no truncation or automatic retry')
        a=np.asarray(rows)
        if a.dtype.kind not in 'iu' or a.shape!=(len(rows),columns):raise ValueError('Noninteger/NULL historical data')
        chunks.append(a.astype(np.int64))
        if progress is not None:progress['rows_loaded']=count
    return np.concatenate(chunks) if chunks else np.empty((0,columns),np.int64)


def connect_history(history: Path, working: Path):
    import duckdb
    working.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect()
    try:
        con.execute('SET threads=4');con.execute("SET memory_limit='6GB'")
        con.execute("SET max_temp_directory_size='3GB'")
        con.execute("SET temp_directory='"+str(working).replace("'","''")+"'")
        con.execute('SET preserve_insertion_order=false')
        con.execute("CREATE TEMP VIEW events AS SELECT session,aid,ts,event_type,event_index FROM read_parquet('"+str(history).replace("'","''")+"')")
        return con
    except BaseException:
        con.close();raise


def backend_smoke() -> dict:
    import duckdb
    # Include repeat events, tied anchor times, an excluded session and cutoff equality.
    data=[(1,10,5,0,0),(1,10,6,0,1),(1,20,8,2,4),(2,10,6,0,0),
          (2,30,7,1,1),(3,10,9,0,0),(4,10,10,0,0),(5,99,5,0,0)]
    with tempfile.TemporaryDirectory(prefix='otto-round07-backend-') as td:
        path=Path(td)/'tiny.parquet';con=duckdb.connect()
        try:
            con.execute('CREATE TABLE fixture(session BIGINT,aid BIGINT,ts BIGINT,event_type BIGINT,event_index BIGINT)')
            con.executemany('INSERT INTO fixture VALUES (?,?,?,?,?)',data)
            con.execute("COPY fixture TO '"+str(path).replace("'","''")+"' (FORMAT PARQUET)")
            con.execute("CREATE TEMP VIEW events AS SELECT * FROM read_parquet('"+str(path).replace("'","''")+"')")
            ids_table(con,'wanted','anchor',[10]);ids_table(con,'excluded','session',[3])
            posting=collect_rows(con.execute(posting_sql(10,1)),4,10)
            if posting.tolist()!=[[10,1,6,2]]:raise ValueError('Actual DuckDB posting oracle mismatch')
            ids_table(con,'selected','session',[1]);history=collect_rows(con.execute(history_sql(10)),5,10)
            if history.tolist()!=[list(x) for x in data[:3]]:raise ValueError('Actual Parquet history oracle mismatch')
            from neighbors import group_history,build_query
            hh=group_history(history,10,{3})
            built=build_query(np.array([10]),np.array([20,30]),[1],hh)
            if built['session_context']['features'][0,11]<=0:raise ValueError('Backend feature smoke failed')
        finally:con.close()
    return {'status':'DUCKDB_PARQUET_SMOKE_PASSED','duckdb_version':duckdb.__version__,'synthetic_rows':len(data),'new_model_fits':0}
