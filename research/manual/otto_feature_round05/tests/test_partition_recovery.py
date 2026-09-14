"""Exercise the actual partition/checkpoint driver via a SQLite SQL adapter.

Production DuckDB/Parquet is checked separately in backend_smoke, on the existing AWS
interpreter, before real-data indexing. This adapter is not presented as a DuckDB test.
"""
import json,sqlite3,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core import sha
from transition_features import pair_keys
from transition_index import build,load,IndexPause,CUTOFF
from backend_smoke import EVENTS

class Connection:
    def __init__(self):self.db=sqlite3.connect(':memory:')
    def execute(self,sql):
        if sql.startswith('SET '):return self
        if sql.startswith('CREATE TEMP VIEW events'):
            self.db.execute('CREATE TABLE events(session BIGINT,aid BIGINT,ts BIGINT,event_index BIGINT,event_type BIGINT)')
            # Supply only the pre-cutoff historical fixture, as production source_check requires.
            self.db.executemany('INSERT INTO events VALUES (?,?,?,?,?)',[(s,a,CUTOFF-500+t,i,k) for s,a,t,i,k in EVENTS])
            return self
        self.last=self.db.execute(sql);return self
    def fetchone(self):return self.last.fetchone()
    def fetchall(self):return self.last.fetchall()
    def register(self,name,frame):
        frame.to_sql(name,self.db,index=False,if_exists='replace')
    def unregister(self,name):self.db.execute('DROP TABLE '+name)
    def close(self):self.db.close()

class Recovery(unittest.TestCase):
    def build(self,root,contract=None,log=lambda *a,**k:None,deadline=None):
        src=np.array([1,1,2,3]);dst=np.array([2,3,1,1]);keys=pair_keys(src,dst)
        with patch.dict('sys.modules',{'duckdb':types.SimpleNamespace(connect=Connection)}):
            return build(root/'fixture.parquet',root/'idx',keys,np.array([4]),contract or {'fixture':True},deadline=deadline or time.monotonic()+60,progress={},log=log)
    def test_complete_reuse_same_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.build(r);before={p.name:sha(p) for p in (r/'idx').glob('*.npz')}
            self.build(r);self.assertEqual(before,{p.name:sha(p) for p in (r/'idx').glob('*.npz')})
            index=load(r/'idx');self.assertEqual(len(index.keys),4)
    def test_pause_after_partition_resumes_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)
            def pause(event,**kwargs):
                if kwargs['bucket']==0:raise IndexPause('synthetic bounded pause')
            with self.assertRaises(IndexPause):self.build(r,log=pause)
            h=sha(r/'idx/part-00.npz');receipt=(r/'idx/part-00.json').read_bytes()
            self.build(r);self.assertEqual(h,sha(r/'idx/part-00.npz'));self.assertEqual(receipt,(r/'idx/part-00.json').read_bytes())
    def test_corrupt_committed_shard_stops(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.build(r);p=r/'idx/part-01.npz';p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):self.build(r)
            self.assertEqual(p.read_bytes(),b'corrupt')
    def test_changed_contract_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.build(r);before=(r/'idx/contract.json').read_bytes()
            with self.assertRaises(ValueError):self.build(r,contract={'fixture':'different'})
            self.assertEqual(before,(r/'idx/contract.json').read_bytes())
    def test_orphan_partition_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);(r/'idx').mkdir();(r/'idx/part-00.npz').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'Orphan'):self.build(r)
    def test_time_boundary_does_not_run_new_unit(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)
            with self.assertRaises(IndexPause):self.build(r,deadline=time.monotonic()+.01)
            self.assertEqual(list((r/'idx').glob('part-*.npz')),[])
    def test_completed_index_corruption_rejected_by_reader(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);self.build(r);(r/'idx/part-05.npz').write_bytes(b'bad')
            with self.assertRaises(ValueError):load(r/'idx')
    def test_uninterrupted_and_resumed_counts_equal(self):
        with tempfile.TemporaryDirectory() as x,tempfile.TemporaryDirectory() as y:
            a,b=Path(x),Path(y);self.build(a)
            def pause(event,**kwargs):
                if kwargs['bucket']==2:raise IndexPause('fixture')
            with self.assertRaises(IndexPause):self.build(b,log=pause)
            self.build(b);i,j=load(a/'idx'),load(b/'idx')
            for attr in ('keys','typed','collapsed','source_ids','outgoing','pooled_outgoing'):
                np.testing.assert_array_equal(getattr(i,attr),getattr(j,attr))

if __name__=='__main__':unittest.main()
