"""Shared-source extraction tested with the identical SQL via a labeled SQLite adapter.
The mandatory AWS test exercises actual DuckDB/Parquet separately.
"""
import json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import shared_data as s
from core import save_arrays,save_json,sha

class SharedContracts(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
  self.out=root/'shared';self.old=root/'round07';self.out.mkdir();self.old.mkdir()
  self.rows=np.array([(1,10,10,0,0),(1,30,11,2,1),(2,20,20,0,0),(2,40,21,1,1),(3,20,22,0,0),(3,50,23,2,4),
       (99,20,30,0,0),(100,20,s.HISTORY_END,0,0)],np.int64)
  save_arrays(self.old/'postings.npz',postings=np.array([[10,1,10,1]],np.int64))
  save_arrays(self.old/'history.npz',events=self.rows[:2]);save_arrays(self.old/'pilot.npz',anchors=np.array([[10,-1,-1,-1]]))
  self.a={'ids':np.array([1000,1001]),'anchors':np.array([[10,20,-1,-1],[20,-1,-1,-1]]),'candidates':np.tile(np.arange(400),(2,1))}
  save_arrays(self.out/'cohort.npz',**self.a)
  self.ctx={'shared':self.out,'round07':self.old,'ids':self.a['ids'],'excluded':np.array([99]),'history':root/'source.parquet'}
  self.calls=0
  def connect(*args):
   self.calls+=1;con=sqlite3.connect(':memory:')
   con.execute('CREATE TABLE events(session BIGINT,aid BIGINT,ts BIGINT,event_type BIGINT,event_index BIGINT)')
   con.executemany('INSERT INTO events VALUES(?,?,?,?,?)',[tuple(map(int,row)) for row in self.rows]);return con
  self.patches=[mock.patch.object(s,'require',return_value=self.a),mock.patch.object(s,'connect_history',side_effect=connect),
      mock.patch.object(s,'backend_smoke',return_value={'status':'SQLITE_TEST_ADAPTER_NOT_DUCKDB'})]
  for p in self.patches:p.start();self.addCleanup(p.stop)
 def test_incremental_two_scans_and_reuse(self):
  s.postings(self.ctx);s.histories(self.ctx)
  self.assertEqual(self.calls,2)
  p=s.load_arrays(self.out/'postings.npz')['postings'];h=s.load_arrays(self.out/'history.npz')['events']
  self.assertEqual(p.tolist(),[[10,1,10,1],[20,2,20,2],[20,3,22,2]])
  self.assertEqual(h.tolist(),self.rows[:6].tolist());self.assertNotIn(99,h[:,0]);self.assertNotIn(100,h[:,0])
  s.postings(self.ctx);s.histories(self.ctx);self.assertEqual(self.calls,2)
  a,h,lookup,freq=s.evidence(self.ctx);self.assertEqual(freq,{10:1,20:2});self.assertEqual(lookup[20],{2,3})
 def test_corrupt_posting_stops_without_scan(self):
  s.postings(self.ctx);(self.out/'postings.npz').write_bytes(b'bad')
  with self.assertRaises(ValueError):s.postings(self.ctx)
  self.assertEqual(self.calls,1)
 def test_corrupt_history_stops(self):
  s.postings(self.ctx);s.histories(self.ctx);(self.out/'history.npz').write_bytes(b'bad')
  with self.assertRaises(ValueError):s.histories(self.ctx)
 def test_orphan_posting_is_preserved(self):
  (self.out/'postings.npz').write_bytes(b'orphan')
  with self.assertRaises(ValueError):s.postings(self.ctx)
  self.assertEqual((self.out/'postings.npz').read_bytes(),b'orphan')
 def test_orphan_history_is_preserved(self):
  s.postings(self.ctx);(self.out/'history.npz').write_bytes(b'orphan')
  with self.assertRaises(ValueError):s.histories(self.ctx)
 def test_duplicate_posting_rejected(self):
  x=np.array([[10,1,1,2],[10,1,1,2]],np.int64)
  with self.assertRaises(ValueError):s.validate_postings(x,[10],[])
 def test_frequency_under_count_rejected(self):
  with self.assertRaises(ValueError):s.validate_postings(np.array([[10,1,1,1],[10,2,2,1]]),[10],[])
 def test_source_count_not_capped_frequency(self):
  s.validate_postings(np.array([[10,1,1,100000]]),[10],[])
 def test_posting_cutoff_strict(self):
  with self.assertRaises(ValueError):s.validate_postings(np.array([[10,1,s.HISTORY_END,1]]),[10],[])
 def test_noninteger_posting_rejected(self):
  with self.assertRaises(ValueError):s.validate_postings(np.array([[10.,1,1,1]]),[10],[])
 def test_unrequested_anchor_rejected(self):
  with self.assertRaises(ValueError):s.validate_postings(np.array([[99,1,1,1]]),[10],[])
 def test_study_session_posting_rejected(self):
  with self.assertRaises(ValueError):s.validate_postings(np.array([[10,99,1,1]]),[10],[99])
