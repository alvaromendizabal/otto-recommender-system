import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from metrics import censored_truth,audit_arrays
from core import save_arrays,save_json,sha
import launch

class CensorTests(unittest.TestCase):
    def call(self,labels,counts=(1,0,0),cutoff=15):return censored_truth([1],[10],[2],[counts],{1:labels},30,cutoff)
    def test_future_excluded(self):self.assertEqual(self.call([(9,0,18,3)])[0][0],set())
    def test_cutoff_equality_excluded(self):self.assertEqual(self.call([(9,0,15,3)])[0][0],set())
    def test_equal_query_time_later_event_allowed(self):self.assertEqual(self.call([(9,0,10,3)])[0][0],{9})
    def test_prior_event_rejected(self):
        with self.assertRaises(ValueError):self.call([(9,0,10,2)])
    def test_wrong_complete_counts_rejected(self):
        with self.assertRaises(ValueError):self.call([(9,0,10,3)],(0,0,0))
    def test_duplicate_target_rejected(self):
        with self.assertRaises(ValueError):self.call([(9,0,10,3),(9,0,11,4)])
    def test_query_reaches_cutoff(self):
        with self.assertRaises(ValueError):self.call([(9,0,11,3)],cutoff=10)
    def test_period_end_excluded(self):
        with self.assertRaises(ValueError):self.call([(9,0,30,3)])

class OracleTests(unittest.TestCase):
    def fixture(self):
        a=np.arange(400,dtype=np.int64)[None];x=np.zeros((1,400,12),np.float32);p=np.full((1,3,100),-1,np.int64)
        return a,x,p
    def test_expansion_uses_missing_item(self):
        a,x,p=self.fixture();p[0,0,0]=999
        r,s=audit_arrays(a,x,x,p,p,[({999},set(),set())])
        self.assertEqual(r['arms']['session_context']['net_capped_hits'],[1,0,0]);self.assertIsNone(r['achieved_ranker_score'])
    def test_zero_denominator_is_not_fake_weighted_metric(self):
        a,x,p=self.fixture();r,_=audit_arrays(a,x,x,p,p,[(set(),set(),set())]);self.assertIsNone(r['baseline_candidate_oracle']['weighted_oracle_at20'])
    def test_capped_20_not_uncapped_coverage(self):
        a,x,p=self.fixture();p[0,2,0]=999
        r,_=audit_arrays(a,x,x,p,p,[({1},set(),set(range(25))|{999})])
        self.assertEqual(r['baseline_candidate_oracle']['hits'][2],20);self.assertEqual(r['arms']['session_context']['net_capped_hits'][2],0)
        self.assertEqual(r['arms']['session_context']['new_distinct_target_hits'][2],1)
    def test_duplicate_proposals_rejected(self):
        a,x,p=self.fixture();p[0,0,:2]=999
        with self.assertRaises(ValueError):audit_arrays(a,x,x,p,p,[({999},set(),set())])
    def test_duplicate_baseline_rejected(self):
        a,x,p=self.fixture();a[0,1]=a[0,0]
        with self.assertRaises(ValueError):audit_arrays(a,x,x,p,p,[(set(),set(),set())])
    def test_positive_support_uses_action(self):
        a,x,p=self.fixture();x[0,9,4]=1
        r,_=audit_arrays(a,x,x,p,p,[({9},{9},set())]);self.assertEqual(r['arms']['session_context']['supported_baseline_positive_items'],[0,1,0])

class StorageTests(unittest.TestCase):
    def test_arrays_replay_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.npz';h=save_arrays(p,x=np.arange(5));self.assertEqual(save_arrays(p,x=np.arange(5)),h)
    def test_json_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.json';save_json(p,{'x':1})
            with self.assertRaises(ValueError):save_json(p,{'x':2})
            self.assertEqual(json.loads(p.read_text()),{'x':1})
    def test_array_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.npz';save_arrays(p,x=np.arange(3));h=sha(p)
            with self.assertRaises(ValueError):save_arrays(p,x=np.arange(4))
            self.assertEqual(sha(p),h)
    def test_object_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):save_arrays(Path(td)/'a.npz',x=np.array(['x'],object))
    def test_collect_excludes_private_large_arrays(self):
        import zipfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'outputs';out.mkdir();(root/'MANIFEST.sha256').write_text('')
            for name in ('pilot.npz','history.npz','postings.npz'):(out/name).write_bytes(b'private')
            (out/'result.json').write_text('{}')
            with patch.object(launch,'ROOT',root):bundle=launch.collect()
            with zipfile.ZipFile(bundle) as z:
                self.assertIn('outputs/result.json',z.namelist());self.assertNotIn('outputs/history.npz',z.namelist())
    def test_failed_run_blocks_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'outputs/runs';out.mkdir(parents=True);(root/'MANIFEST.sha256').write_text('')
            (out/'launcher-tests-1.json').write_text(json.dumps({'exit_code':1,'reason':'failed'}))
            with patch.object(launch,'ROOT',root), self.assertRaisesRegex(RuntimeError,'PRIOR_STOP_DETECTED'):launch._run_stage('prepare')
    def test_missing_prerequisite_blocks_work(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'MANIFEST.sha256').write_text('')
            with patch.object(launch,'ROOT',root), self.assertRaisesRegex(RuntimeError,'Complete the prepare'):launch._run_stage('postings')
    def test_changed_code_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);f=root/'one.py';f.write_text('one');(root/'MANIFEST.sha256').write_text(sha(f)+'  one.py\n');f.write_text('two')
            with patch.object(launch,'ROOT',root),self.assertRaisesRegex(ValueError,'Package file changed'):launch.package_check()
