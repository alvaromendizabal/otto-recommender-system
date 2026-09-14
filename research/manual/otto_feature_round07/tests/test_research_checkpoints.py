"""Exercise real feature/audit stages with synthetic arrays and mocked source preflight.

This does not claim replay of private AWS corpus contracts or DuckDB/Parquet.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import research
from core import save_arrays,save_json,sha

class StageTests(unittest.TestCase):
    def fixture(self,root):
        n=256;ids=np.arange(1000,1000+n,dtype=np.int64)
        p={'ids':ids,'anchors':np.tile([20,10,-1,-1],(n,1)).astype(np.int64),'candidates':np.tile(np.arange(400,dtype=np.int64),(n,1)),
           'baseline_signals':np.zeros((n,400,7),np.float32),'query_ts':np.full(n,11,np.int64),'last_index':np.zeros(n,np.int64),'true_counts':np.ones((n,3),np.int64)}
        save_json(root/'input_contract.json',{'synthetic':True});ph=save_arrays(root/'pilot.npz',**p)
        meta={'sha256':ph,'cutoff_ms':18};save_json(root/'pilot.json',meta)
        source=np.array([[1,10,1,0,0],[1,20,2,1,1],[1,900,3,2,5],[2,10,4,0,0],[2,800,5,1,1]],np.int64)
        posts=np.array([[10,1,1,2],[10,2,4,2],[20,1,2,1]],np.int64)
        hs=save_arrays(root/'history.npz',events=source);ps=save_arrays(root/'postings.npz',postings=posts)
        save_json(root/'history.json',{'sha256':hs,'source_sessions':2,'retained_events':5});save_json(root/'postings.json',{'sha256':ps,'requested_anchors':2})
        labels={int(i):[(10,0,12,1),(800,1,13,2),(900,2,14,3)] for i in ids}
        ctx={'excluded':ids,'paths':{},'prior':SimpleNamespace(labels_for=lambda paths,queryids:(labels,'synthetic-label-sha'))}
        return p,meta,ctx
    def test_feature_stage_reuse_and_audit(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);p,m,ctx=self.fixture(out)
            with patch.object(research,'OUT',out),patch.object(research,'require',return_value=(m,p)),patch.object(research,'DEADLINE',float('inf')):
                a=research.features(ctx);self.assertEqual(a['status'],'ROUND07_FEATURES_READY')
                h=sha(out/'features/part-00.npz');self.assertEqual(research.features(ctx)['status'],'ROUND07_FEATURES_READY');self.assertEqual(sha(out/'features/part-00.npz'),h)
                result=research.audit(ctx);self.assertEqual(result['status'],'ROUND07_AUDIT_READY');self.assertEqual(result['new_model_fits'],0)
    def test_tampered_feature_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);p,m,ctx=self.fixture(out)
            with patch.object(research,'OUT',out),patch.object(research,'require',return_value=(m,p)),patch.object(research,'DEADLINE',float('inf')):
                research.features(ctx);path=out/'features/part-00.npz';path.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Missing or changed'):research.features(ctx)
                self.assertEqual(path.read_bytes(),b'corrupt')
    def test_orphan_checkpoint_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);p,m,ctx=self.fixture(out);(out/'features').mkdir();path=out/'features/part-00.npz';path.write_bytes(b'orphan')
            with patch.object(research,'OUT',out),patch.object(research,'require',return_value=(m,p)),patch.object(research,'DEADLINE',float('inf')):
                with self.assertRaisesRegex(ValueError,'Orphan'):research.features(ctx)
                self.assertEqual(path.read_bytes(),b'orphan')
