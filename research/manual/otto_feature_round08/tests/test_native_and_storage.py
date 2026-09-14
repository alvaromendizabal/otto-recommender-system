import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import save_arrays,save_json,sha,write_once
from experiment import control_prediction

class Storage(unittest.TestCase):
    def test_atomic_json_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.json';save_json(p,{'x':1});h=sha(p);save_json(p,{'x':1});self.assertEqual(h,sha(p))
            with self.assertRaises(ValueError):save_json(p,{'x':2})
    def test_npz_byte_identical_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';x=np.arange(50);h=save_arrays(p,x=x);self.assertEqual(h,save_arrays(p,x=x))
    def test_symlink_write_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p';p.symlink_to(Path(d)/'elsewhere')
            with self.assertRaises(ValueError):write_once(p,b'x')
    def test_object_array_rejected(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):save_arrays(Path(d)/'a.npz',x=np.array([{}],dtype=object))

class NativeReplay(unittest.TestCase):
    def test_control_reload_exact_and_mismatched_contract_rejected(self):
        import lightgbm as lgb
        rng=np.random.default_rng(193);x=rng.normal(size=(80,3));y=np.tile([0,0,0,1],20)
        model=lgb.train({'objective':'lambdarank','verbosity':-1,'num_threads':1,'num_leaves':3,
                         'min_data_in_leaf':2,'deterministic':True,'force_col_wise':True},
            lgb.Dataset(x,label=y,group=[4]*20,feature_name=['a','b','c']),num_boost_round=3)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model.save_model(str(p/'model.txt'));h=sha(p/'model.txt')
            save_json(p/'receipt.json',{'contract':{'fixture':True},'model_sha256':h})
            pred=control_prediction(p,{'fixture':True},x,h,['a','b','c'])
            np.testing.assert_array_equal(pred,model.predict(x,num_threads=1))
            with self.assertRaises(ValueError):control_prediction(p,{'fixture':False},x,h,['a','b','c'])
            with self.assertRaises(ValueError):control_prediction(p,{'fixture':True},x,h,['b','a','c'])

if __name__=='__main__':unittest.main()
