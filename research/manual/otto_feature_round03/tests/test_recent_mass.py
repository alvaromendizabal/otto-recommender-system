import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from core import (array_digest, choose_cohort, forward_folds, hits_at20, paired_interval,
                  pooled, save_arrays, save_json, sha, target_arrays, write_once)
from intent_features import seed_pools, summarize_support
from recent_mass_features import (NAMES, catalog, summarize_mass, transform, validate_values, zero_ablation)
from experiment import control_prediction, safe_path, load_prior_module


def fixture():
    g=csr_matrix((np.array([1.,3.,100.]),([1,2,7],[7,7,7])),shape=(10,10))
    return {'symmetric':[g.copy() for _ in range(3)],'forward':[g.copy() for _ in range(3)]}

class MassTests(unittest.TestCase):
    def test_explicit_mass_fraction(self):
        np.testing.assert_allclose(summarize_mass([[1],[3]],[1,2],[7],[0,1]),[[.25,.25,1]])
    def test_no_evidence_is_minus_one(self):
        np.testing.assert_array_equal(summarize_mass([[0]],[1],[7],[0]),[[-1,-1,-1]])
    def test_old_evidence_is_zero_not_missing(self):
        np.testing.assert_array_equal(summarize_mass([[8]],[1],[7],[3]),[[0,0,0]])
    def test_zero_age_evidence_is_one(self):
        np.testing.assert_array_equal(summarize_mass([[8]],[1],[7],[0]),[[1,1,1]])
    def test_window_boundary_inclusive(self):
        np.testing.assert_array_equal(summarize_mass([[1]],[1],[7],[5/60]),[[1,1,1]])
    def test_just_outside_first_boundary(self):
        np.testing.assert_array_equal(summarize_mass([[1]],[1],[7],[5/60+1e-6]),[[0,1,1]])
    def test_self_affinity_excluded(self):
        np.testing.assert_array_equal(summarize_mass([[100],[3]],[7,2],[7],[0,3]),[[0,0,0]])
    def test_self_only_evidence_is_missing(self):
        np.testing.assert_array_equal(summarize_mass([[100]],[7],[7],[0]),[[-1,-1,-1]])
    def test_empty_pool(self):
        np.testing.assert_array_equal(summarize_mass(np.empty((0,1)),np.array([],int),[7],[]),[[-1,-1,-1]])
    def test_weight_scale_invariant(self):
        x=np.array([[1.,2],[3,2]])
        np.testing.assert_array_equal(summarize_mass(x,[1,2],[7,8],[0,1]),summarize_mass(x*100,[1,2],[7,8],[0,1]))
    def test_candidate_permutation(self):
        x=np.array([[1.,2],[3,2]])
        a=summarize_mass(x,[1,2],[7,8],[0,1]);b=summarize_mass(x[:,::-1],[1,2],[8,7],[0,1])
        np.testing.assert_array_equal(a,b[::-1])
    def test_inputs_unchanged(self):
        x=np.array([[100.,1],[3,0]]);before=x.copy();summarize_mass(x,[7,2],[7,8],[0,1]);np.testing.assert_array_equal(x,before)
    def test_negative_graph_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[-1]],[1],[7],[0])
    def test_nan_graph_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[np.nan]],[1],[7],[0])
    def test_negative_age_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[1]],[1],[7],[-1])
    def test_bad_dimensions_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[1,2]],[1],[7],[0])
    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[1,2]],[1],[7,7],[0])
    def test_duplicate_seeds_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[1],[2]],[1,1],[7],[0,1])
    def test_float_ids_rejected(self):
        with self.assertRaises(ValueError):summarize_mass([[1]],[1.2],[7],[0])
    def test_full_schema_36(self):
        self.assertEqual(len(NAMES),36);self.assertEqual(len(set(NAMES)),36);self.assertEqual(len(catalog()),36)
    def test_full_transform(self):
        a=transform([1,2],[100,3600100],[0,1],[7,8],fixture(),100)
        self.assertEqual(a.shape,(2,36));validate_values(a)
    def test_unknown_ids_zero_edges_not_index_error(self):
        a=transform([999],[100],[0],[888],fixture(),100);np.testing.assert_array_equal(a,np.full((1,36),-1))
    def test_cutoff_precedes_prefix(self):
        with self.assertRaises(ValueError):transform([1],[99],[0],[7],fixture(),100)
    def test_wrong_family(self):
        g=fixture();del g['forward']
        with self.assertRaises(ValueError):transform([1],[100],[0],[7],g,100)
    def test_three_channels_required(self):
        g=fixture();g['forward']=g['forward'][:1]
        with self.assertRaises(ValueError):transform([1],[100],[0],[7],g,100)
    def test_square_graph_required(self):
        g=fixture();g['forward'][0]=csr_matrix((10,11))
        with self.assertRaises(ValueError):transform([1],[100],[0],[7],g,100)
    def test_pending_pool_observed_event_order(self):
        p=seed_pools([1,1,2],[100,100,100],[1,2,1],100)
        np.testing.assert_array_equal(p['pending_cart'][0],[2])
    def test_pending_pool_after_new_cart(self):
        p=seed_pools([1,1,1],[100,100,100],[1,2,1],100)
        np.testing.assert_array_equal(p['pending_cart'][0],[1])
    def test_recent_pool_deduplicates(self):
        p=seed_pools([1,2,1],[100,200,300],[0,0,0],300)
        np.testing.assert_array_equal(p['recent_unique'][0],[1,2])
    def test_missing_encoding_ablation(self):
        x=np.tile([[-1,-1,-1,0,.5,1]],(2,6));y=zero_ablation(x)
        self.assertTrue((y>=0).all());self.assertEqual(y[0,0],0);self.assertEqual(y[0,4],.5);self.assertEqual(x[0,0],-1)
    def test_window_monotonicity_required(self):
        with self.assertRaises(ValueError):validate_values(np.tile([[1,.5,1]],(1,12)))
    def test_partial_missing_rejected(self):
        with self.assertRaises(ValueError):validate_values(np.tile([[-1,0,1]],(1,12)))
    def test_nonfinite_feature_rejected(self):
        with self.assertRaises(ValueError):validate_values(np.full((1,36),np.nan))
    def test_fraction_distribution_vs_average(self):
        x=[[1],[1]];ages_a=[0,2];ages_b=[np.sqrt(3)-1]*2
        a=summarize_support(x,[1,2],[7],ages_a);b=summarize_support(x,[1,2],[7],ages_b)
        self.assertAlmostEqual(float(a[0,2]),float(b[0,2]),places=6)
        self.assertNotEqual(float(summarize_mass(x,[1,2],[7],ages_a)[0,0]),float(summarize_mass(x,[1,2],[7],ages_b)[0,0]))
    def test_prefix_future_time_rejected(self):
        with self.assertRaises(ValueError):seed_pools([1,2],[100,200],[0,0],150)

class IntegrityTests(unittest.TestCase):
    def test_immutable_json_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';save_json(p,{'x':1});save_json(p,{'x':1})
            with self.assertRaises(ValueError):save_json(p,{'x':2})
    def test_stable_npz_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';a=np.arange(20);s=save_arrays(p,a=a);self.assertEqual(s,save_arrays(p,a=a))
    def test_refuse_object_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):save_arrays(Path(d)/'a.npz',a=np.array([{}],object))
    def test_symlink_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.symlink_to(Path(d)/'b')
            with self.assertRaises(ValueError):write_once(p,b'no')
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):safe_path(Path(d),'../elsewhere')
    def test_pooled_weights_and_denominators(self):
        self.assertAlmostEqual(pooled(np.array([[1,1,2]]),np.array([[1,2,4]]))['weighted_recall_at_20'],.55)
    def test_reject_hits_above_denominator(self):
        with self.assertRaises(ValueError):pooled(np.array([[1,2,3]]),np.array([[1,1,3]]))
    def test_target_censor_before_sampling(self):
        ids=np.array([1]);a=np.array([[7,8]]);q=np.array([100]);labels={1:[(7,0,110,2),(8,1,150,3)]}
        result=target_arrays(ids,a,q,[1],[[1,1,0]],labels,200,cutoff=150)
        np.testing.assert_array_equal(result,[[[1,0,0],[0,0,0]]])
    def test_unknown_targets_remain_in_full_counts(self):
        a=target_arrays(np.array([1]),np.array([[7]]),np.array([100]),[0],[[1,0,0]],{1:[(99,0,110,1)]},200)
        self.assertEqual(int(a.sum()),0)
    def test_fold_embargo_strict(self):
        ids=np.arange(4096);q=np.arange(4096,dtype=np.int64)*3600_000
        for f in forward_folds(ids,q):self.assertTrue((q[f['train']]<f['cutoff']-6*3600_000).all())
    def test_score_ties_item_id(self):
        aids=np.array([np.arange(30)[::-1]]);truth=(aids==0).astype(int)
        self.assertEqual(int(hits_at20(np.ones((1,30)),aids,truth)[0]),1)
    def test_shape_and_dtype_change_digest(self):
        a=np.array([1,2],np.int64);self.assertNotEqual(array_digest(a),array_digest(a.astype(np.int32)))
    def test_paired_zero_difference(self):
        d=np.ones((40,3),np.int64);c=paired_interval(np.zeros_like(d),d,np.r_[np.zeros(20,int),np.ones(20,int)],100)
        self.assertEqual(c['descriptive_95_interval'],[0.,0.])
    def test_control_native_reuse_and_changed_contract_rejection(self):
        import lightgbm as lgb
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);x=np.arange(80,dtype=float).reshape(40,2);y=np.tile([0,1,0,0],10)
            model=lgb.train({'objective':'lambdarank','verbose':-1,'min_data_in_leaf':1,'num_threads':1,'num_leaves':3},
                            lgb.Dataset(x,label=y,group=[4]*10,feature_name=['a','b']),num_boost_round=2)
            (root/'model.txt').write_text(model.model_to_string());digest=sha(root/'model.txt')
            save_json(root/'receipt.json',{'contract':{'fixture':True},'model_sha256':digest})
            result=control_prediction(root,{'fixture':True},x,digest,['a','b'])
            np.testing.assert_array_equal(result,model.predict(x,num_threads=4))
            with self.assertRaises(ValueError):control_prediction(root,{'fixture':False},x,digest,['a','b'])
    def test_control_hash_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'model.txt').write_text('bad');save_json(p/'receipt.json',{'contract':{},'model_sha256':'wrong'})
            with self.assertRaises(ValueError):control_prediction(p,{},np.zeros((1,2)),'wrong',['a','b'])

if __name__=='__main__':unittest.main()
