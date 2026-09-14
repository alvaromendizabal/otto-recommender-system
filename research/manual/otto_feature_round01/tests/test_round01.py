"""Offline numerical, leakage-boundary, persistence, metric and plotting contracts.

Real Polars/Parquet loading in the user's locked runtime is not claimed by these tests.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import intent_features as f
import experiment as e


class IntentTests(unittest.TestCase):
    def prefix(self):
        return np.array([1,2,1,2,3,1]),np.arange(6,dtype=np.int64)*3600000,np.array([0,1,1,2,1,0])
    def graphs(self):
        g=csr_matrix((np.array([1.,2.,3.,4.]),([1,2,3,1],[4,4,4,1])),shape=(8,8))
        return {n:[g.copy() for _ in range(3)] for n in f.FAMILIES}
    def test_schema_and_groups(self):
        self.assertEqual(len(f.NAMES),48);self.assertEqual(len(set(f.NAMES)),48)
        self.assertEqual(len(f.SUPPORT),24);self.assertEqual(len(f.TIMING),24)
        self.assertEqual(set(f.SUPPORT)|set(f.TIMING),set(range(48)))
    def test_pending_means_cart_after_last_observed_order(self):
        a,t,k=self.prefix();p=f.seed_pools(a,t,k,t[-1])
        np.testing.assert_array_equal(p['pending_cart'][0],[1,3])
    def test_later_cart_reopens_observed_pending_state(self):
        a=np.array([1,1,1]);t=np.array([1,2,3]);k=np.array([1,2,1])
        np.testing.assert_array_equal(f.seed_pools(a,t,k,3)['pending_cart'][0],[1])
    def test_order_does_not_claim_abandonment(self):
        a=np.array([1,1,1]);t=np.array([1,2,3]);k=np.array([1,2,0])
        self.assertEqual(len(f.seed_pools(a,t,k,3)['pending_cart'][0]),0)
    def test_recency_is_most_recent_observed_event(self):
        a,t,k=self.prefix();p=f.seed_pools(a,t,k,t[-1])
        np.testing.assert_array_equal(p['recent_unique'][0],[1,3,2])
        np.testing.assert_array_equal(p['recent_unique'][1],[0,1,2])
    def test_pool_cap_distinct_not_events(self):
        a=np.r_[np.arange(25),np.arange(25)];t=np.arange(50);k=np.ones(50,int)
        p=f.seed_pools(a,t,k,49)
        self.assertEqual(len(p['recent_unique'][0]),20);self.assertEqual(len(p['pending_cart'][0]),20)
    def test_future_event_rejected(self):
        a,t,k=self.prefix()
        with self.assertRaises(ValueError):f.seed_pools(a,t,k,t[-1]-1)
    def test_unsorted_prefix_rejected(self):
        a,t,k=self.prefix();t[2]=0
        with self.assertRaises(ValueError):f.seed_pools(a,t,k,t[-1])
    def test_unsigned_unsorted_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            f.seed_pools(np.array([1,2]),np.array([5,4],dtype=np.uint64),np.array([0,0]),4)
    def test_empty_transform_prefix_rejected(self):
        with self.assertRaises(ValueError):
            f.transform(np.array([],int),np.array([],int),np.array([],int),np.array([1]),self.graphs(),0)
    def test_same_timestamp_later_index_is_valid(self):
        p=f.seed_pools(np.array([1,1]),np.array([5,5]),np.array([1,2]),5)
        self.assertEqual(len(p['pending_cart'][0]),0)
    def test_shape_and_kind_guards(self):
        for kinds in (np.array([9,0]),np.array([0])):
            with self.subTest(kinds=kinds),self.assertRaises(ValueError):
                f.seed_pools(np.array([1,2]),np.array([1,2]),kinds,2)
    def test_hand_calculated_coverage_entropy_and_age(self):
        x=f.summarize_support(np.array([[1.],[3.]]),np.array([1,2]),np.array([3]),np.array([0.,3.]))[0]
        self.assertAlmostEqual(x[0],1)
        self.assertAlmostEqual(x[1],-(.25*np.log(.25)+.75*np.log(.75))/np.log(2),6)
        self.assertAlmostEqual(x[2],.75*np.log(4),6);self.assertAlmostEqual(x[3],np.log(4),6)
    def test_self_edges_excluded_from_values_and_denominator(self):
        x=f.summarize_support(np.array([[100.],[2.]]),np.array([1,2]),np.array([1]),np.array([0.,3.]))[0]
        np.testing.assert_allclose(x,[1,0,np.log(4),np.log(4)],rtol=1e-6)
    def test_only_self_seed_is_zero(self):
        x=f.summarize_support(np.array([[100.]]),np.array([1]),np.array([1]),np.array([0.]))
        np.testing.assert_array_equal(x,[[0,0,0,0]])
    def test_no_support_zero_encoding(self):
        x=f.summarize_support(np.zeros((2,1)),np.array([1,2]),np.array([3]),np.array([0.,1.]))
        np.testing.assert_array_equal(x,[[0,0,0,0]])
    def test_empty_pool(self):
        x=f.summarize_support(np.zeros((0,2)),np.array([],int),np.array([3,4]),np.array([]))
        np.testing.assert_array_equal(x,np.zeros((2,4)))
    def test_strongest_tie_prefers_recent_seed(self):
        x=f.summarize_support(np.ones((2,1)),np.array([1,2]),np.array([3]),np.array([0.,5.]))
        self.assertEqual(x[0,3],0)
    def test_unknown_seeds_count_as_unsupported(self):
        x=f.summarize_support(np.array([[1.],[0.]]),np.array([1,900]),np.array([3]),np.array([0.,3.]))
        self.assertEqual(x[0,0],.5)
    def test_scaling_affinity_does_not_change_relative_summaries(self):
        args=(np.array([1,2]),np.array([3]),np.array([0.,2.]))
        np.testing.assert_allclose(f.summarize_support(np.array([[1.],[3.]]),*args),
                                   f.summarize_support(np.array([[10.],[30.]]),*args))
    def test_bad_graph_values_rejected(self):
        for x in (-1,np.nan,np.inf):
            with self.subTest(x=x),self.assertRaises(ValueError):
                f.summarize_support(np.array([[x]]),np.array([1]),np.array([2]),np.array([0.]))
    def test_negative_ages_rejected(self):
        with self.assertRaises(ValueError):
            f.summarize_support(np.ones((1,1)),np.array([1]),np.array([2]),np.array([-1.]))
    def test_float_negative_duplicate_and_overflow_ids(self):
        for a in (np.array([1.]),np.array([-1]),np.array([2**63],dtype=np.uint64),np.array([1,1])):
            with self.subTest(a=a),self.assertRaises(ValueError):f.ids_array(a,True)
    def test_transform_shape_and_nonmutation(self):
        a,t,k=self.prefix();g=self.graphs();before=g['symmetric'][0].copy()
        x=f.transform(a,t,k,np.array([1,4,99]),g,0)
        self.assertEqual(x.shape,(3,48));self.assertTrue(np.isfinite(x).all())
        self.assertEqual((before!=g['symmetric'][0]).nnz,0)
    def test_transform_exact_replay(self):
        a,t,k=self.prefix();g=self.graphs()
        np.testing.assert_array_equal(f.transform(a,t,k,np.array([1,4]),g,0),
                                      f.transform(a,t,k,np.array([1,4]),g,0))
    def test_wrong_cutoff_and_graph_family(self):
        a,t,k=self.prefix();g=self.graphs()
        with self.assertRaises(ValueError):f.transform(a,t,k,np.array([4]),g,1)
        del g['forward']
        with self.assertRaises(ValueError):f.transform(a,t,k,np.array([4]),g,0)
    def test_catalog_has_availability_and_groups(self):
        c=f.catalog();self.assertEqual(len(c),48)
        self.assertTrue(all('observed prefix' in x['availability'] for x in c))


class PersistenceAndDecisionTests(unittest.TestCase):
    def test_atomic_reuse_conflict_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';e.write_once(p,b'good');e.write_once(p,b'good')
            with self.assertRaises(ValueError):e.write_once(p,b'bad')
            self.assertEqual(p.read_bytes(),b'good')
    def test_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.symlink_to(Path(d)/'missing')
            with self.assertRaises(ValueError):e.write_once(p,b'x')
    def test_npz_digest_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';h=e.save_npz(p,x=np.arange(4))
            self.assertEqual(e.sha(p),h);self.assertEqual(e.save_npz(p,x=np.arange(4)),h)
            with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['x'],np.arange(4))
    def test_nonfinite_json_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):e.save_json(Path(d)/'bad.json',{'x':float('nan')})
            self.assertFalse((Path(d)/'bad.json').exists())
    def test_bad_input_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'a')
            with self.assertRaises(ValueError):e.check_file(p,'0'*64)
    def test_decision_rejects_regression(self):
        self.assertEqual(e.classify(-.01,[-.01,0],0,[-.02,.01]),'STOP_THIS_VARIANT')
    def test_mixed_folds_not_promoted(self):
        self.assertEqual(e.classify(.01,[-.01,.03],.01,[.001,.02]),'MIXED_DIAGNOSE_NO_AUTOMATIC_SCALE')
    def test_order_regression_not_promoted(self):
        self.assertEqual(e.classify(.01,[.01,.01],-.001,[.001,.02]),'MIXED_DIAGNOSE_NO_AUTOMATIC_SCALE')
    def test_positive_uncertain_only_replication(self):
        self.assertEqual(e.classify(.01,[.01,.01],0,[-.01,.03]),'PROMISING_POINT_ESTIMATE_UNCERTAIN_REPLICATE_ONLY')
    def test_small_positive_low_priority(self):
        self.assertEqual(e.classify(.001,[.001,.001],0,[0,.003]),'SMALL_GAIN_LOW_PRIORITY')
    def test_bootstrap_is_paired_stratified_and_deterministic(self):
        d=np.ones((32,3),int);delta=np.zeros_like(d);delta[:,2]=1;fids=np.repeat([0,1],16)
        a=e.paired_interval(delta,d,fids);b=e.paired_interval(delta,d,fids)
        self.assertEqual(a,b);np.testing.assert_allclose(a['descriptive_95_interval'],[.6,.6])
    def test_invalid_bootstrap_support_fails(self):
        d=np.zeros((32,3),int)
        with self.assertRaises(ValueError):e.paired_interval(d,d,np.repeat([0,1],16))
    def test_reference_download_is_bounded_metadata_only(self):
        self.assertEqual(len(e.REFERENCES),6)
        self.assertLess(sum(v[2] for v in e.REFERENCES.values()),60000)
        self.assertTrue(all(v[0].endswith('.json') for v in e.REFERENCES.values()))


if __name__=='__main__':unittest.main()
