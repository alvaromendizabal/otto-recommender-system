import inspect
import unittest
import numpy as np
from relative_features import (SIGNALS, NAMES, GLOBAL_NAMES, SEEN_SIGNAL,
                               context_transforms, transform, coverage_statistics)

class RelativeFeatures(unittest.TestCase):
    def setUp(self):
        self.a=np.arange(400,dtype=np.int64)
        self.x=np.zeros((400,6),np.float64)
        self.seen=self.a<20
    def run_transform(self, x=None, seen=None, aids=None):
        return context_transforms(self.x if x is None else x,self.seen if seen is None else seen,self.a if aids is None else aids)
    def test_schema(self):
        self.assertEqual((len(SIGNALS),len(NAMES),len(GLOBAL_NAMES)),(6,12,12))
        self.assertEqual(len(set(NAMES+GLOBAL_NAMES)),24)
    def test_not_another_demand_feature(self):
        self.assertTrue(all('graph' in s or 'domain_norm' in s for s in SIGNALS))
        self.assertTrue(all(not s.startswith('hist_') for s in SIGNALS))
    def test_zero_evidence_is_zero_not_rank_middle(self):
        x,y=self.run_transform();np.testing.assert_array_equal(x,0);np.testing.assert_array_equal(y,0)
    def test_positive_ties_receive_identical_midrank(self):
        self.x[:2,0]=1;r,g=self.run_transform()
        np.testing.assert_allclose(r[:2,:2],[[19.5/20,.5],[19.5/20,.5]])
        np.testing.assert_allclose(g[:2,0],399.5/400)
    def test_hand_computation_with_two_groups(self):
        self.x[[0,1,20,21],0]=[1,3,2,6];r,g=self.run_transform()
        np.testing.assert_allclose(r[[0,1,20,21],:2],[[19/20,.25],[1,.75],[379/380,.25],[1,.75]])
        np.testing.assert_allclose(g[[0,1,20,21],1],np.array([1,3,2,6])/12)
    def test_all_equal_positive(self):
        r,g=self.run_transform(np.ones_like(self.x))
        np.testing.assert_allclose(r[:20,::2],10.5/20);np.testing.assert_allclose(r[20:,::2],190.5/380)
        np.testing.assert_allclose(g[:,::2],200.5/400)
    def test_exact_permutation_invariance(self):
        rng=np.random.default_rng(32);x=rng.lognormal(2,4,size=(400,6));ix=rng.permutation(400)
        r,g=self.run_transform(x);rr,gg=self.run_transform(x[ix],self.seen[ix],self.a[ix])
        np.testing.assert_array_equal(rr,r[ix]);np.testing.assert_array_equal(gg,g[ix])
    def test_scale_invariance(self):
        x=np.arange(2400).reshape(400,6).astype(float)
        r,g=self.run_transform(x);rr,gg=self.run_transform(x*8)
        np.testing.assert_array_equal(rr,r);np.testing.assert_array_equal(gg,g)
    def test_identifiers_not_predictive(self):
        x=np.arange(2400).reshape(400,6);r,g=self.run_transform(x)
        rr,gg=self.run_transform(x,aids=self.a+50000)
        np.testing.assert_array_equal(r,rr);np.testing.assert_array_equal(g,gg)
    def test_opposite_stratum_does_not_change_primary_group(self):
        x=np.ones_like(self.x);r,g=self.run_transform(x);x[-1]=100;rr,gg=self.run_transform(x)
        np.testing.assert_array_equal(r[:20],rr[:20]);self.assertFalse(np.array_equal(g[0],gg[0]))
    def test_group_change_affects_primary_not_global(self):
        x=np.arange(2400).reshape(400,6);r,g=self.run_transform(x);s=self.a<60;rr,gg=self.run_transform(x,s)
        self.assertFalse(np.array_equal(r,rr));np.testing.assert_array_equal(g,gg)
    def test_all_seen_matches_ablation(self):
        r,g=self.run_transform(np.ones_like(self.x),np.ones(400,bool));np.testing.assert_array_equal(r,g)
    def test_all_unseen_matches_ablation(self):
        r,g=self.run_transform(np.ones_like(self.x),np.zeros(400,bool));np.testing.assert_array_equal(r,g)
    def test_singleton_positive_group(self):
        r,_=self.run_transform(np.ones_like(self.x),self.a==0);np.testing.assert_array_equal(r[0],1)
    def test_complete_pool_required(self):
        with self.assertRaises(ValueError):context_transforms(self.x[:60],self.seen[:60],self.a[:60])
    def test_duplicate_candidates_rejected(self):
        a=self.a.copy();a[-1]=a[0]
        with self.assertRaises(ValueError):self.run_transform(aids=a)
    def test_invalid_inputs_rejected(self):
        for bad in [np.nan,np.inf,-1]:
            with self.subTest(bad=bad):
                x=self.x.copy();x[0,0]=bad
                with self.assertRaises(ValueError):self.run_transform(x)
    def test_seen_must_be_boolean(self):
        with self.assertRaises(ValueError):self.run_transform(seen=self.seen.astype(int))
    def test_float_ids_rejected(self):
        with self.assertRaises(ValueError):self.run_transform(aids=self.a.astype(float))
    def test_no_labels_argument(self):
        for fn in [transform,context_transforms]:
            self.assertFalse(set(inspect.signature(fn).parameters)&{'labels','truth','targets','y'})
    def test_nonmutation(self):
        x=self.x.copy();a=self.a.copy();s=self.seen.copy();self.run_transform(x,s,a)
        np.testing.assert_array_equal(x,self.x);np.testing.assert_array_equal(a,self.a);np.testing.assert_array_equal(s,self.seen)
    def test_extreme_finite_values(self):
        r,g=self.run_transform(np.ones_like(self.x)*1e300);self.assertTrue(np.isfinite(r).all() and np.isfinite(g).all())
    def test_source_schema_mapping_and_missing_columns(self):
        names=list(reversed(SIGNALS))+[SEEN_SIGNAL]
        base=np.column_stack([np.arange(2400).reshape(400,6),self.seen*.1])
        r,g=transform(base,names,self.a);rr,gg=self.run_transform(base[:,:6][:,::-1])
        np.testing.assert_array_equal(r,rr);np.testing.assert_array_equal(g,gg)
        with self.assertRaises(ValueError):transform(base,['bad']*7,self.a)
    def test_prefix_share_rejected_out_of_range(self):
        base=np.zeros((400,7));base[:,6]=1.1
        with self.assertRaises(ValueError):transform(base,list(SIGNALS)+[SEEN_SIGNAL],self.a)
    def test_group_shares_each_sum_one(self):
        r,g=self.run_transform(np.arange(2400).reshape(400,6))
        np.testing.assert_allclose(r[self.seen,1::2].sum(axis=0),1,atol=2e-6)
        np.testing.assert_allclose(r[~self.seen,1::2].sum(axis=0),1,atol=2e-6)
        np.testing.assert_allclose(g[:,1::2].sum(axis=0),1,atol=2e-6)

class Coverage(unittest.TestCase):
    def test_metric_caps_and_gap_partition(self):
        y=np.zeros((2,400,3),np.int8);y[:,0,0]=1;y[:,0,1]=1;y[:,:25,2]=1
        d=np.array([[1,2,20],[1,2,20]],np.int64);h=np.array([[1,1,7],[1,0,8]],np.int64)
        r=coverage_statistics(y,d,h)
        self.assertEqual(r['candidate_oracle_at20']['hits'],[2,2,40])
        self.assertAlmostEqual(r['weighted_missing_candidate_gap']+r['weighted_within_candidate_ranking_gap'],1-r['achieved_control']['weighted_recall_at_20'])
    def test_achieved_hits_cannot_exceed_available_targets(self):
        with self.assertRaises(ValueError):coverage_statistics(np.zeros((2,400,3),int),np.ones((2,3),int),np.ones((2,3),int))
    def test_no_retrieval_miss_denominator_drop(self):
        y=np.zeros((2,400,3),int);y[:,0,:]=1;d=np.array([[1,3,3],[1,3,3]])
        r=coverage_statistics(y,d,np.ones((2,3),int));self.assertEqual(r['candidate_oracle_at20']['denominators'],[2,6,6])
