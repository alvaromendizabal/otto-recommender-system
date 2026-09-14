"""Independent same-product action-order oracles, including adversarial cases."""
import copy,math,unittest
from unittest import mock
import numpy as np
from funnel_features import (PAIRS,MAX_ELAPSED_MS,DECAY_MS,feature_names,
    nearest_lag,summarize_session,SummaryCache,aggregate,build_funnel)
from neighbors import group_history
from feature_logic import build,names,select
from shared_reuse import validate_protocol
import experiment

CUT=100000000

def oracle_lag(source,dest,ordered):
    # Exhaustive oracle used only for tiny fixtures, intentionally unlike production.
    gaps=[abs(td-ts) for isrc,ts in source for idst,td in dest
          if (not ordered or isrc<idst) and abs(td-ts)<=MAX_ELAPSED_MS]
    return min(gaps) if gaps else None

class FunnelFeatures(unittest.TestCase):
    def setUp(self):
        self.rows=np.array([(1,10,10,0,0),(1,80,20,0,1),(1,80,30,1,2),(1,80,40,2,3),
         (2,10,10,0,0),(2,80,20,2,1),(2,80,30,1,2),(2,80,40,0,3),
         (3,10,10,0,0),(3,80,20,0,1),(3,99,30,2,2)],np.int64)
        self.h=group_history(self.rows,CUT,set());self.aa=np.array([10]);self.cc=np.array([80,99,999]);self.freq={10:3}
    def go(self,history=None,source=None,candidates=None,cache=None):
        return build(10,self.aa,self.cc if candidates is None else candidates,
            [1,2,3] if source is None else source,self.h if history is None else history,self.freq,cache=cache)
    def test_48_unique_names(self):
        self.assertEqual(len(feature_names()),24);self.assertEqual(len(set(feature_names()+feature_names(True))),48)
        self.assertEqual(names(10),feature_names())
    def test_width_dtype_finite(self):
        for a in self.go()[:2]:
            self.assertEqual(a.shape,(3,24));self.assertEqual(a.dtype,np.float32);self.assertTrue(np.isfinite(a).all() and (a>=0).all())
    def test_ordered_vs_unordered_count_oracle(self):
        a,b,_=self.go()
        for k in range(3):
            self.assertAlmostEqual(a[0,k*8],math.log(2),places=6)
            self.assertAlmostEqual(b[0,k*8],math.log(3),places=6)
    def test_conditional_source_denominator_differs_from_global_mass(self):
        a,b,_=self.go();self.assertAlmostEqual(a[0,2],1/23,places=6)
        self.assertAlmostEqual(a[0,10],1/22,places=6)
        # Cart-source sessions 1 and 2 have equal weights: one progresses.
        self.assertAlmostEqual(a[0,11],.5,places=6)
    def test_no_cross_product_funnel(self):
        a,b,_=self.go();self.assertFalse(a[1].any());self.assertFalse(b[1].any())
    def test_missing_candidate_zero(self):
        a,b,_=self.go();self.assertFalse(a[2].any());self.assertFalse(b[2].any())
    def test_same_neighbor_set(self):
        _,_,d=self.go();self.assertEqual(d['neighbor_counts'],[3,3]);self.assertEqual(d['neighbor_identity_overlap'],3)
    def test_candidate_permutation(self):
        a,b,_=self.go();aa,bb,_=self.go(candidates=self.cc[::-1]);np.testing.assert_array_equal(a,aa[::-1]);np.testing.assert_array_equal(b,bb[::-1])
    def test_source_permutation(self):
        a,b,_=self.go();aa,bb,_=self.go(source=[3,1,2]);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
    def test_no_neighbors_zero(self):
        a,b,d=self.go(source=[]);self.assertFalse(a.any());self.assertFalse(b.any());self.assertEqual(d['neighbor_counts'],[0,0])
    def test_cache_equals_uncached(self):
        cache=SummaryCache(self.h,2)
        for _ in range(2):
            a,b,_=self.go();aa,bb,_=self.go(cache=cache);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        self.assertLessEqual(len(cache._cache),2)
    def test_cache_rejects_other_history_same_ids(self):
        with self.assertRaisesRegex(ValueError,'identity'):self.go(cache=SummaryCache(dict(self.h)))
    def test_invalid_cache_bounds(self):
        for n in (0,-1,1.2):
            with self.assertRaises(ValueError):SummaryCache(self.h,n)
    def test_no_targets_in_signature(self):
        import inspect
        for func in (build_funnel,summarize_session,aggregate):
            self.assertFalse({'labels','target','targets'}&set(inspect.signature(func).parameters))
    def test_pair_count_once_despite_repeated_orders(self):
        rows=np.concatenate([self.rows,np.array([(1,80,50,2,4),(1,80,60,2,5)],np.int64)])
        h=group_history(rows[np.lexsort((rows[:,4],rows[:,0]))],CUT,set())
        a,b,_=self.go(history=h);aa,bb,_=self.go();np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
    def test_time_tie_uses_original_index(self):
        self.assertEqual(nearest_lag([(1,100)],[(2,100)],True),0)
        self.assertIsNone(nearest_lag([(2,100)],[(1,100)],True))
        self.assertEqual(nearest_lag([(2,100)],[(1,100)],False),0)
    def test_original_position_gaps_allowed_not_assumed_adjacent(self):
        self.assertEqual(nearest_lag([(4,100)],[(90,200)],True),100)
    def test_window_boundary_inclusive(self):
        self.assertEqual(nearest_lag([(0,100)],[(1,100+MAX_ELAPSED_MS)],True),MAX_ELAPSED_MS)
        self.assertIsNone(nearest_lag([(0,100)],[(1,101+MAX_ELAPSED_MS)],True))
    def test_nonmonotonic_time_rejected(self):
        with self.assertRaises(ValueError):nearest_lag([(0,100)],[(1,90)],True)
    def test_multiple_source_visits_uses_nearest_preceding(self):
        self.assertEqual(nearest_lag([(0,100),(2,180)],[(3,200)],True),20)
    def test_later_source_not_allowed_in_primary(self):
        self.assertEqual(nearest_lag([(0,100),(3,199)],[(2,180)],True),80)
        self.assertEqual(nearest_lag([(0,100),(3,199)],[(2,180)],False),19)
    def test_randomized_minimum_gap_oracle(self):
        rng=np.random.default_rng(918)
        for _ in range(150):
            times=np.cumsum(rng.integers(0,500000,20));kind=rng.integers(0,3,20)
            src=[(i,int(t)) for i,t in enumerate(times) if kind[i]==0];dst=[(i,int(t)) for i,t in enumerate(times) if kind[i]==1]
            for order in (True,False):self.assertEqual(nearest_lag(src,dst,order),oracle_lag(src,dst,order))
    def test_ordered_support_subset_not_effective_n(self):
        a,b,_=self.go();cols=[j for j in range(24) if j%8!=7]
        self.assertTrue((a[:,cols]<=b[:,cols]+1e-7).all())
    def test_future_and_excluded_history_rejected(self):
        with self.assertRaises(ValueError):group_history(self.rows,CUT,{1})
        with self.assertRaises(ValueError):group_history(self.rows,35,set())
    def test_cache_readonly_features(self):
        original=self.rows.copy();self.go(cache=SummaryCache(self.h));np.testing.assert_array_equal(original,self.rows)
    def test_three_distinct_action_pairs(self):self.assertEqual(PAIRS,((0,1),(1,2),(0,2)))
    def test_finite_shrunk_support_upper_bound(self):
        for a in self.go()[:2]:self.assertTrue((a[:,[2,10,18]]<1).all())
    def test_absent_source_is_not_population_negative(self):
        h=group_history(np.array([(1,10,10,0,0),(1,80,20,2,1)],np.int64),CUT,set())
        a,b,_=self.go(history=h,source=[1]);self.assertFalse(a.any());self.assertFalse(b.any())
    def test_protocol_preserves_exact_ranker(self):
        p=experiment.protocol();old=copy.deepcopy(p);old['round']=8;validate_protocol(p,old)
        p['model_params']['learning_rate']=.3
        with self.assertRaisesRegex(ValueError,'comparison changed'):validate_protocol(p,old)
    def test_protocol_forbids_window_tuning(self):
        p=experiment.protocol();old=copy.deepcopy(p);p['funnel_max_elapsed_ms']+=1
        with self.assertRaisesRegex(ValueError,'preregistered'):validate_protocol(p,old)
    def test_protocol_forbids_new_history_scan(self):
        p=experiment.protocol();old=copy.deepcopy(p);p['maximum_shared_additional_parquet_scans']=1
        with self.assertRaises(ValueError):validate_protocol(p,old)
