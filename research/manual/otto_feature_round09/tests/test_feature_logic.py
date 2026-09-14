import unittest
import numpy as np
from feature_logic import build,names,select,rarity_votes,continuation_votes
from neighbors import group_history

CUT=100000000
class Formulas(unittest.TestCase):
 def setUp(self):
  self.rows=np.array([(1,20,100,0,0),(1,10,200,0,1),(1,30,300,2,2),(1,30,400,2,3),(1,40,500,1,8),
    (2,10,100,0,0),(2,20,200,0,1),(2,50,300,1,2),(3,10,100,0,0),(3,60,200,2,1)],np.int64)
  self.h=group_history(self.rows,CUT,set());self.aa=np.array([10,20]);self.cc=np.array([10,20,30,40,50,60,70]);self.freq={10:100,20:4}
 def go(self,r,aa=None,cc=None,source=None,h=None,freq=None):
  return build(r,self.aa if aa is None else aa,self.cc if cc is None else cc,
      [1,2,3] if source is None else source,self.h if h is None else h,self.freq if freq is None else freq)
 def test_exact_width_names_unique(self):
  for r in (8,9):
   self.assertEqual(len(names(r)),24);self.assertEqual(len(set(names(r)+names(r,True))),48)
   a,b,_=self.go(r);self.assertEqual(a.shape,(7,24));self.assertEqual(b.shape,a.shape)
 def test_no_target_arguments(self):
  import inspect
  self.assertNotIn('labels',inspect.signature(build).parameters);self.assertNotIn('targets',inspect.signature(build).parameters)
 def test_finite_nonnegative(self):
  for r in (8,9):
   for x in self.go(r)[:2]:self.assertTrue(np.isfinite(x).all() and (x>=0).all())
 def test_unknown_candidate_zero(self):
  for r in (8,9):
   a,b,_=self.go(r);self.assertFalse(a[-1].any());self.assertFalse(b[-1].any())
 def test_candidate_permutation(self):
  for r in (8,9):
   a,b,_=self.go(r);x,y,_=self.go(r,cc=self.cc[::-1]);np.testing.assert_array_equal(a,x[::-1]);np.testing.assert_array_equal(b,y[::-1])
 def test_source_permutation(self):
  for r in (8,9):
   a,b,_=self.go(r);x,y,_=self.go(r,source=[3,2,1]);np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
 def test_no_historical_support_zero(self):
  for r in (8,9):
   a,b,d=self.go(r,source=[]);self.assertFalse(a.any());self.assertFalse(b.any());self.assertEqual(d['neighbor_counts'],[0,0])
 def test_duplicate_candidates_fail(self):
  with self.assertRaises(ValueError):self.go(8,cc=np.array([1,1]))
 def test_negative_candidates_fail(self):
  with self.assertRaises(ValueError):self.go(9,cc=np.array([-1]))
 def test_float_candidates_fail(self):
  with self.assertRaises(ValueError):self.go(8,cc=np.array([1.]))
 def test_duplicate_anchors_fail(self):
  with self.assertRaises(ValueError):self.go(8,aa=np.array([10,10]))
 def test_empty_anchors_fail(self):
  with self.assertRaises(ValueError):self.go(8,aa=np.array([],np.int64))
 def test_fifth_anchor_fails(self):
  with self.assertRaises(ValueError):self.go(8,aa=np.arange(5))
 def test_missing_source_fails(self):
  with self.assertRaises(ValueError):self.go(8,source=[100])
 def test_duplicate_source_fails(self):
  with self.assertRaises(ValueError):self.go(8,source=[1,1])
 def test_negative_frequency_fails(self):
  with self.assertRaises(ValueError):self.go(8,freq={10:-1,20:4})
 def test_false_zero_frequency_fails(self):
  with self.assertRaises(ValueError):self.go(9,freq={10:0,20:4})
 def test_unknown_round_fails(self):
  with self.assertRaises(ValueError):self.go(10)
 def test_rare_anchor_changes_similarity(self):
  a,b,_=self.go(8);self.assertGreater(float(np.abs(a-b).sum()),0)
 def test_equal_anchor_frequency_collapses_ablation(self):
  a,b,_=self.go(8,freq={10:4,20:4});np.testing.assert_array_equal(a,b)
 def test_one_anchor_rarity_is_scale_invariant(self):
  a,b,_=self.go(8,aa=np.array([10]));np.testing.assert_allclose(a,b,rtol=1e-6)
 def test_support_is_distinct_sessions_not_events(self):
  a,b,_=self.go(8);self.assertAlmostEqual(float(a[2,16]),np.log(2),places=6)
 def test_effective_support_single_session(self):
  a,b,_=self.go(8);self.assertAlmostEqual(float(a[2,22]),np.log(2),places=6)
 def test_neighbor_rank_mass_normalized(self):
  a,b,_=self.go(8);self.assertTrue((a[:,[3,11,19]]<=1).all())
 def test_future_source_rejected(self):
  rows=self.rows.copy();rows[0,2]=CUT
  with self.assertRaises(ValueError):group_history(rows,CUT,set())
 def test_study_session_rejected(self):
  with self.assertRaises(ValueError):group_history(self.rows,CUT,{1})
 def test_duplicate_original_index_rejected(self):
  rows=self.rows.copy();rows[1,4]=0
  with self.assertRaises(ValueError):group_history(rows,CUT,set())
 def test_forward_is_subset_of_unsigned(self):
  a,b,_=self.go(9);self.assertTrue((a<=b+1e-7).all())
 def test_forward_keeps_later_order(self):
  a,b,_=self.go(9);self.assertGreater(a[2,16],0)
 def test_original_gaps_not_compressed(self):
  a,b,_=self.go(9);self.assertEqual(a[3,8],0);self.assertEqual(b[3,8],0)
 def test_self_anchor_excluded_both_arms(self):
  a,b,_=self.go(9);self.assertFalse(a[0].any());self.assertFalse(b[0].any())
 def test_before_only_item_removed_by_direction(self):
  h=group_history(np.array([(1,80,100,2,0),(1,10,200,0,1)],np.int64),CUT,set())
  a,b,_=self.go(9,aa=np.array([10]),cc=np.array([80]),source=[1],h=h,freq={10:1})
  self.assertFalse(a.any());self.assertAlmostEqual(float(b[0,16]),np.log(2),places=6)
 def test_tied_time_forward_original_index_allowed(self):
  h=group_history(np.array([(1,10,100,0,0),(1,80,100,2,1)],np.int64),CUT,set())
  a,b,_=self.go(9,aa=np.array([10]),cc=np.array([80]),source=[1],h=h,freq={10:1})
  self.assertEqual(float(a[0,20]),1.);self.assertEqual(float(a[0,21]),1.)
 def test_time_window_inclusive_boundary(self):
  h=group_history(np.array([(1,10,100,0,0),(1,80,1800100,2,5),(1,90,1800101,2,6)],np.int64),CUT,set())
  a,b,_=self.go(9,aa=np.array([10]),cc=np.array([80,90]),source=[1],h=h,freq={10:1})
  self.assertGreater(a[0,16],0);self.assertFalse(a[1].any());self.assertAlmostEqual(float(a[0,19]),.2,places=6)
 def test_latest_pivot_not_first_occurrence(self):
  h=group_history(np.array([(1,10,100,0,0),(1,80,200,2,1),(1,10,300,0,2)],np.int64),CUT,set())
  a,b,_=self.go(9,aa=np.array([10]),cc=np.array([80]),source=[1],h=h,freq={10:1})
  self.assertFalse(a.any());self.assertGreater(b[0,16],0)
 def test_continuation_same_neighbor_set(self):
  _,_,d=self.go(9);self.assertEqual(d['neighbor_counts'][0],d['neighbor_identity_overlap'])
 def test_neighbor_cap64_and_deterministic_tie(self):
  h=group_history(np.array([(sid,item,ts,0,ix) for sid in range(100) for ix,(item,ts) in enumerate([(10,100),(80,200)])],np.int64),CUT,set())
  nn=select(np.array([10]),list(reversed(range(100))),h,{10:100},False,True)
  self.assertEqual([x[0].session for x in nn],list(range(64)))
 def test_event_repetition_not_inflate_forward_support(self):
  h=group_history(np.array([(1,10,100,0,0),(1,80,200,2,1),(1,80,300,2,2)],np.int64),CUT,set())
  a,b,_=self.go(9,aa=np.array([10]),cc=np.array([80]),source=[1],h=h,freq={10:1})
  self.assertAlmostEqual(float(a[0,16]),np.log(2),places=6);self.assertEqual(a[0,17],1.)
 def test_action_specific_not_pooled(self):
  a,b,_=self.go(9);self.assertFalse(a[2,:16].any());self.assertTrue(a[2,16:].any())
 def test_absent_last_anchor_returns_zero(self):
  a,b,_=self.go(9,aa=np.array([99,20]),freq={99:0,20:4});self.assertFalse(a.any());self.assertFalse(b.any())
