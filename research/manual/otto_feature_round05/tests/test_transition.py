from __future__ import annotations
import sqlite3, tempfile, unittest
from pathlib import Path
import numpy as np
from core import save_arrays, checked, sha
from transition_features import NAMES,COLLAPSED_NAMES,anchors,request_keys,pair_keys,split_keys,TransitionIndex,statistics,transform
from transition_index import extract_partition,pack_partition,extraction_sql
from backend_smoke import EVENTS,REQUESTS,EXCLUDED,reference,populate


def fixture_index():
    con=sqlite3.connect(':memory:');populate(con)
    result=extract_partition(con,0,partitions=1,cutoff=200,gap=10);con.close()
    requests=np.array(sorted(REQUESTS),np.int64);keys=pair_keys(requests[:,0],requests[:,1]);ss=np.unique(requests[:,0])
    return TransitionIndex(**pack_partition(keys,ss,result))


class TransitionTests(unittest.TestCase):
    def test_names_exact_and_disjoint(self):
        self.assertEqual((len(NAMES),len(COLLAPSED_NAMES)),(27,9));self.assertEqual(len(set(NAMES+COLLAPSED_NAMES)),36)
    def test_anchor_last_each_type(self):
        np.testing.assert_array_equal(anchors(np.array([1,2,3,4]),np.array([0,1,0,2])),[3,2,4])
    def test_missing_anchor(self):
        np.testing.assert_array_equal(anchors(np.array([5]),np.array([1])),[-1,5,-1])
    def test_invalid_anchor_inputs(self):
        for a,k in [(np.array([1.]),np.array([0])),(np.array([1]),np.array([3])),(np.array([1]),np.array([0,1]))]:
            with self.subTest(a=a,k=k),self.assertRaises(ValueError):anchors(a,k)
    def test_pair_key_roundtrip(self):
        a=np.array([0,1,2**31-1]);b=np.array([9,2**31-1,0]);s,t=split_keys(pair_keys(a,b));np.testing.assert_array_equal(s,a);np.testing.assert_array_equal(t,b)
    def test_reject_pair_overflow(self):
        for a in [np.array([-1]),np.array([2**31]),np.array([2**63],np.uint64),np.array([1.])]:
            with self.subTest(a=a),self.assertRaises(ValueError):pair_keys(a,np.array([1]))
    def test_pair_shape_reject(self):
        with self.assertRaises(ValueError):pair_keys(np.array([1]),np.array([1,2]))
    def test_request_keys_include_reverse_not_self(self):
        k=request_keys(np.array([[1,2,-1]]),np.array([[1,2,3]]));s,t=split_keys(k)
        self.assertEqual(set(zip(s,t)),{(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)})
    def test_requests_deduplicated(self):
        a=np.array([[1,1,1],[1,1,1]]);c=np.array([[2,3],[2,3]])
        self.assertEqual(len(request_keys(a,c)),4)
    def test_request_invalid_missing(self):
        with self.assertRaises(ValueError):request_keys(np.array([[-2,-1,-1]]),np.array([[2]]))
    def test_reference_and_sql_agree_all_partitions(self):
        con=sqlite3.connect(':memory:');populate(con)
        try:
            for n in [1,2,4]:
                for b in range(n):
                    self.assertEqual(extract_partition(con,b,n,200,10),reference(EVENTS,REQUESTS,EXCLUDED,200,10,b,n))
        finally:con.close()
    def test_equal_timestamp_and_exact_gap_are_eligible(self):
        index=fixture_index();tt,_=index.lookup(np.array([1]),np.array([2]));self.assertEqual(tt[0,1],2)
    def test_multiplicity_once_per_session_pair_type(self):
        index=fixture_index();tt,_=index.lookup(np.array([1]),np.array([2]));self.assertEqual(tt[0,1],2);self.assertEqual(tt[0,5],1)
    def test_collapsed_counts_not_sum_of_types(self):
        index=fixture_index();t,p=index.lookup(np.array([1]),np.array([2]));self.assertEqual(p[0],2);self.assertEqual(t[0].sum(),3)
    def test_no_skipping_intermediate_unrequested_item(self):
        index=fixture_index();t,p=index.lookup(np.array([1]),np.array([3]));self.assertEqual(p[0],0)
    def test_denominators_include_nonrequested_destination(self):
        index=fixture_index();d,p=index.denominators(1);self.assertEqual(d[0],3);self.assertEqual(p,3)
    def test_no_cross_session_adjacent_edges(self):
        con=sqlite3.connect(':memory:');ev=[(1,1,10,0,0),(2,2,11,1,1)]
        populate(con,ev,REQUESTS,[]);result=extract_partition(con,0,1,200,10);con.close();self.assertEqual(result['typed'],[])
    def test_exclusion_removes_all_session_actions(self):
        con=sqlite3.connect(':memory:');populate(con)
        result=extract_partition(con,0,1,1000,1000);con.close()
        self.assertEqual(result,reference(EVENTS,REQUESTS,EXCLUDED,1000,1000,0,1))
    def test_cutoff_is_exclusive(self):
        ev=[(1,1,199,0,0),(1,2,200,1,1)]
        con=sqlite3.connect(':memory:');populate(con,ev,REQUESTS,[]);r=extract_partition(con,0,1,200,10);con.close();self.assertEqual(r['typed'],[])
    def test_event_gap_and_self_pairs_excluded(self):
        ev=[(1,1,0,0,0),(1,1,1,1,0),(1,2,100,2,1)]
        con=sqlite3.connect(':memory:');populate(con,ev,REQUESTS,[]);r=extract_partition(con,0,1,200,10);con.close();self.assertEqual(r['typed'],[])
    def test_physical_order_does_not_define_adjacency(self):
        con=sqlite3.connect(':memory:');populate(con,EVENTS[::-1]);r=extract_partition(con,0,1,200,10);con.close();self.assertEqual(r,reference(EVENTS,REQUESTS,EXCLUDED,200,10,0,1))
    def test_missing_event_index_not_bridged(self):
        ev=[(1,1,0,0,0),(1,2,1,2,1)]
        con=sqlite3.connect(':memory:');populate(con,ev,REQUESTS,[]);r=extract_partition(con,0,1,200,10);con.close();self.assertEqual(r['typed'],[])
    def test_feature_values_independent(self):
        index=fixture_index();x,c=transform(index,np.array([1,-1,-1]),np.array([1,2,3,999]))
        self.assertEqual(x.shape,(4,27));self.assertEqual(c.shape,(4,9))
        np.testing.assert_allclose(x[1,3:6],[np.log1p(2),2/23,0],rtol=1e-6)
        self.assertTrue(np.all(x[0]==0));self.assertTrue(np.all(x[3]==0))
    def test_unknown_source_is_zero(self):
        x,y=transform(fixture_index(),np.array([999,-1,-1]),np.array([2,3]));self.assertFalse(x.any());self.assertFalse(y.any())
    def test_reverse_swaps_action_types(self):
        idx=fixture_index();f,_=idx.lookup(np.array([1]),np.array([2]));r,_=idx.lookup(np.array([2]),np.array([1]));self.assertEqual(f[0,1],2);self.assertEqual(r[0,3],2)
    def test_no_source_not_a_fake_zero_kind(self):
        x,y=transform(fixture_index(),np.array([-1,-1,-1]),np.array([2,3]));self.assertFalse(x.any());self.assertFalse(y.any())
    def test_negative_directionality_allowed(self):
        z=statistics(np.array([0]),np.array([4]),0);self.assertLess(z[0,2],0);self.assertTrue(np.all(np.isfinite(z)))
    def test_invalid_metric_counts_rejected(self):
        for args in [(np.array([-1]),np.array([0]),1),(np.array([2]),np.array([0]),1),(np.array([np.nan]),np.array([0]),1)]:
            with self.subTest(args=args),self.assertRaises(ValueError):statistics(*args)
    def test_deterministic_nonmutating_transform(self):
        index=fixture_index();keys=index.keys.copy();x=index.typed.copy();aa=np.array([1,2,-1]);cc=np.array([1,2,3])
        first=transform(index,aa,cc);second=transform(index,aa,cc)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(index.keys,keys);np.testing.assert_array_equal(index.typed,x)
    def test_candidate_order_equivariance(self):
        index=fixture_index();a,b=transform(index,np.array([1,2,-1]),np.array([1,2,3]));c,d=transform(index,np.array([1,2,-1]),np.array([3,1,2]))
        np.testing.assert_array_equal(c,a[[2,0,1]]);np.testing.assert_array_equal(d,b[[2,0,1]])
    def test_corrupt_index_support_rejected(self):
        x=fixture_index();x.typed[0,0]=10000
        with self.assertRaises(ValueError):TransitionIndex(x.keys,x.typed,x.collapsed,x.source_ids,x.outgoing,x.pooled_outgoing)
    def test_duplicate_keys_rejected(self):
        x=fixture_index();x.keys[1]=x.keys[0]
        with self.assertRaises(ValueError):TransitionIndex(x.keys,x.typed,x.collapsed,x.source_ids,x.outgoing,x.pooled_outgoing)
    def test_sql_bounds_reject_bool_and_negative(self):
        for b in [-1,16,True]:
            with self.subTest(bucket=b),self.assertRaises(ValueError):extraction_sql(b)
    def test_model_feature_data_checkpoint_roundtrip(self):
        x,y=transform(fixture_index(),np.array([1,2,-1]),np.array([1,2,3]))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'features.npz';h=save_arrays(p,typed=x,collapsed=y);save_arrays(p,typed=x,collapsed=y);checked(p,h)
            with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['typed'],x)
            with self.assertRaises(ValueError):save_arrays(p,typed=x+1,collapsed=y)
            checked(p,h)

if __name__=='__main__':unittest.main()
