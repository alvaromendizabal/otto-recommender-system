import math
import unittest
import numpy as np
from neighbors import (query_anchors,HistoricalSession,group_history,select_neighbors,votes,project,
                       propose,build_query,NAMES,ABLATION_NAMES)


def h(s,events):
    x=np.asarray(events,np.int64)
    return HistoricalSession(s,x[:,0],x[:,1],x[:,2],x[:,3])

class AnchorTests(unittest.TestCase):
    def test_unique_recency(self):self.assertEqual(query_anchors(np.array([1,2,3,2,4])).tolist(),[4,2,3,1])
    def test_four_bound(self):self.assertEqual(query_anchors(np.arange(9)).tolist(),[8,7,6,5])
    def test_single(self):self.assertEqual(query_anchors(np.array([2,2])).tolist(),[2])
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):query_anchors(np.array([],np.int64))
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):query_anchors(np.array([-1]))
    def test_float_rejected(self):
        with self.assertRaises(ValueError):query_anchors(np.array([1.2]))
    def test_bad_bound(self):
        with self.assertRaises(ValueError):query_anchors(np.array([1]),5)

class HistoricalTests(unittest.TestCase):
    def test_cutoff_equality_rejected(self):
        with self.assertRaises(ValueError):h(1,[(2,0,0,10)]).validate(10)
    def test_duplicate_index_rejected(self):
        with self.assertRaises(ValueError):h(1,[(2,0,0,1),(3,1,0,2)]).validate(10)
    def test_gapped_index_valid(self):h(1,[(2,0,0,1),(3,1,8,2)]).validate(10)
    def test_invalid_action(self):
        with self.assertRaises(ValueError):h(1,[(2,3,0,1)]).validate(10)
    def test_time_nonmonotonic(self):
        with self.assertRaises(ValueError):h(1,[(2,0,0,2),(3,1,1,1)]).validate(10)
    def test_exclusion(self):
        with self.assertRaises(ValueError):group_history(np.array([[1,2,1,0,0]]),10,{1})
    def test_empty(self):self.assertEqual(group_history(np.empty((0,5),np.int64),10,set()),{})
    def test_group_shuffled(self):
        a=group_history(np.array([[2,3,2,0,1],[1,2,1,0,0],[2,2,1,1,0]]),10,set())
        self.assertEqual(a[2].aids.tolist(),[2,3])

class NeighborTests(unittest.TestCase):
    def setUp(self):
        self.hist={1:h(1,[(10,0,0,1),(20,0,2,2),(30,2,9,3),(30,2,10,4)]),
                   2:h(2,[(10,0,0,1),(40,1,1,2)]),3:h(3,[(50,0,0,1)])}
    def test_no_overlap(self):self.assertEqual(select_neighbors(np.array([99]),[1,2,3],self.hist),[])
    def test_duplicate_source_rejected(self):
        with self.assertRaises(ValueError):select_neighbors(np.array([10]),[1,1],self.hist)
    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):select_neighbors(np.array([10]),[99],self.hist)
    def test_feature_schema(self):self.assertEqual(len(NAMES),12);self.assertEqual(len(ABLATION_NAMES),12)
    def test_distinct_session_support_not_events(self):
        nn=select_neighbors(np.array([20,10]),[1],self.hist);v=votes(nn)
        self.assertAlmostEqual(v[(30,2)][0],math.log(2));self.assertEqual(v[(30,2)][1],1)
    def test_after_latest_matched_anchor(self):
        nn=select_neighbors(np.array([20,10]),[1],self.hist);v=votes(nn)
        self.assertEqual(v[(30,2)][3],1);self.assertEqual(v[(10,0)][3],0);self.assertEqual(v[(20,0)][3],0)
    def test_shared_neighbor_denominator(self):
        nn=select_neighbors(np.array([10]),[1,2],self.hist);v=votes(nn)
        expected=(1/math.sqrt(3))/(1/math.sqrt(3)+1/math.sqrt(2))
        self.assertAlmostEqual(v[(30,2)][1],expected)
    def test_single_anchor_arms_identical(self):
        r=build_query(np.array([10]),np.array([10,30,40]),[1,2,3],self.hist)
        np.testing.assert_array_equal(r['session_context']['features'],r['last_anchor']['features'])
    def test_candidate_order_invariance(self):
        a=build_query(np.array([20,10]),np.array([10,30,40]),[1,2],self.hist)
        b=build_query(np.array([20,10]),np.array([40,30,10]),[2,1],self.hist)
        for arm in a:np.testing.assert_array_equal(a[arm]['features'],b[arm]['features'][::-1])
    def test_unknown_candidate_zero(self):
        a=build_query(np.array([20]),np.array([99]),[1,2],self.hist)
        self.assertEqual(a['session_context']['features'].sum(),0)
    def test_repeat_candidate_allowed(self):
        a=build_query(np.array([10]),np.array([10]),[1,2],self.hist)
        self.assertGreater(a['session_context']['features'][0,0],0)
    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):project({},np.array([1,1]))
    def test_empty_votes_zero(self):self.assertEqual(project(votes([]),np.arange(400)).sum(),0)
    def test_tied_proposals_aid_order(self):
        p,_=propose({(2,0):np.array([1,.5,.5,0]),(1,0):np.array([1,.5,.5,0])});self.assertEqual(p[0,:2].tolist(),[1,2])
    def test_proposals_padding(self):p,s=propose({});self.assertTrue((p==-1).all());self.assertEqual(s.sum(),0)
    def test_top64_neighbors(self):
        hs={i:h(i,[(10,0,0,i+1)]) for i in range(100)}
        self.assertEqual(len(select_neighbors(np.array([10]),list(hs),hs)),64)
    def test_similarity_normalized_by_full_session(self):
        nn=select_neighbors(np.array([10]),[1],self.hist);self.assertAlmostEqual(nn[0][1],1/math.sqrt(3))
    def test_multi_item_similarity(self):
        nn=select_neighbors(np.array([20,10]),[1],self.hist)
        self.assertAlmostEqual(nn[0][1],1.5/math.sqrt(1.25*3));self.assertEqual(nn[0][3],2)
    def test_randomized_independent_oracle(self):
        # Independent dictionary/count implementation, including both context arms.
        for seed in range(30):
            rng=np.random.default_rng(seed);history={}
            for sid in range(8):
                m=int(rng.integers(1,12));history[sid]=h(sid,[(int(rng.integers(0,9)),int(rng.integers(0,3)),2*j,j+1) for j in range(m)])
            anchors=np.array([1,2,3]);candidates=np.arange(12)
            actual=build_query(anchors,candidates,list(history),history)
            for arm in actual:
                query=[1,2,3] if arm=='session_context' else [1];weights={a:1/(i+1) for i,a in enumerate(query)}
                chosen=[]
                for sid,hh in history.items():
                    common=set(map(int,hh.aids))&set(query)
                    if common:
                        score=sum(weights[a] for a in sorted(common))/math.sqrt(sum(w*w for w in weights.values())*len(set(hh.aids.tolist())))
                        pivot=max(int(ix) for a,ix in zip(hh.aids,hh.indices) if int(a) in common)
                        chosen.append((sid,score,pivot))
                chosen.sort(key=lambda x:(-x[1],-int(history[x[0]].timestamps[-1]),x[0]));chosen=chosen[:64]
                total=sum(s for _,s,_ in chosen);expected=np.zeros((12,12))
                for candidate in candidates:
                    for kind in range(3):
                        support=[];after=[]
                        for sid,score,pivot in chosen:
                            hh=history[sid];indices=[int(ix) for a,k,ix in zip(hh.aids,hh.kinds,hh.indices) if a==candidate and k==kind]
                            if indices:support.append(score)
                            if any(ix>pivot for ix in indices):after.append(score)
                        if support:expected[candidate,4*kind:4*kind+4]=[math.log1p(len(support)),sum(support)/total,max(support),sum(after)/total]
                np.testing.assert_allclose(actual[arm]['features'],expected,rtol=1e-6,atol=1e-7)
