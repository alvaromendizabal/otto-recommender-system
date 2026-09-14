"""Tests supplied for USER execution; no pass claim is made during preparation."""
import unittest
import numpy as np
from scipy import sparse
from latent_features import affinity_summaries, names, normalize, positions
import latent_models as lm

class FeatureContracts(unittest.TestCase):
    def setUp(self):
        self.v=np.array([10,20,30,40],np.int64)
        x=np.array([[1.,0],[0,1],[1,1],[-1,0]])
        self.q=np.broadcast_to(normalize(x),(3,4,2)).copy();self.t=self.q.copy()
    def f(self,a=(10,20),c=(30,40)):
        return affinity_summaries(np.array(a,np.int64),np.array(c,np.int64),self.v,self.q,self.t)
    def test_equal_24_feature_arms(self):
        for r in (11,12):
            self.assertEqual(len(names(r)),24);self.assertEqual(len(names(r,True)),24)
            self.assertEqual(len(set(names(r)+names(r,True))),48)
    def test_known_cosines(self):
        x=self.f();self.assertAlmostEqual(float(x[0,0]),2**-.5,places=6)
        self.assertAlmostEqual(float(x[0,3]),0.,places=6)
    def test_negative_cosine_preserved(self):self.assertEqual(float(self.f()[1,0]),-1.)
    def test_self_anchor_removed(self):
        x=self.f(a=(10,),c=(10,));np.testing.assert_array_equal(x,0)
    def test_unknown_candidate_zero(self):np.testing.assert_array_equal(self.f(c=(999,)),0)
    def test_unknown_anchor_zero(self):np.testing.assert_array_equal(self.f(a=(999,)),0)
    def test_partial_availability(self):self.assertEqual(float(self.f(a=(10,999))[0,7]),.5)
    def test_query_permutation_is_meaningful(self):self.assertNotEqual(float(self.f()[1,0]),float(self.f(a=(20,10))[1,0]))
    def test_candidate_permutation(self):np.testing.assert_array_equal(self.f(c=(40,30)),self.f()[::-1])
    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):self.f(c=(30,30))
    def test_duplicate_anchors_rejected(self):
        with self.assertRaises(ValueError):self.f(a=(10,10))
    def test_zero_norm_is_unavailable(self):
        self.t[:]=0;np.testing.assert_array_equal(self.f(),0)
    def test_nonfinite_refused(self):
        self.q[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.f()
    def test_invalid_vocabulary_rejected(self):
        self.v=self.v[::-1]
        with self.assertRaises(ValueError):self.f()
    def test_fixed_dtype(self):self.assertEqual(self.f().dtype,np.float32)
    def test_rotations_leave_cosine_features_unchanged(self):
        base=self.f();rotation=np.array([[0.,-1.],[1.,0.]])
        self.q=self.q@rotation;self.t=self.t@rotation
        np.testing.assert_allclose(self.f(),base,atol=1e-6)

class RepresentationContracts(unittest.TestCase):
    def setUp(self):
        cut=lm.HISTORY_END
        self.x=np.array([[1,10,cut-6000,0,0],[1,20,cut-5000,1,1],
                         [1,20,cut-4000,1,2],[1,30,cut-3000,2,4],
                         [2,10,cut-6000,0,0],[2,40,cut-2000,2,8]],np.int64)
        self.v=np.array([10,20,30,40],np.int64)
    def test_cutoff_exclusive(self):
        self.x[0,2]=lm.HISTORY_END
        with self.assertRaises(ValueError):lm.validate_events(self.x)
    def test_study_session_excluded(self):
        with self.assertRaises(ValueError):lm.validate_events(self.x,[1])
    def test_duplicate_index_rejected(self):
        self.x[1,4]=0
        with self.assertRaises(ValueError):lm.validate_events(self.x)
    def test_backward_timestamp_rejected(self):
        self.x[1,2]=lm.HISTORY_END-9000
        with self.assertRaises(ValueError):lm.validate_events(self.x)
    def test_vocabulary_uses_distinct_sessions(self):
        v,f=lm.select_vocabulary(self.x)
        self.assertEqual(dict(zip(v,f))[20],1);self.assertEqual(dict(zip(v,f))[10],2)
    def test_stable_vocabulary_ties(self):
        v,_=lm.select_vocabulary(self.x,2);np.testing.assert_array_equal(v,[10,20])
    def test_incidence_is_binary(self):
        m=lm.incidence(self.x,self.v);self.assertEqual(float(m[2][1,0]),1.)
    def test_original_gaps_not_compressed(self):
        f,b=lm.pair_rows(self.x,self.v,1)
        self.assertFalse(any(r[0]==2 for r in f))
    def test_no_cross_session_pairs(self):
        f,b=lm.pair_rows(self.x,self.v,1)
        self.assertTrue(all(r[0]==1 for r in f))
    def test_reverse_uses_destination_action(self):
        f,b=lm.pair_rows(self.x,self.v,1)
        self.assertIn([1,1,0,0],b.tolist());self.assertIn([1,0,1,1],f.tolist())
    def test_pair_counts_once_per_session(self):
        r=np.array([[1,0,1,1],[1,0,1,1],[2,0,1,1]],np.int64)
        self.assertEqual(float(lm.count_pairs(r,4,1)[0,1]),2.)
    def test_ppmi_known_diagonal(self):
        m=sparse.csr_matrix(np.eye(2,dtype=np.float32))
        np.testing.assert_allclose(lm.ppmi(m).toarray(),np.eye(2)*np.log(2),atol=1e-6)
    def test_ppmi_independent_counts_zero(self):
        self.assertEqual(lm.ppmi(sparse.csr_matrix(np.ones((3,3)))).nnz,0)
    def test_time_window_respected(self):
        self.x[0,2]-=lm.MAX_MS+1
        f,_=lm.pair_rows(self.x,self.v,1)
        self.assertFalse(any(r[1]==0 for r in f))
    def test_sparse_factorization_shapes(self):
        m=sparse.csr_matrix(np.diag(np.arange(1,7,dtype=np.float32)))
        q,t,s=lm.transition_embedding(m)
        self.assertEqual(q.shape,(6,32));self.assertEqual(t.shape,(6,32));self.assertLessEqual(len(s),32)
    def test_source_row_permutation_preserves_membership(self):
        a=lm.incidence(self.x,self.v);b=lm.incidence(self.x[::-1],self.v)
        for x,y in zip(a,b):np.testing.assert_array_equal(x.toarray(),y.toarray())
    def test_session_projection_uses_shared_basis(self):
        m=lm.incidence(self.x,self.v);q,t,s=lm.session_embedding(m,True)
        self.assertEqual(q.shape,(3,4,32));self.assertEqual(t.shape,(3,4,32))
        np.testing.assert_allclose(q[0],q[1],atol=1e-7)
    def test_rerun_svd_same_seed(self):
        m=sparse.csr_matrix(np.diag(np.arange(1,7,dtype=np.float32)))
        a=lm.transition_embedding(m);b=lm.transition_embedding(m)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-6)
    def test_no_information_empty_matrix_stops(self):
        with self.assertRaises(ValueError):lm.svd(sparse.csr_matrix((3,3)))

class TrainingSourceBoundary(unittest.TestCase):
    def test_validation_anchor_changes_do_not_change_selected_history(self):
        ids=np.arange(64,dtype=np.int64)+9000
        times=lm.HISTORY_END+3600000+np.arange(64,dtype=np.int64)*3500000
        anchors=np.full((64,4),-1,np.int64);anchors[:,0]=np.arange(64)+100
        postings=np.column_stack((np.arange(64)+100,np.arange(64)+10000,np.full(64,lm.HISTORY_END-1000),np.ones(64))).astype(np.int64)
        selected,meta=lm.training_anchor_sessions(ids,times,anchors,postings)
        valid=lm.forward_folds(ids,times)[0]['valid'];changed=anchors.copy();changed[valid,0]=999999
        selected2,meta2=lm.training_anchor_sessions(ids,times,changed,postings)
        np.testing.assert_array_equal(selected,selected2);self.assertEqual(meta,meta2)
        self.assertFalse(meta['validation_anchors_used'])

    def test_selected_sessions_match_training_not_validation(self):
        ids=np.arange(64,dtype=np.int64)+9000
        times=lm.HISTORY_END+3600000+np.arange(64,dtype=np.int64)*3500000
        anchors=np.full((64,4),-1,np.int64);anchors[:,0]=np.arange(64)+100
        postings=np.column_stack((np.arange(64)+100,np.arange(64)+10000,np.full(64,lm.HISTORY_END-1000),np.ones(64))).astype(np.int64)
        selected,_=lm.training_anchor_sessions(ids,times,anchors,postings)
        train=lm.forward_folds(ids,times)[0]['train']
        np.testing.assert_array_equal(selected,np.arange(64)[train]+10000)

if __name__=='__main__':unittest.main()
