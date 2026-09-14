import unittest
import copy
import numpy as np
import core

def target_fixture():
    ids=np.array([10]);aids=np.array([[1,2,3]]);q=np.array([100]);last=np.array([2]);counts=np.array([[1,1,1]])
    labels={10:[(1,0,100,3),(2,1,110,4),(9,2,130,5)]}
    return ids,aids,q,last,counts,labels,200


class CohortAndFoldTests(unittest.TestCase):
    def test_disjoint_deterministic_selection(self):
        a=np.arange(10000);old=np.arange(1024)
        b=core.choose_cohort(a,old)
        self.assertEqual(len(b),4096);self.assertEqual(np.intersect1d(b,old).size,0)
        np.testing.assert_array_equal(b,core.choose_cohort(a[::-1],old))
    def test_no_target_argument_in_selection(self):
        import inspect
        self.assertEqual(list(inspect.signature(core.choose_cohort).parameters),['all_fit_ids','excluded_ids','count','seed'])
    def test_seed_changes_membership(self):
        a=np.arange(10000);old=np.arange(1024)
        self.assertFalse(np.array_equal(core.choose_cohort(a,old),core.choose_cohort(a,old,seed=123)))
    def test_invalid_cohort_input(self):
        for values,old in [(np.array([1,1]),np.array([1])),(np.array([1.,2.]),np.array([1])),
                           (np.arange(10),np.array([99])),(np.arange(10),np.array([1]))]:
            with self.subTest(values=values),self.assertRaises(ValueError):core.choose_cohort(values,old)
    def test_forward_folds_full_replication_shape(self):
        ids=np.arange(4096);q=np.arange(4096,dtype=np.int64)*60000+10**12
        folds=core.forward_folds(ids,q)
        self.assertEqual([len(f['valid']) for f in folds],[1024,1024])
        for f in folds:
            self.assertTrue((q[f['train']]<f['cutoff']-21600000).all())
            self.assertEqual(np.intersect1d(f['train'],f['valid']).size,0)
        self.assertEqual(np.intersect1d(folds[0]['valid'],folds[1]['valid']).size,0)
    def test_tied_queries_do_not_leak_across_time(self):
        ids=np.arange(32);q=np.repeat(np.arange(8,dtype=np.int64),4)*100
        for f in core.forward_folds(ids,q,embargo_ms=0):
            self.assertTrue((q[f['train']]<f['cutoff']).all())
    def test_insufficient_embargo_support(self):
        with self.assertRaises(ValueError):core.forward_folds(np.arange(16),np.ones(16,dtype=int))
    def test_duplicate_fold_ids_rejected(self):
        with self.assertRaises(ValueError):core.forward_folds(np.ones(16,dtype=int),np.arange(16))


class TargetAndMetricTests(unittest.TestCase):
    def test_equal_timestamp_later_index_valid(self):
        y=core.target_arrays(*target_fixture());np.testing.assert_array_equal(y[0,:,0],[1,0,0])
    def test_exclusive_censor_happens_before_sampling(self):
        y=core.target_arrays(*target_fixture(),cutoff=110)
        self.assertEqual(int(y.sum()),1)
    def test_target_after_period_end_rejected(self):
        args=list(target_fixture());args[5][10][1]=(2,1,200,4)
        with self.assertRaises(ValueError):core.target_arrays(*args)
    def test_pre_prefix_label_rejected(self):
        args=list(target_fixture());args[5][10][0]=(1,0,99,3)
        with self.assertRaises(ValueError):core.target_arrays(*args)
    def test_same_index_label_rejected(self):
        args=list(target_fixture());args[5][10][0]=(1,0,100,2)
        with self.assertRaises(ValueError):core.target_arrays(*args)
    def test_duplicate_label_rejected(self):
        args=list(target_fixture());args[5][10].append(args[5][10][1])
        with self.assertRaises(ValueError):core.target_arrays(*args)
    def test_missing_truth_not_removed_from_denominator(self):
        args=target_fixture();y=core.target_arrays(*args)
        self.assertEqual(y[:,:,2].sum(),0)
        self.assertEqual(args[4][0,2],1)
    def test_count_mismatch_and_multiple_clicks_rejected(self):
        args=list(target_fixture());args[4][0,1]=2
        with self.assertRaises(ValueError):core.target_arrays(*args)
        args=list(target_fixture());args[4][0,0]=2;args[5][10].append((3,0,105,6))
        with self.assertRaises(ValueError):core.target_arrays(*args)
    def test_immutable_input_target_generation(self):
        args=target_fixture();labels=copy.deepcopy(args[5]);core.target_arrays(*args)
        self.assertEqual(labels,args[5])
    def test_training_query_after_cutoff_rejected(self):
        with self.assertRaises(ValueError):core.target_arrays(*target_fixture(),cutoff=100)
    def test_hit_ranking_ties_use_item_id(self):
        aids=np.array([np.arange(30)[::-1]]);scores=np.zeros_like(aids,float);y=(aids<20).astype(int)
        self.assertEqual(core.hits_at20(scores,aids,y).tolist(),[20])
    def test_complete_pool_no_positive_dropping(self):
        h=np.array([[0,1,0],[1,0,1]]);d=np.array([[1,2,1],[1,0,1]])
        self.assertAlmostEqual(core.pooled(h,d)['weighted_recall_at_20'],.5)
    def test_metric_invalid_support(self):
        with self.assertRaises(ValueError):core.pooled(np.zeros((2,3),int),np.zeros((2,3),int))
        with self.assertRaises(ValueError):core.pooled(np.ones((2,3),int)*2,np.ones((2,3),int))
    def test_bootstrap_identical_models_zero_interval(self):
        ci=core.paired_interval(np.zeros((40,3),int),np.ones((40,3),int),np.repeat([0,1],20),replicates=50)
        self.assertEqual(ci['descriptive_95_interval'],[0.,0.])
    def test_bootstrap_repeatable_and_paired(self):
        d=np.ones((40,3),int);delta=np.zeros_like(d);delta[:20,2]=1;f=np.repeat([0,1],20)
        a=core.paired_interval(delta,d,f,100);b=core.paired_interval(delta,d,f,100)
        self.assertEqual(a,b);self.assertAlmostEqual(a['descriptive_95_interval'][0],.3)
    def test_decision_does_not_promote_uncertain_or_unstable(self):
        self.assertIn('UNCERTAINTY',core.decision(.01,[.01,.01],0,[-.001,.02]))
        self.assertIn('UNSTABLE',core.decision(.01,[-.001,.03],.01,[.001,.02]))
        self.assertIn('DO_NOT_RETAIN',core.decision(-.01,[0,0],0,[-.02,0]))
        self.assertIn('NOT_PROMOTION',core.decision(.01,[.01,.01],0,[.001,.02]))


