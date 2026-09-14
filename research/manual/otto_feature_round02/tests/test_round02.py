"""Offline contract, frozen-formula, native-checkpoint and report tests; synthetic data only."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.sparse import csr_matrix
import core
import intent_features as timing
import replication
import report

ROOT=Path(__file__).resolve().parents[1]


def target_fixture():
    ids=np.array([10]);aids=np.array([[1,2,3]]);q=np.array([100]);last=np.array([2]);counts=np.array([[1,1,1]])
    labels={10:[(1,0,100,3),(2,1,110,4),(9,2,130,5)]}
    return ids,aids,q,last,counts,labels,200


def synthetic_report(out):
    rng=np.random.default_rng(73);n=2048
    den=np.ones((n,3),dtype=np.int64)
    base=rng.integers(0,2,(n,3),dtype=np.int64);new=base.copy()
    for j in range(3):
        for group in (np.arange(1024),np.arange(1024,2048)):
            locations=group[base[group,j]==0][:4];new[locations,j]=1
    f=np.repeat([0,1],1024).astype(np.int8);ids=np.arange(n,dtype=np.int64)
    arms={name:core.pooled(h,den) for name,h in [('control_shared',base),('plus_timing',new)]}
    records=[]
    for k in (0,1):
        for name,h in [('control_shared',base),('plus_timing',new)]:
            records.append({'fold':k,'arm':name,'features':134 if name=='control_shared' else 158,
                            **core.pooled(h[f==k],den[f==k])})
    ci=core.paired_interval(new-base,den,f);g=arms['plus_timing']['weighted_recall_at_20']-arms['control_shared']['weighted_recall_at_20']
    fg=[core.pooled(new[f==k],den[f==k])['weighted_recall_at_20']-
        core.pooled(base[f==k],den[f==k])['weighted_recall_at_20'] for k in (0,1)]
    og=arms['plus_timing']['recall']['orders']-arms['control_shared']['recall']['orders']
    extra=(new-base).sum(axis=0);tot=den.sum(axis=0)
    digest=core.save_arrays(out/'evaluation_statistics.npz',ids=ids,folds=f,denominator=den,control_shared=base,plus_timing=new)
    r={'status':'ROUND02_SCREEN_COMPLETED','source_commit':replication.COMMIT,'round01_session_overlap':0,
       'model_count':12,'selection_access':False,'evaluation_access':False,'competition_test_access':False,
       'sessions':4096,'validation_sessions':2048,'arms':arms,'fold_results':records,'statistics_sha256':digest,
       'comparison':{'gain':g,'fold_gains':fg,'order_gain':og,**ci,
                     'decision':core.decision(g,fg,og,ci['descriptive_95_interval']),
                     'net_extra_hits':dict(zip(core.OBJECTIVES,map(int,extra),strict=True)),
                     'weighted_contributions':dict(zip(core.OBJECTIVES,map(float,extra/tot*core.WEIGHTS),strict=True)),
                     'one_order_hit_weight':float(.6/tot[2]),'gain_less_one_order_hit':float(g-.6/tot[2])}}
    core.save_json(out/'result.json',r)
    diagnostics=[{'fold':k,'name':name,'training_sessions_sampled':64,'nonzero_fraction':.1+j*.02,
                  'std':.3,'max_abs_corr_control':.7,'automatic_selection':False}
                 for k in (0,1) for j,name in enumerate(replication.TIMING_NAMES)]
    core.save_json(out/'diagnostics.json',diagnostics)
    return r


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


class FrozenFeatureTests(unittest.TestCase):
    def test_original_bytes_and_schema_preserved(self):
        ref=json.loads((ROOT/'evidence/ROUND01_REFERENCE.json').read_text())
        self.assertEqual(core.sha(ROOT/'intent_features.py'),ref['timing_code_sha256'])
        self.assertEqual(len(replication.TIMING_NAMES),24)
    def test_weighted_mean_log_age_not_log_mean_age(self):
        x=timing.summarize_support(np.array([[1.],[3.]]),np.array([1,2]),np.array([9]),np.array([0.,1.]))
        self.assertAlmostEqual(float(x[0,2]),.75*np.log(2),places=6)
        self.assertAlmostEqual(float(x[0,3]),np.log(2),places=6)
    def test_self_edges_excluded(self):
        x=timing.summarize_support(np.array([[100.],[1.]]),np.array([9,2]),np.array([9]),np.array([0.,2.]))
        self.assertAlmostEqual(float(x[0,2]),np.log(3),places=6)
    def test_pending_cart_is_observed_action_state(self):
        p=timing.seed_pools(np.array([1,1,2,2]),np.array([1,2,3,4]),np.array([1,2,2,1]),4)
        np.testing.assert_array_equal(p['pending_cart'][0],[2])
    def test_future_prefix_rejected(self):
        with self.assertRaises(ValueError):timing.seed_pools(np.array([1]),np.array([9]),np.array([0]),8)
    def test_empty_support_stays_zero(self):
        x=timing.summarize_support(np.zeros((0,2)),np.array([],int),np.array([1,2]),np.array([]))
        np.testing.assert_array_equal(x,np.zeros((2,4)))
    def test_complete_prefix_contract(self):
        ledger={'first_ts':100,'query_ts':110,'observed_events':2,'observed_last_index':1,'period_end':200}
        core.validate_prefix(np.array([1,2]),np.array([100,110]),np.array([0,1]),np.array([0,1]),ledger,100,200)
        with self.assertRaises(ValueError):core.validate_prefix(np.array([1,2]),np.array([100,110]),np.array([0,1]),np.array([0,2]),ledger,100,200)
    def test_unknown_catalogue_items_zero_edges(self):
        graphs={f:[csr_matrix((3,3)) for _ in range(3)] for f in timing.FAMILIES}
        x=timing.transform(np.array([99]),np.array([100]),np.array([0]),np.array([1,99]),graphs,100)
        np.testing.assert_array_equal(x[:,list(timing.TIMING)],np.zeros((2,24)))


class CheckpointTests(unittest.TestCase):
    def test_atomic_replay_and_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';core.save_json(p,{'v':1});core.save_json(p,{'v':1})
            with self.assertRaises(ValueError):core.save_json(p,{'v':2})
            self.assertEqual(json.loads(p.read_text()),{'v':1})
    def test_stable_arrays_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';h=core.save_arrays(p,a=np.arange(10));self.assertEqual(h,core.save_arrays(p,a=np.arange(10)))
            with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['a'],np.arange(10))
    def test_pickle_and_unsafe_npz_names_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):core.save_arrays(Path(d)/'x.npz',x=np.array([{}],object))
            with self.assertRaises(ValueError):core.save_arrays(Path(d)/'x.npz',**{'../x':np.arange(3)})
    def test_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'link';p.symlink_to(Path(d)/'absent')
            with self.assertRaises(ValueError):core.write_once(p,b'a')
    def test_corrupt_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file';p.write_bytes(b'old');h=core.sha(p);p.write_bytes(b'new')
            with self.assertRaises(ValueError):core.checked(p,h)
    @unittest.skipUnless(importlib.util.find_spec('lightgbm'),'LightGBM required for native checkpoint test')
    def test_native_model_roundtrip_and_zero_refit(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(80,2)).astype(np.float32);y=np.tile([1,0,0,0],20)
        params={'objective':'lambdarank','metric':'None','verbosity':-1,'num_threads':1,'num_leaves':3,'min_data_in_leaf':2,'deterministic':True,'force_col_wise':True,'seed':4}
        with tempfile.TemporaryDirectory() as d:
            p,n=replication.model_or_reuse(Path(d),{'test':True},x,y,[4]*20,x,['a','b'],params,rounds=5)
            p2,n2=replication.model_or_reuse(Path(d),{'test':True},x,y,[4]*20,x,['a','b'],params,rounds=5)
            self.assertEqual((n,n2),(1,0));np.testing.assert_array_equal(p,p2)
            with self.assertRaises(ValueError):replication.model_or_reuse(Path(d),{'test':False},x,y,[4]*20,x,['a','b'],params,5)
    def test_orphan_model_is_not_refitted(self):
        if importlib.util.find_spec('lightgbm') is None:self.skipTest('LightGBM unavailable')
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'model.txt').write_text('orphan')
            with self.assertRaisesRegex(ValueError,'Orphan'):
                replication.model_or_reuse(Path(d),{},np.ones((4,1)),np.ones(4),[4],np.ones((4,1)),['a'],{},5)


class ReportTests(unittest.TestCase):
    def test_independent_report_score_and_interval_replay(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic_report(Path(d));r,data=report.validate_saved_result(Path(d))
            self.assertEqual(len(data['ids']),2048);self.assertEqual(r['status'],'ROUND02_SCREEN_COMPLETED')
    def test_tampered_score_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=synthetic_report(p);r['arms']['plus_timing']['weighted_recall_at_20']+=.01
            (p/'result.json').write_bytes(core.encode(r))
            with self.assertRaises(ValueError):report.validate_saved_result(p)
    def test_tampered_interval_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=synthetic_report(p);r['comparison']['descriptive_95_interval'][0]-=.01
            (p/'result.json').write_bytes(core.encode(r))
            with self.assertRaises(ValueError):report.validate_saved_result(p)
    def test_immutable_seven_plotly_specs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);out=p/'outputs';out.mkdir();synthetic_report(out)
            report.make_report(p);report.make_report(p)
            self.assertEqual(len(list((out/'plots').glob('*.json'))),7)
            self.assertGreater((out/'round02_report.html').stat().st_size,10000)


if __name__=='__main__':unittest.main()
