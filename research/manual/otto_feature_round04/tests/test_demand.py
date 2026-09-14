"""Independent snapshot correctness, leakage and restart fixtures. No real dataset/model fits."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import orjson
import core
from demand_features import transform, percentile, NAMES, HOUR, WINDOWS
from demand_index import chunk, aggregate, group_cumsum, HourlyIndex, build, PauseIndex

MINIMUM=1000;MAXIMUM=1200

def events_fixture():
    return [
        {'session':1,'events':[{'aid':1,'ts':999*HOUR+2,'type':'orders'},
                              {'aid':1,'ts':1150*HOUR,'type':'clicks'},
                              {'aid':2,'ts':1167*HOUR+20,'type':'carts'},
                              {'aid':1,'ts':1168*HOUR,'type':'clicks'},
                              {'aid':1,'ts':1169*HOUR-1,'type':'orders'},
                              {'aid':1,'ts':1169*HOUR,'type':'orders'}]},
        {'session':2,'events':[{'aid':100,'ts':1168*HOUR+5,'type':'clicks'}]},
        {'session':9,'events':[{'aid':1,'ts':1168*HOUR+6,'type':'orders'}]},
        {'session':3,'events':[{'aid':2,'ts':1200*HOUR,'type':'clicks'}]}
    ]

def write_raw(path,records):
    path.write_bytes(b''.join(orjson.dumps(r)+b'\n' for r in records))

def direct(records,candidates,cut,excluded={9}):
    count=np.zeros((len(candidates),5,3),np.int64);last=np.full((len(candidates),3),-1,np.int64)
    total=np.zeros((5,3),np.int64);ix={a:i for i,a in enumerate(candidates)};actions={'clicks':0,'carts':1,'orders':2}
    for s in records:
        if s['session'] in excluded:continue
        for e in s['events']:
            t=e['ts'];k=actions[e['type']]
            if t>=cut:continue
            if e['aid'] in ix:last[ix[e['aid']],k]=max(last[ix[e['aid']],k],t)
            for wi,h in enumerate(WINDOWS):
                if t>=cut-h*HOUR:
                    total[wi,k]+=1
                    if e['aid'] in ix:count[ix[e['aid']],wi,k]+=1
    return count,last,total

def make_index(raw,records=None,catalogue={1,2},budget=10**9):
    if records is None:records=events_fixture()
    write_raw(raw,records)
    a,r=chunk(raw,0,minimum_hour=MINIMUM,maximum_hour=MAXIMUM,excluded={9},catalogue=catalogue,budget_bytes=budget)
    return HourlyIndex(a['keys'],group_cumsum(a['keys'],a['counts'],201),a['last'],a['global_counts'],MINIMUM,MAXIMUM),a,r

class IndexTests(unittest.TestCase):
    def test_snapshot_equals_independent_event_scan(self):
        with tempfile.TemporaryDirectory() as d:
            index,_,_=make_index(Path(d)/'raw')
            for cut in (1168*HOUR,1169*HOUR,1170*HOUR,1200*HOUR):
                got=index.snapshot(np.array([1,2,3]),cut);expected=direct(events_fixture(),[1,2,3],cut)
                for a,b in zip(got,expected):np.testing.assert_array_equal(a,b)
    def test_exclusive_upper_inclusive_lower_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            index,_,_=make_index(Path(d)/'raw');c,l,g=index.snapshot(np.array([1]),1169*HOUR)
            self.assertEqual(c[0,0,0],1);self.assertEqual(c[0,0,2],1)
            self.assertEqual(l[0,2],1169*HOUR-1)
    def test_current_study_session_excluded_whole(self):
        with tempfile.TemporaryDirectory() as d:
            index,a,r=make_index(Path(d)/'raw')
            self.assertEqual(r['study_sessions_excluded'],1)
            c,_,_=index.snapshot(np.array([1]),1169*HOUR);self.assertEqual(c[0,0,2],1)
    def test_global_prior_keeps_non_candidate_item(self):
        with tempfile.TemporaryDirectory() as d:
            index,_,_=make_index(Path(d)/'raw');c,l,g=index.snapshot(np.array([1,2]),1169*HOUR)
            self.assertGreater(g[0,0],c[:,0,0].sum())
    def test_old_event_preserves_last_seen_without_window_count(self):
        rows=[{'session':1,'events':[{'aid':5,'ts':950*HOUR,'type':'orders'}]}]
        with tempfile.TemporaryDirectory() as d:
            index,_,_=make_index(Path(d)/'raw',rows,{5});c,l,_=index.snapshot(np.array([5]),1168*HOUR)
            self.assertEqual(c.sum(),0);self.assertEqual(l[0,2],950*HOUR)
    def test_future_event_change_cannot_affect_earlier_snapshot(self):
        rows=events_fixture();other=copy.deepcopy(rows);other[0]['events'][-1]['aid']=2
        with tempfile.TemporaryDirectory() as d:
            first,_,_=make_index(Path(d)/'a',rows);second,_,_=make_index(Path(d)/'b',other)
            for a,b in zip(first.snapshot(np.array([1,2]),1169*HOUR),second.snapshot(np.array([1,2]),1169*HOUR)):
                np.testing.assert_array_equal(a,b)
    def test_no_indexed_rows_for_whole_excluded_source(self):
        rows=[{'session':9,'events':[{'aid':1,'ts':1168*HOUR,'type':'orders'}]}]
        with tempfile.TemporaryDirectory() as d:
            index,a,r=make_index(Path(d)/'raw',rows)
            self.assertEqual(len(a['keys']),0);self.assertEqual(a['global_counts'].sum(),0)
            c,l,g=index.snapshot(np.array([1]),1169*HOUR);self.assertEqual(c.sum(),0);self.assertTrue((l==-1).all())
    def test_unknown_candidates_are_zero_counts_not_wrong_row(self):
        with tempfile.TemporaryDirectory() as d:
            index,_,_=make_index(Path(d)/'raw');c,l,_=index.snapshot(np.array([0,3,9999]),1169*HOUR)
            self.assertEqual(c.sum(),0);self.assertTrue((l==-1).all())
    def test_chunk_boundaries_preserve_full_session_and_recombine(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw';_,whole,_=make_index(p);parts=[];position=0
            while position<p.stat().st_size:
                a,r=chunk(p,position,minimum_hour=MINIMUM,maximum_hour=MAXIMUM,excluded={9},catalogue={1,2},budget_bytes=1)
                parts.append(a);self.assertGreater(r['end'],position);position=r['end']
            k,c,t=aggregate(np.concatenate([a['keys'] for a in parts]),np.concatenate([a['counts'] for a in parts]),np.concatenate([a['last'] for a in parts]))
            for a,b in zip((k,c,t),(whole['keys'],whole['counts'],whole['last'])):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(sum(a['global_counts'] for a in parts),whole['global_counts'])
    def test_bad_start_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw';write_raw(p,events_fixture())
            with self.assertRaisesRegex(ValueError,'boundary'):chunk(p,1,minimum_hour=MINIMUM,maximum_hour=MAXIMUM,excluded=set(),catalogue={1})
    def test_invalid_hour_alignment_or_range(self):
        with tempfile.TemporaryDirectory() as d:
            ix,_,_=make_index(Path(d)/'raw')
            for cut in (1168*HOUR+1,1167*HOUR,1201*HOUR):
                with self.subTest(cut=cut),self.assertRaises(ValueError):ix.snapshot(np.array([1]),cut)
    def test_duplicate_candidates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ix,_,_=make_index(Path(d)/'raw')
            with self.assertRaises(ValueError):ix.snapshot(np.array([1,1]),1169*HOUR)
    def test_malformed_actions_or_timestamps_rejected(self):
        for change in ({'type':'unknown'},{'ts':-1},{'aid':-1},{'ts':1.2}):
            rows=events_fixture();rows[0]['events'][0].update(change)
            with tempfile.TemporaryDirectory() as d,self.subTest(change=change),self.assertRaises(ValueError):make_index(Path(d)/'raw',rows)
    def test_nonmonotone_event_order_rejected(self):
        rows=events_fixture();rows[0]['events']=rows[0]['events'][::-1]
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):make_index(Path(d)/'raw',rows)
    def test_invalid_cumulative_shape_rejected(self):
        with self.assertRaises(ValueError):HourlyIndex(np.array([1,1]),np.array([1,2]),np.array([10,20]),np.zeros((201,3),int),1000,1200)
    def test_full_index_build_and_replay_skips_raw_processing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=p/'raw';write_raw(raw,events_fixture());out=p/'index'
            c={'minimum_hour':1000,'maximum_hour':1200,'excluded_sessions':[9],'candidate_catalogue':[1,2]}
            r=build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None,chunk_bytes=1)
            with patch('demand_index.chunk',side_effect=AssertionError('Reprocessed valid chunk')):
                second=build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None,chunk_bytes=1)
            self.assertEqual(r,second);ix=HourlyIndex.load(out)
            for a,b in zip(ix.snapshot(np.array([1,2]),1169*HOUR),direct(events_fixture(),[1,2],1169*HOUR)):
                np.testing.assert_array_equal(a,b)
    def test_pause_keeps_completed_chunks_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=p/'raw';write_raw(raw,events_fixture());out=p/'index'
            c={'minimum_hour':1000,'maximum_hour':1200,'excluded_sessions':[9],'candidate_catalogue':[1,2]}
            with self.assertRaises(PauseIndex):build(raw,out,c,deadline=0,progress={},log=lambda *a,**k:None)
            self.assertTrue((out/'index_contract.json').exists())
            r=build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None)
            self.assertEqual(r['raw_bytes_processed'],raw.stat().st_size)
    def test_changed_contract_rejected_on_restart(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=p/'raw';write_raw(raw,events_fixture());out=p/'index'
            c={'minimum_hour':1000,'maximum_hour':1200,'excluded_sessions':[9],'candidate_catalogue':[1,2]}
            build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None)
            c['excluded_sessions']=[]
            with self.assertRaises(ValueError):build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None)
    def test_corrupt_chunk_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=p/'raw';write_raw(raw,events_fixture());out=p/'index'
            c={'minimum_hour':1000,'maximum_hour':1200,'excluded_sessions':[9],'candidate_catalogue':[1,2]}
            build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None)
            next((out/'chunks').glob('*-s00.npz')).write_bytes(b'bad')
            with self.assertRaises(ValueError):build(raw,out,c,deadline=time.monotonic()+30,progress={},log=lambda *a,**k:None)
    def test_randomized_snapshots_equal_naive_calculation(self):
        rng=np.random.default_rng(81);records=[]
        for s in range(50):
            times=np.sort(rng.integers(980*HOUR,1202*HOUR,50));ids=rng.integers(0,8,50);acts=rng.choice(['clicks','carts','orders'],50)
            records.append({'session':s,'events':[{'aid':int(a),'ts':int(t),'type':str(k)} for a,t,k in zip(ids,times,acts)]})
        with tempfile.TemporaryDirectory() as d:
            ix,_,_=make_index(Path(d)/'raw',records,set(range(8)))
            for h in range(1168,1201,3):
                a=ix.snapshot(np.arange(8),h*HOUR);b=direct(records,list(range(8)),h*HOUR)
                for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

class FeatureTests(unittest.TestCase):
    def snapshot(self):return direct(events_fixture(),[1,2,3],1169*HOUR)
    def test_exact_75_unique_names_and_matrix(self):
        self.assertEqual(len(NAMES),75);self.assertEqual(len(set(NAMES)),75)
        x=transform(*self.snapshot(),cutoff=1169*HOUR,query_ts=1169*HOUR+5)
        self.assertEqual(x.shape,(3,75));self.assertEqual(x.dtype,np.float32);self.assertTrue(np.isfinite(x).all())
    def test_zero_counts_tied_percentiles(self):
        np.testing.assert_array_equal(percentile(np.zeros(3)),[.5,.5,.5])
        np.testing.assert_array_equal(percentile(np.array([0,0,2])),[.25,.25,1])
    def test_no_evidence_uniform_mix(self):
        x=transform(np.zeros((2,5,3),int),np.full((2,3),-1),np.zeros((5,3),int),cutoff=1169*HOUR,query_ts=1169*HOUR)
        np.testing.assert_allclose(x[:,-6:],[ [1/3]*6]*2,atol=1e-7)
    def test_mix_uses_global_not_candidate_distribution(self):
        c,l,g=self.snapshot();x=transform(c,l,g,cutoff=1169*HOUR,query_ts=1169*HOUR)
        g2=g.copy();g2[:,0]+=1000;y=transform(c,l,g2,cutoff=1169*HOUR,query_ts=1169*HOUR)
        self.assertFalse(np.array_equal(x[:,-6:],y[:,-6:]));np.testing.assert_array_equal(x[:,:-6],y[:,:-6])
    def test_mass_shares_sum_one_or_zero(self):
        x=transform(*self.snapshot(),cutoff=1169*HOUR,query_ts=1169*HOUR)
        for j,n in enumerate(NAMES):
            if n.endswith('mass_share'):self.assertTrue(abs(float(x[:,j].sum()))<1e-7 or abs(float(x[:,j].sum())-1)<1e-7)
    def test_action_mix_sums_one(self):
        x=transform(*self.snapshot(),cutoff=1169*HOUR,query_ts=1169*HOUR)
        np.testing.assert_allclose(x[:,-6:].reshape(3,2,3).sum(2),1,atol=1e-7)
    def test_future_last_timestamp_rejected(self):
        c,l,g=self.snapshot();l[0,0]=1169*HOUR
        with self.assertRaises(ValueError):transform(c,l,g,cutoff=1169*HOUR,query_ts=1169*HOUR)
    def test_count_monotonicity_rejected(self):
        c,l,g=self.snapshot();c[0,0,0]=20
        with self.assertRaises(ValueError):transform(c,l,g,cutoff=1169*HOUR,query_ts=1169*HOUR)
    def test_fractional_counts_rejected(self):
        c,l,g=self.snapshot()
        with self.assertRaises(ValueError):transform(c.astype(float),l,g,cutoff=1169*HOUR,query_ts=1169*HOUR)
    def test_last_seen_count_consistency(self):
        c,l,g=self.snapshot();c[0,0,2]=0
        with self.assertRaises(ValueError):transform(c,l,g,cutoff=1169*HOUR,query_ts=1169*HOUR)
    def test_invalid_prior_and_query_time(self):
        for strength,q in [(0,1169*HOUR),(float('nan'),1169*HOUR),(20,1168*HOUR)]:
            with self.subTest(strength=strength,q=q),self.assertRaises(ValueError):transform(*self.snapshot(),cutoff=1169*HOUR,query_ts=q,prior_strength=strength)
    def test_input_nonmutation_and_replay(self):
        c,l,g=self.snapshot();before=(c.copy(),l.copy(),g.copy())
        x=transform(c,l,g,cutoff=1169*HOUR,query_ts=1170*HOUR);y=transform(c,l,g,cutoff=1169*HOUR,query_ts=1170*HOUR)
        np.testing.assert_array_equal(x,y)
        for a,b in zip((c,l,g),before):np.testing.assert_array_equal(a,b)
    def test_unknown_last_age_zero_with_separate_seen(self):
        x=transform(*self.snapshot(),cutoff=1169*HOUR,query_ts=1169*HOUR)
        self.assertEqual(x[2,NAMES.index('asof_orders_last_seen')],0)
        self.assertEqual(x[2,NAMES.index('asof_orders_last_age_log_hours')],0)
    def test_candidate_order_equivariance(self):
        c,l,g=self.snapshot();x=transform(c,l,g,cutoff=1169*HOUR,query_ts=1169*HOUR)
        perm=[2,0,1];y=transform(c[perm],l[perm],g,cutoff=1169*HOUR,query_ts=1169*HOUR)
        np.testing.assert_array_equal(y,x[perm])

if __name__=='__main__':unittest.main()
