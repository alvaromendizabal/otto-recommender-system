"""Synthetic integration of the NEW runner with a fixture-only prior-data adapter.

Uses real LightGBM. This is not a replay of private AWS data, native controls, or
its installed Polars loader. Every fixture and its small model stays in a tempdir.
"""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import lightgbm as lgb
import experiment as e
import report
from core import save_json,save_arrays,sha,array_digest,target_arrays,forward_folds,hits_at20,pooled
from experiment import NAMES, SEEN_SIGNAL


def fixture_sample(y, aids, sid, budget, seed):
    positives=np.flatnonzero(y.any(axis=1))
    negatives=np.flatnonzero(~y.any(axis=1))
    rng=np.random.default_rng(sid+seed)
    return np.sort(np.concatenate([positives,rng.choice(negatives,min(budget,len(negatives)),replace=False)]))


class SyntheticScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup);cls.root=Path(cls.tmp.name)
        cls.old=cls.root/'old';cls.out=cls.root/'outputs';cls.out.mkdir()
        base_names=json.loads((e.ROOT/'evidence/ROUND02_REFERENCE.json').read_text())['previous_feature_contract']['names'][:134]
        rng=np.random.default_rng(3306);n=64
        cls.ids=np.arange(n,dtype=np.int64)+9000
        cls.aids=np.broadcast_to(np.arange(400,dtype=np.int64),(n,400)).copy()
        cls.base=rng.lognormal(1,1,size=(n,400,134)).astype(np.float32)
        # Fixture includes repeated ties and zero affinity; no held-out/private data.
        cls.base[rng.random(cls.base.shape)<.2]=0
        cls.base[:,:,base_names.index(SEEN_SIGNAL)]=np.where(rng.random((n,400))<.15,.1,0)
        q=e.HISTORY_END+3600000+np.arange(n,dtype=np.int64)*3500000
        cls.co={'ids':cls.ids.tolist(),'query_ts':q.tolist(),'observed_last_index':[4]*n,
                'true_counts':[[1,1,1]]*n,'denominator':[[1,1,1]]*n}
        cls.labels={int(sid):[(int((i*7+o*11)%400) if i%7 else 990000, o,int(q[i]+1000),5+o) for o in range(3)] for i,sid in enumerate(cls.ids)}
        labelsha='fixture-label-ledger';p=e.protocol();parts=[]
        for start in range(0,n,64):
            end=start+64;xx=np.concatenate([cls.base[start:end],np.zeros((64,400,24),np.float32)],axis=2)
            rel=f'features/part-{start//64:03d}.npz';digest=save_arrays(cls.old/rel,ids=cls.ids[start:end],aids=cls.aids[start:end],x=xx)
            parts.append({'start':start,'end':end,'path':rel,'sha256':digest})
        save_json(cls.old/'feature_contract.json',{'fixture_only':True,'names':base_names})
        save_json(cls.old/'feature_manifest.json',{'parts':parts,'names':base_names})
        save_json(cls.old/'cohort.json',cls.co)
        study={'labels_sha256':labelsha,'fixture_only':True}
        save_json(cls.old/'screen_contract.json',study)
        controls={};all_h=[];all_ids=[];all_folds=[];all_d=[]
        truth=target_arrays(cls.ids,cls.aids,q,np.array(cls.co['observed_last_index']),np.array(cls.co['true_counts']),cls.labels,e.FIT_END)
        for fi,fold in enumerate(forward_folds(cls.ids,q)):
            tr,va=fold['train'],fold['valid']
            tt=target_arrays(cls.ids[tr],cls.aids[tr],q[tr],np.full(len(tr),4),np.ones((len(tr),3),int),cls.labels,e.FIT_END,cutoff=fold['cutoff'])
            selected=[fixture_sample(tt[j],cls.aids[i],int(cls.ids[i]),60,20260908) for j,i in enumerate(tr)]
            groups=[len(v) for v in selected]
            yy=np.concatenate([tt[j,s] for j,s in enumerate(selected)])
            xt=np.concatenate([cls.base[i,s] for i,s in zip(tr,selected,strict=True)])
            xv=cls.base[va].reshape(-1,134)
            foldpath=cls.old/'folds'/f'fold-{fi}.json'
            save_json(foldpath,{'training_ids':cls.ids[tr].tolist(),'validation_ids':cls.ids[va].tolist(),
                'cutoff_ms':fold['cutoff'],'sampled_row_ids_sha256':array_digest(np.concatenate(selected))})
            hits=np.empty((len(va),3),np.int64)
            for j,o in enumerate(e.OBJECTIVES):
                model=lgb.train(p['model_params'],lgb.Dataset(xt,label=yy[:,j],group=groups,feature_name=base_names),num_boost_round=150)
                rel=f'models/fold-{fi}/control_shared/{o}/model.txt';path=cls.old/rel;path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(path));hh=sha(path);controls[rel]=hh
                contract={'study':study,'fold':fi,'arm':'control_shared','objective':o,'names':base_names,
                    'fold_sha256':sha(foldpath),'training_x_sha256':array_digest(xt),'training_y_sha256':array_digest(yy[:,j]),
                    'validation_x_sha256':array_digest(xv),'groups_sha256':array_digest(np.asarray(groups,np.int64))}
                save_json(path.parent/'receipt.json',{'contract':contract,'model_sha256':hh})
                hits[:,j]=hits_at20(model.predict(xv,num_threads=4).reshape(-1,400),cls.aids[va],truth[va,:,j])
            den=np.ones((len(va),3),np.int64)
            save_json(cls.old/'arm_reports'/f'fold-{fi}-control_shared.json',{'session_ids':cls.ids[va].tolist(),
                'hits_by_session':hits.tolist(),'denominators':den.tolist(),'metrics':pooled(hits,den)})
            all_h.append(hits);all_ids.append(cls.ids[va]);all_folds.extend([fi]*len(va));all_d.append(den)
        cls.controlstats=cls.root/'control_stats.npz'
        save_arrays(cls.controlstats,ids=np.concatenate(all_ids),folds=np.array(all_folds,np.int8),denominator=np.concatenate(all_d),control_shared=np.concatenate(all_h))
        cls.new_fits=0;cls.control_calls=0
        def model_or_reuse(directory,contract,x,y,groups,xv,names,params,rounds):
            if cls.control_calls!=6:raise AssertionError('Challenger fit before all six saved-control replays')
            directory.mkdir(parents=True,exist_ok=True);path=directory/'model.txt'
            if path.exists():
                r=json.loads((directory/'receipt.json').read_text());assert r['contract']==contract;assert sha(path)==r['model_sha256']
                return lgb.Booster(model_file=str(path)).predict(xv,num_threads=4),0
            model=lgb.train(params,lgb.Dataset(x,label=y,group=groups,feature_name=names),num_boost_round=rounds)
            model.save_model(str(path));loaded=lgb.Booster(model_file=str(path));pred=model.predict(xv,num_threads=4)
            np.testing.assert_array_equal(pred,loaded.predict(xv,num_threads=4))
            save_json(directory/'receipt.json',{'contract':contract,'model_sha256':sha(path)})
            cls.new_fits+=1
            return pred,1
        cls.ctx={'old':cls.old,'paths':{},'ids':cls.ids,'cohort':cls.co,'manifest':{'parts':parts},
            'base_names':base_names,'feature':{},'protocol':p,'controls':controls,
            'prior':types.SimpleNamespace(labels_for=lambda paths,ids:(cls.labels,labelsha),model_or_reuse=model_or_reuse)}
        module=types.ModuleType('otto_recsys.research.dataset');module.sampled_rows=fixture_sample
        cls.module=module
        real_control=e.control_prediction
        def counted(*args,**kwargs):
            pred=real_control(*args,**kwargs);cls.control_calls+=1;return pred
        with mock.patch.object(e,'OUT',cls.out),mock.patch.dict('sys.modules',{'otto_recsys.research.dataset':module}),mock.patch.object(e,'control_prediction',side_effect=counted):
            cls.fixture_data={'ids':cls.ids,'candidates':cls.aids,'anchors':np.array([[10+i%20,40+i%17,-1,-1] for i in range(n)],np.int64)}
            from neighbors import group_history
            histrows=[]
            for sid in range(1,301):
                items=[10+sid%20,40+sid%17,(sid*7)%400,(sid*7+11)%400,(sid*7+22)%400]
                for ix,item in enumerate(items):histrows.append((sid,item,e.HISTORY_END-10000+ix*1000,min(max(ix-2,0),2),ix))
            history=group_history(np.array(histrows,np.int64),e.HISTORY_END,set())
            lookup={}
            for sid,h in history.items():
                for a in set(map(int,h.aids)):lookup.setdefault(a,set()).add(sid)
            frequencies={a:len(ss) for a,ss in lookup.items()}
            cls.identity=lambda ctx:{'fixture_only':True,'cohort_sha256':sha(ctx['old']/'cohort.json'),'ids_sha256':array_digest(ctx['ids'])}
            cls.patches=[mock.patch.object(e,'feature_identity',side_effect=cls.identity),
              mock.patch.object(e,'evidence',return_value=(cls.fixture_data,history,lookup,frequencies))]
            for patch in cls.patches:patch.start();cls.addClassCleanup(patch.stop)
            cls.feature_run=e.features(cls.ctx)
            cls.stage_result=e.screen(cls.ctx)
        cls.result=report.validate_result(cls.out,cls.controlstats,expected_sessions=n//2)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_all_control_replays_before_twelve_fits(self):
        self.assertEqual((self.control_calls,self.new_fits),(6,12));self.assertEqual(self.stage_result['control_models_replayed'],6)
    def test_both_arms_use_158_features(self):
        self.assertEqual({r['features'] for r in self.result['fold_results'] if r['arm']!='control_shared'},{158})
    def test_true_missing_candidates_remain_in_metric(self):
        c=json.loads((self.out/'candidate_coverage.json').read_text())['pooled']
        self.assertGreater(c['weighted_missing_candidate_gap'],0)
    def test_context_before_sampling(self):
        fm=json.loads((self.out/'feature_manifest.json').read_text());self.assertEqual(fm['sessions'],len(self.ids))
    def test_native_models_and_zero_refits(self):
        self.assertEqual(len(self.result['model_inventory']),12);self.assertEqual(self.result['control_models_refit'],0)
    def test_training_support_not_holdout_selected(self):
        records=json.loads((self.out/'positive_support.json').read_text())
        self.assertEqual(len(records),48)
        self.assertTrue(all(v['censored_targets'] and not v['label_based_feature_selection'] for v in records))
    def test_feature_replay_reuses_completed_chunks(self):
        with mock.patch.object(e,'OUT',self.out):run=e.features(self.ctx)
        self.assertEqual(run['new_sessions'],0)
    def test_result_replay_does_not_refit(self):
        actual=report.validate_result
        with mock.patch.object(e,'OUT',self.out),mock.patch.object(report,'validate_result',side_effect=lambda out:actual(out,self.controlstats,expected_sessions=len(self.ids)//2)):
            run=e.screen(self.ctx)
        self.assertEqual(run['new_model_fits'],0);self.assertEqual(self.new_fits,12)
    def test_report_has_nine_plotly_charts(self):
        r=report.create_report(self.out,self.controlstats,expected_sessions=len(self.ids)//2)
        receipt=json.loads((self.out/'report_receipt.json').read_text())
        self.assertEqual(receipt['charts'],9)
        self.assertEqual(len(list((self.out/'plots').glob('*.json'))),9)
        self.assertEqual(self.new_fits,12)
    def test_context_diagnostics_use_training_only(self):
        rows=json.loads((self.out/'context_diagnostics.json').read_text())
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['training_only'] and not r['labels_used_to_define_strata'] for r in rows))
        self.assertTrue(all(r['mean_seen_candidates']>0 for r in rows))
    def test_corrupted_schema_fails_before_fitting(self):
        ctx=copy.copy(self.ctx);ctx['ids']=self.ids[::-1]
        with mock.patch.object(e,'OUT',self.out),self.assertRaises(ValueError):e.screen(ctx)
        self.assertEqual(self.new_fits,12)
    def test_feature_identity_tamper_stops(self):
        ctx=copy.copy(self.ctx);ctx['old']=self.root/'missing'
        with mock.patch.object(e,'OUT',self.out),self.assertRaises(FileNotFoundError):e.screen(ctx)
