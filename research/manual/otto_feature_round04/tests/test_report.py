import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import pooled, save_arrays, save_json, sha
from analysis_utils import compare
from demand_features import NAMES
from report import validate_result, create_report


def make_fixture(root):
    """Fictional challenger outputs on actual saved control rows; TEST ONLY, never shipped as results."""
    source=Path(__file__).resolve().parents[1]/'evidence/ROUND02_CONTROL_STATISTICS.npz'
    with np.load(source,allow_pickle=False) as z:
        v={k:z[k] for k in ('ids','folds','denominator','control_shared')}
    # Equal arms make an exact no-improvement fixture rather than a fabricated promising score.
    for a in ('demand_rolling','demand_static_ablation'):v[a]=v['control_shared'].copy()
    save_arrays(root/'evaluation_statistics.npz',**v)
    arms=['control_shared','demand_rolling','demand_static_ablation'];rows=[]
    for f in (0,1):
        ix=v['folds']==f
        for a in arms:rows.append({'arm':a,'fold':f,'features':134 if a==arms[0] else 209,**pooled(v[a][ix],v['denominator'][ix])})
    comp={}
    for l,r in [('demand_rolling','control_shared'),('demand_static_ablation','control_shared'),('demand_rolling','demand_static_ablation')]:
        comp[l+'_minus_'+r]=compare(v[l],v[r],v['denominator'],v['folds'])
    result={'status':'ROUND04_SCREEN_COMPLETED','fixture_only':True,
            'arms':{a:pooled(v[a],v['denominator']) for a in arms},'fold_results':rows,'comparisons':comp,
            'statistics_sha256':sha(root/'evaluation_statistics.npz')}
    save_json(root/'result.json',result)
    save_json(root/'diagnostics.json',[{'fold':f,'name':n,'support_fraction':.3,'recent_positive_fraction':.1,
             'max_abs_corr_control':.4,'max_abs_corr_added':.5,'std':.5,'training_sessions_sampled':64} for f in (0,1) for n in NAMES])
    return result

class Reports(unittest.TestCase):
    def test_exact_integer_replay(self):
        with tempfile.TemporaryDirectory() as d:
            r=make_fixture(Path(d));self.assertEqual(validate_result(Path(d)),r)
    def test_corrupt_score_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=make_fixture(p);r['arms']['demand_rolling']['weighted_recall_at_20']+=.01
            (p/'result.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):validate_result(p)
    def test_changed_decision_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=make_fixture(p);r['comparisons']['demand_rolling_minus_control_shared']['decision']='WINNER'
            (p/'result.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):validate_result(p)
    def test_seven_charts_and_repeat_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);make_fixture(p);before=sha(p/'result.json');create_report(p);create_report(p)
            self.assertEqual(before,sha(p/'result.json'));self.assertEqual(len(list((p/'plots').glob('*.json'))),7)
    def test_primary_negative_decision(self):
        den=np.ones((40,3),np.int64);base=den.copy();worse=base.copy();worse[0,0]=0
        f=np.r_[np.zeros(20,int),np.ones(20,int)]
        self.assertEqual(compare(worse,base,den,f)['decision'],'STOP_THIS_CONFIGURATION_NO_GAIN')

if __name__=='__main__':unittest.main()
