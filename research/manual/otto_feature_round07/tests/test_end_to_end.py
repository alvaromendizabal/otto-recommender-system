"""Synthetic retrieval/features/audit/report replay. SQLite adapter, not real Parquet."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import numpy as np
from neighbors import group_history,build_query,NAMES
from history_io import ids_table,posting_sql,history_sql,collect_rows
from metrics import audit_arrays,censored_truth
from core import save_json,save_arrays,sha
from report import render

class PipelineTests(unittest.TestCase):
    def test_extraction_to_eight_chart_report_and_replay(self):
        con=sqlite3.connect(':memory:');self.addCleanup(con.close)
        con.execute('CREATE TABLE events(session BIGINT,aid BIGINT,ts BIGINT,event_type BIGINT,event_index BIGINT)')
        raw=[(1,10,1,0,0),(1,20,2,1,1),(1,900,3,2,8),(2,10,4,0,0),(2,800,5,1,3),(3,99,5,0,0)]
        con.executemany('INSERT INTO events VALUES (?,?,?,?,?)',raw)
        ids_table(con,'wanted','anchor',[10,20]);ids_table(con,'excluded','session',[99])
        post=collect_rows(con.execute(posting_sql(10)),4,65536);selected=np.unique(post[:,1]);ids_table(con,'selected','session',selected)
        source=collect_rows(con.execute(history_sql(10)),5,2000000);hist=group_history(source,10,{99})
        candidates=np.arange(400)[None];built=build_query(np.array([20,10]),candidates[0],list(map(int,selected)),hist)
        x=built['session_context']['features'][None];y=built['last_anchor']['features'][None]
        px=built['session_context']['proposal_ids'][None];py=built['last_anchor']['proposal_ids'][None]
        truth=censored_truth([100],[11],[0],[[1,1,1]],{100:[(10,0,12,1),(800,1,13,2),(900,2,14,3)]},20,18)
        result,stats=audit_arrays(candidates,x,y,px,py,truth)
        self.assertEqual(result['arms']['session_context']['new_distinct_target_hits'],[0,1,1])
        result.update(status='ROUND07_AUDIT_READY',decision='SYNTHETIC_FIXTURE_ONLY',limitations=['Synthetic fixture, not project evidence.'])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);save_json(out/'result.json',result)
            save_json(out/'diagnostics.json',[{'name':n,'support_fraction':float((x[:,:,j]>0).mean()),'max_abs_corr_seven_cached_signals':0} for j,n in enumerate(NAMES)])
            save_json(out/'neighbor_diagnostics.json',{'neighbor_counts':[[2,2]],'query_anchor_counts':[2]})
            save_arrays(out/'audit_statistics.npz',ids=np.array([100]),**stats)
            self.assertEqual(render(out)['charts'],8)
            before=sha(out/'round07_report.html');self.assertEqual(render(out)['charts'],8);self.assertEqual(sha(out/'round07_report.html'),before)
            receipt=json.loads((out/'report_receipt.json').read_text());self.assertEqual(len(receipt['plot_hashes']),8)
            self.assertIn('Zero ranker fits',(out/'round07_report.html').read_text())
            result['arms']['session_context']['expanded_candidate_oracle']['hits'][2]+=1
            (out/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):render(out)
