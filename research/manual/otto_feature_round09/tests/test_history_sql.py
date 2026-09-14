import sqlite3
import unittest
import numpy as np
from history_io import ids_table,posting_sql,history_sql,collect_rows

class SQLTests(unittest.TestCase):
    def make(self,rows,wanted,excluded):
        con=sqlite3.connect(':memory:');self.addCleanup(con.close)
        con.execute('CREATE TABLE events(session BIGINT,aid BIGINT,ts BIGINT,event_type BIGINT,event_index BIGINT)')
        con.executemany('INSERT INTO events VALUES (?,?,?,?,?)',rows)
        ids_table(con,'wanted','anchor',wanted);ids_table(con,'excluded','session',excluded)
        return con
    def test_matches_independent_grouping_and_topk(self):
        for seed in range(20):
            rng=np.random.default_rng(seed);rows=[(int(rng.integers(1,10)),int(rng.integers(1,6)),int(rng.integers(1,15)),0,i) for i in range(100)]
            con=self.make(rows,[1,2,3],[2,4]);actual=collect_rows(con.execute(posting_sql(10,3)),4,100)
            groups={}
            for sid,aid,ts,_,_ in rows:
                if aid in (1,2,3) and sid not in (2,4) and 0<=ts<10:groups[(aid,sid)]=max(ts,groups.get((aid,sid),-1))
            expected=[]
            for a in (1,2,3):
                vals=[(sid,t) for (aa,sid),t in groups.items() if aa==a];vals.sort(key=lambda x:(-x[1],x[0]))
                expected.extend([[a,s,t,len(vals)] for s,t in vals[:3]])
            expected.sort(key=lambda r:(r[0],r[1]))
            self.assertEqual(actual.tolist(),expected)
    def test_strict_cutoff_and_tie(self):
        con=self.make([(2,1,9,0,0),(1,1,9,0,0),(3,1,10,0,0)],[1],[])
        self.assertEqual(collect_rows(con.execute(posting_sql(10,1)),4,10).tolist(),[[1,1,9,2]])
    def test_repeat_not_double_counted(self):
        con=self.make([(1,1,1,0,0),(1,1,2,1,1)],[1],[])
        self.assertEqual(collect_rows(con.execute(posting_sql(10)),4,10).tolist(),[[1,1,2,1]])
    def test_history_keeps_nonanchor_products(self):
        con=self.make([(1,1,1,0,0),(1,8,3,2,5),(2,2,2,0,0)],[1],[]);ids_table(con,'selected','session',[1])
        self.assertEqual(collect_rows(con.execute(history_sql(10)),5,10).tolist(),[[1,1,1,0,0],[1,8,3,2,5]])
    def test_history_exclusion(self):
        con=self.make([(1,1,1,0,0)],[1],[1]);ids_table(con,'selected','session',[1])
        self.assertEqual(collect_rows(con.execute(history_sql(10)),5,10).shape,(0,5))
    def test_row_cap_stops_no_trimming(self):
        con=self.make([(1,1,1,0,0),(2,1,2,0,0)],[1],[])
        with self.assertRaises(ValueError):collect_rows(con.execute(posting_sql(10)),4,1)
    def test_unsafe_identifier(self):
        con=self.make([],[],[])
        with self.assertRaises(ValueError):ids_table(con,'evil','x',[1])
    def test_duplicate_identifiers(self):
        con=self.make([],[],[])
        with self.assertRaises(ValueError):ids_table(con,'selected','session',[1,1])
    def test_invalid_bound(self):
        with self.assertRaises(ValueError):posting_sql(10,65)
    def test_null_not_silently_cast(self):
        con=self.make([(1,None,1,0,0)],[],[])
        with self.assertRaises(ValueError):collect_rows(con.execute('SELECT * FROM events'),5,10)
