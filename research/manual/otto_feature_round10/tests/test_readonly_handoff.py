"""Exercise the verified handoff using a labeled, tiny read-only test adapter.

No private source is available in these tests. The production adapter still
loads the actual certified Round08 reader and verifies its own identity.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import shared_reuse as sr


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class ReadOnlyHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.home=Path(self.tmp.name)/'home';self.root=self.home/'otto_feature_round10'
        self.old=self.home/'otto_feature_round08';self.nine=self.home/'otto_feature_round09'
        self.root.mkdir(parents=True);self.old.mkdir()
        self.p=json.loads((sr.ROOT/'protocol.json').read_text());write(self.root/'protocol.json',self.p)
        oldp=copy.deepcopy(self.p);oldp['round']=8;write(self.old/'protocol.json',oldp)
        self.blob=self.old/'outputs/shared/fixture.bin'
        self.blob.parent.mkdir(parents=True);self.blob.write_bytes(b'CERTIFIED FIXTURE DATA')
        self.expected_blob=sha(self.blob)
        source='''# LABELED TEST ADAPTER, NOT THE PRIVATE SOURCE READER.
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
CALLS=[]
def context(home):
    CALLS.append('context')
    return {'shared':ROOT/'outputs/shared','protocol':{'round':8}}
def require(ctx):
    CALLS.append('require')
    actual=hashlib.sha256((ctx['shared']/'fixture.bin').read_bytes()).hexdigest()
    if actual != EXPECTED:raise ValueError('Cached binary changed')
    return {'fixture_only':True}
def evidence(ctx):
    require(ctx);CALLS.append('evidence');return ('read-only fixture',)
def prepare(*args):raise AssertionError('History rebuild forbidden')
def postings(*args):raise AssertionError('Posting scan forbidden')
def histories(*args):raise AssertionError('History scan forbidden')
'''.replace('EXPECTED',repr(self.expected_blob))
        (self.old/'shared_data.py').write_text(source)
        write(self.old/'outputs/result.json',{'status':'ROUND08_SCREEN_COMPLETED','gain':-0.0066})
        self.ref={'source_files':{n:sha(self.old/n) for n in ('shared_data.py','protocol.json')},
                  'outputs':{'result.json':sha(self.old/'outputs/result.json')}}
        write(self.root/'evidence/ROUND08_REUSE_CONTRACT.json',self.ref)
        write(self.nine/'outputs/result.json',{'status':'ROUND09_SCREEN_COMPLETED','gain':-0.02})
        write(self.nine/'outputs/report_receipt.json',{'status':'ROUND09_REPORT_READY',
            'result_sha256':sha(self.nine/'outputs/result.json')})
        self.patch=mock.patch.object(sr,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)

    def test_negative_round09_allows_independent_round10(self):
        ctx=sr.context(self.home)
        self.assertEqual(ctx['protocol']['round'],10)
        self.assertEqual(sr.evidence(ctx),('read-only fixture',))
        self.assertEqual(sr.reader(ctx).ROOT,self.old)
        self.assertNotIn('histories',sr.reader(ctx).CALLS)

    def test_handoff_does_not_mutate_prior_bytes(self):
        before={str(p):p.read_bytes() for p in self.home.rglob('*') if p.is_file()}
        sr.evidence(sr.context(self.home))
        after={str(p):p.read_bytes() for p in self.home.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        self.assertEqual(before,after)

    def test_missing_round09_result_stops(self):
        (self.nine/'outputs/result.json').unlink()
        with self.assertRaisesRegex(ValueError,'Finish pending Round09'):sr.context(self.home)

    def test_wrong_round09_status_stops(self):
        write(self.nine/'outputs/result.json',{'status':'PAUSED_CHECKPOINTED'})
        with self.assertRaisesRegex(ValueError,'not completed'):sr.context(self.home)

    def test_changed_round09_result_stops(self):
        write(self.nine/'outputs/result.json',{'status':'ROUND09_SCREEN_COMPLETED','gain':1.0})
        with self.assertRaises(ValueError):sr.context(self.home)

    def test_uncertified_reader_cannot_execute(self):
        (self.old/'shared_data.py').write_text("raise AssertionError('must not execute')")
        with self.assertRaises(ValueError):sr.context(self.home)

    def test_changed_round08_receipt_stops(self):
        write(self.old/'outputs/result.json',{'status':'fabricated'})
        with self.assertRaises(ValueError):sr.context(self.home)

    def test_changed_actual_binary_stops(self):
        self.blob.write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError,'Cached binary changed'):sr.context(self.home)

    def test_changed_ranker_is_refused(self):
        p=copy.deepcopy(self.p);p['rounds']=151;write(self.root/'protocol.json',p)
        with self.assertRaisesRegex(ValueError,'Frozen comparison changed'):sr.context(self.home)

    def test_unbound_reader_is_refused(self):
        with self.assertRaisesRegex(ValueError,'Missing verified'):sr.require({})
