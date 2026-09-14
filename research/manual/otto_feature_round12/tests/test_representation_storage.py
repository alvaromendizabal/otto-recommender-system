"""Synthetic, USER-RUN preparation/factor checkpoint tests. No private data."""
import json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from core import save_arrays,save_json,sha
import latent_models as lm
import experiment as e
class RepresentationCheckpoints(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.base=Path(self.temp.name);self.root=self.base/'code';self.root.mkdir();self.out=self.base/'outputs';self.out.mkdir();shared=self.base/'shared';shared.mkdir()
  for name in ('protocol.json','latent_models.py','latent_features.py'):(self.root/name).write_bytes((e.ROOT/name).read_bytes())
  ids=np.arange(64,dtype=np.int64)+9000;q=lm.HISTORY_END+3600000+np.arange(64,dtype=np.int64)*3500000
  anchors=np.full((64,4),-1,np.int64);anchors[:,0]=np.arange(64)+100
  postings=np.column_stack((np.arange(64)+100,np.arange(64)+10000,np.full(64,lm.HISTORY_END-5000),np.ones(64))).astype(np.int64)
  events=[]
  for i in range(64):
   # Unique local sequence with all destination action contexts represented.
   for j,(aid,kind) in enumerate(((100+i,0),(200+i,0),(300+i,1),(400+i,2))):
    events.append((10000+i,aid,lm.HISTORY_END-5000+j*1000,kind,j))
  h=save_arrays(shared/'history.npz',events=np.array(events,np.int64));save_json(shared/'history.json',{'sha256':h})
  h=save_arrays(shared/'cohort.npz',ids=ids,anchors=anchors);save_json(shared/'cohort.json',{'sha256':h})
  h=save_arrays(shared/'postings.npz',postings=postings);save_json(shared/'postings.json',{'sha256':h})
  self.ctx={'shared':shared,'excluded':np.array([],np.int64),'ids':ids,'cohort':{'query_ts':q.tolist()}}
 def prepare(self):return lm.prepare(self.ctx,self.root,self.out,e.ROUND)
 def test_prepare_reuses_manifest(self):
  self.prepare()
  with mock.patch.object(lm,'source_events',side_effect=AssertionError('Unexpected source reread')):
   result=self.prepare()
  self.assertTrue(result['reused'])
 def test_factor_units_reuse_without_refit(self):
  self.prepare();lm.learn(self.ctx,self.root,self.out,e.ROUND)
  with mock.patch.object(lm,'svd',side_effect=AssertionError('Unexpected factor refit')):
   result=lm.learn(self.ctx,self.root,self.out,e.ROUND)
  self.assertEqual(result['verified_units'],2 if e.ROUND==11 else 6)
 def test_corrupt_input_is_rejected(self):
  self.prepare();p=self.out/'representation_inputs/vocabulary.npz';p.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.prepare()
 def test_representation_code_change_is_rejected(self):
  self.prepare();lm.learn(self.ctx,self.root,self.out,e.ROUND)
  with (self.root/'latent_features.py').open('a') as f:f.write('\n# changed\n')
  with self.assertRaises(ValueError):lm.load_embeddings(self.root,self.out,e.ROUND)
 def test_orphan_factor_is_preserved_and_stops(self):
  self.prepare();d=self.out/'representations';d.mkdir();p=d/('primary.npz' if e.ROUND==11 else 'primary-0.npz');p.write_bytes(b'orphan')
  with self.assertRaises(ValueError):lm.learn(self.ctx,self.root,self.out,e.ROUND)
  self.assertEqual(p.read_bytes(),b'orphan')
 def test_resume_after_planned_boundary_keeps_units(self):
  self.prepare();calls=0
  def stop_at_second_unit():
   nonlocal calls
   calls+=1
   if calls==2:raise RuntimeError('fixture interruption')
  with self.assertRaises(RuntimeError):lm.learn(self.ctx,self.root,self.out,e.ROUND,before=stop_at_second_unit)
  first=self.out/'representations'/('primary.npz' if e.ROUND==11 else 'primary-0.npz');before=sha(first)
  result=lm.learn(self.ctx,self.root,self.out,e.ROUND)
  self.assertEqual(sha(first),before);self.assertEqual(result['ranker_fits'],0)
