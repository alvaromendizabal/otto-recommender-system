import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import launch

class NotebookIntegrity(unittest.TestCase):
 def setup_files(self,root,source='print(1)'):
  n={'cells':[{'cell_type':'code','source':source,'execution_count':None,'outputs':[]}], 'metadata':{},'nbformat':4,'nbformat_minor':5}
  (root/'n.ipynb').write_text(json.dumps(n));sig={'n.ipynb':[{'cell_type':'code','source':source}]}
  (root/'notebook_sources.json').write_text(json.dumps(sig))
  (root/'MANIFEST.sha256').write_text('\n'.join(hashlib.sha256((root/x).read_bytes()).hexdigest()+'  '+x for x in ['n.ipynb','notebook_sources.json'])+'\n')
  return n
 def test_execution_outputs_allowed(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);n=self.setup_files(p);n['cells'][0]['execution_count']=1;n['cells'][0]['outputs']=[{'output_type':'stream','name':'stdout','text':'1'}];(p/'n.ipynb').write_text(json.dumps(n))
   with mock.patch.object(launch,'ROOT',p):launch.package_check()
 def test_notebook_source_change_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);n=self.setup_files(p);n['cells'][0]['source']='changed';(p/'n.ipynb').write_text(json.dumps(n))
   with mock.patch.object(launch,'ROOT',p),self.assertRaisesRegex(ValueError,'Notebook source changed'):launch.package_check()
 def test_registry_change_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.setup_files(p);(p/'notebook_sources.json').write_text('{}')
   with mock.patch.object(launch,'ROOT',p),self.assertRaises((ValueError,KeyError)):launch.package_check()
 def test_all_notebook_code_compiles(self):
  for path in Path(__file__).resolve().parents[1].glob('*.ipynb'):
   for i,c in enumerate(json.loads(path.read_text())['cells']):
    if c['cell_type']=='code':compile(''.join(c['source']) if isinstance(c['source'],list) else c['source'],str(path)+':'+str(i),'exec')
 def test_catalog_has_equal_width(self):
  root=Path(__file__).resolve().parents[1];catalog=json.loads((root/'feature_catalog.json').read_text())['features']
  self.assertEqual(len(catalog),48);self.assertEqual(sum(x['arm']=='primary' for x in catalog),24);self.assertEqual(len({x['name'] for x in catalog}),48)
 def test_both_hypotheses_documented(self):
  root=Path(__file__).resolve().parents[1];self.assertIn('Historical source', (root/'RESEARCH_PLAN.md').read_text())
