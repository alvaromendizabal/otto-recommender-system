"""Prepared regression checks; no models or real project data."""
import hashlib, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from legacy_imports import load_replication

class LegacyImportRegression(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source={'core.py':'VALUE=17\n',
                     'intent_features.py':'from core import VALUE\nNAMES=(VALUE,)\n',
                     'replication.py':'from intent_features import NAMES\ndef paths_for(home): return NAMES\ndef labels_for(*args): return []\n'}
        self.expected={}
        for name,value in self.source.items():
            (self.root/name).write_text(value)
            self.expected[name]=hashlib.sha256(value.encode()).hexdigest()
        self.patch=patch.dict(sys.modules);self.patch.start();self.addCleanup(self.patch.stop)
        for name in ('core','intent_features','round02_certified_helpers'):sys.modules.pop(name,None)
    def test_siblings_resolve_without_directory_on_sys_path(self):
        self.assertNotIn(str(self.root),sys.path)
        self.assertEqual(load_replication(self.root,self.expected).paths_for(None),(17,))
    def test_sys_path_is_not_mutated(self):
        before=list(sys.path);load_replication(self.root,self.expected);self.assertEqual(before,sys.path)
    def test_same_verified_module_is_reused(self):
        first=load_replication(self.root,self.expected)
        self.assertIs(first,load_replication(self.root,self.expected))
    def test_missing_sibling_fails_before_import(self):
        (self.root/'intent_features.py').unlink()
        with self.assertRaises(ImportError):load_replication(self.root,self.expected)
    def test_changed_sibling_fails(self):
        (self.root/'intent_features.py').write_text('NAMES=()')
        with self.assertRaises(ImportError):load_replication(self.root,self.expected)
    def test_import_failure_removes_partial_modules(self):
        text='raise RuntimeError("fixture failure")\n';(self.root/'replication.py').write_text(text)
        self.expected['replication.py']=hashlib.sha256(text.encode()).hexdigest()
        with self.assertRaises(RuntimeError):load_replication(self.root,self.expected)
        self.assertNotIn('intent_features',sys.modules)
    def test_shadowed_module_is_refused(self):
        import types
        sys.modules['intent_features']=types.ModuleType('intent_features')
        with self.assertRaises(ImportError):load_replication(self.root,self.expected)
