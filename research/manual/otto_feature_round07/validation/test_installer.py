"""Test the distributed installer and ZIP together; no cloud resources required.

Keep install_round07.py and otto_feature_round07.zip beside the extracted package.
Run: python -m unittest discover -s validation -v
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

BASE=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('installer_under_test',BASE/'install_round07.py')
INSTALLER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)
ARCHIVE=BASE/'otto_feature_round07.zip'

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.home=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def install(self):return INSTALLER.install(self.home,ARCHIVE)
    def test_01_archive_and_manifest(self):
        p=INSTALLER.verify_archive(ARCHIVE)
        self.assertIn('MANIFEST.sha256',p)
        self.assertIn('01_build_neighbor_evidence.ipynb',p)
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),INSTALLER.EXPECTED_SHA256)
    def test_02_install_preserves_previous_round(self):
        prior=self.home/'otto_feature_round06';prior.mkdir();(prior/'checkpoint').write_bytes(b'preserve me')
        root,new=self.install();self.assertTrue(new)
        self.assertEqual((prior/'checkpoint').read_bytes(),b'preserve me')
        self.assertIn('run tests once',INSTALLER.next_step(root))
    def test_03_identical_reinstall(self):
        root,_=self.install();before=(root/'research.py').stat().st_mtime_ns
        again,new=self.install();self.assertFalse(new);self.assertEqual(root,again)
        self.assertEqual((root/'research.py').stat().st_mtime_ns,before)
    def test_04_preserves_notebook_execution_and_outputs(self):
        root,_=self.install();p=root/'01_build_neighbor_evidence.ipynb';nb=json.loads(p.read_text())
        c=next(x for x in nb['cells'] if x['cell_type']=='code');c['execution_count']=3
        c['outputs']=[{'output_type':'stream','name':'stdout','text':'saved result\n'}]
        p.write_text(json.dumps(nb));(root/'outputs').mkdir();artifact=root/'outputs/checkpoint'
        artifact.write_bytes(b'keep');saved=p.read_bytes();self.install()
        self.assertEqual(saved,p.read_bytes());self.assertEqual(artifact.read_bytes(),b'keep')
    def test_05_refuses_changed_code_without_reset(self):
        root,_=self.install();p=root/'research.py';p.write_bytes(b'# work in progress\n')
        with self.assertRaisesRegex(ValueError,'changed'):self.install()
        self.assertEqual(p.read_bytes(),b'# work in progress\n')
    def test_06_refuses_changed_notebook_source(self):
        root,_=self.install();p=root/'01_build_neighbor_evidence.ipynb';nb=json.loads(p.read_text())
        nb['cells'][0]['source']='Changed research plan';p.write_text(json.dumps(nb));saved=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'source changed'):self.install()
        self.assertEqual(saved,p.read_bytes())
    def test_07_refuses_incomplete_existing_package(self):
        root,_=self.install();(root/'research.py').unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'):self.install()
        self.assertFalse((root/'research.py').exists())
    def test_08_refuses_symlink_destination(self):
        other=self.home/'unrelated';other.mkdir();(self.home/INSTALLER.PACKAGE).symlink_to(other,target_is_directory=True)
        with self.assertRaises(ValueError):self.install()
        self.assertEqual(list(other.iterdir()),[])
    def test_09_refuses_symlink_member(self):
        root,_=self.install();target=self.home/'external';target.write_bytes(b'preserve')
        p=root/'research.py';p.unlink();p.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'symlink'):self.install()
        self.assertEqual(target.read_bytes(),b'preserve')
    def test_10_archive_digest_tamper(self):
        bad=self.home/'tampered.zip';bad.write_bytes(ARCHIVE.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'ZIP checksum'):INSTALLER.install(self.home,bad)
        self.assertFalse((self.home/INSTALLER.PACKAGE).exists())
    def test_11_unsafe_zip_member_inner_gate(self):
        cases=[('../escape','normal'),('otto_feature_round07/../escape','normal'),
               ('/absolute/file','normal'),('otto_feature_round07\\escape','normal'),
               ('other/package.py','normal'),('otto_feature_round07/link','symlink')]
        for name,kind in cases:
            with self.subTest(name=name):
                p=self.home/'unsafe.zip'
                with zipfile.ZipFile(p,'w') as z:
                    i=zipfile.ZipInfo(name)
                    if kind=='symlink':i.external_attr=(stat.S_IFLNK|0o777)<<16
                    z.writestr(i,b'bad')
                with patch.object(INSTALLER,'EXPECTED_SHA256',hashlib.sha256(p.read_bytes()).hexdigest()):
                    with self.assertRaisesRegex(ValueError,'Unsafe ZIP'):INSTALLER.verify_archive(p)
    def test_12_inner_manifest_tamper(self):
        p=self.home/'bad-manifest.zip'
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('otto_feature_round07/a.py',b'a')
            z.writestr('otto_feature_round07/MANIFEST.sha256','0'*64+'  a.py\n')
        with patch.object(INSTALLER,'EXPECTED_SHA256',hashlib.sha256(p.read_bytes()).hexdigest()):
            with self.assertRaisesRegex(ValueError,'Inner checksum'):INSTALLER.verify_archive(p)
    def test_13_prior_stop_blocks_retry_instruction(self):
        root,_=self.install();p=root/'outputs/runs';p.mkdir(parents=True)
        (p/'launcher-1.json').write_text(json.dumps({'phase':'postings','exit_code':2,'reason':'pause'}))
        self.assertIn('PRIOR_STOP_DETECTED',INSTALLER.next_step(root))
    def test_14_completed_audit_no_rerun(self):
        root,_=self.install();p=root/'outputs';p.mkdir()
        (p/'result.json').write_text(json.dumps({'status':'ROUND07_AUDIT_READY'}))
        (p/'report_receipt.json').write_text('{}')
        self.assertIn('ROUND07_ALREADY_COMPLETED',INSTALLER.next_step(root))
    def test_15_phase_receipts_choose_next_stage(self):
        root,_=self.install();p=root/'outputs/runs';p.mkdir(parents=True)
        (p/'launcher-1.json').write_text(json.dumps({'phase':'prepare','exit_code':0,'reason':None}))
        self.assertIn('postings cell next',INSTALLER.next_step(root))

if __name__=='__main__':unittest.main()
