"""No processes/cloud calls: bounds, fail-closed sequencing and return-bundle tests."""
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest import mock
import launch

class Launcher(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.patch=mock.patch.object(launch,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        (self.root/'outputs/runs').mkdir(parents=True)
    def put(self,rel,value):
        p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value);return p
    def test_caps_are_bounded(self):self.assertEqual(launch.CAPS,{'tests':60,'features':200,'screen':260,'report':60})
    def test_no_unattended_all_stage(self):
        with self.assertRaises(ValueError):launch._run_stage('all')
    def test_prior_failure_blocks_even_after_later_report_success(self):
        self.put('outputs/runs/launcher-features-1.json',json.dumps({'exit_code':2}))
        self.put('outputs/runs/launcher-report-2.json',json.dumps({'exit_code':0}))
        with mock.patch.object(launch,'package_check'),self.assertRaisesRegex(RuntimeError,'PRIOR_STOP_DETECTED'):launch._run_stage('features')
    def test_features_require_successful_tests(self):
        with mock.patch.object(launch,'package_check'),self.assertRaisesRegex(RuntimeError,'tests stage'):launch._run_stage('features')
    def test_screen_requires_features(self):
        with mock.patch.object(launch,'package_check'),self.assertRaisesRegex(RuntimeError,'feature stage'):launch._run_stage('screen')
    def test_bundle_excludes_large_binaries(self):
        self.put('outputs/models/model.txt','PRIVATE');self.put('outputs/features/chunk.npz','ARRAY');self.put('outputs/index/counts.npz','INDEX')
        self.put('outputs/result.json','{}');self.put('review/01.json','{}')
        with zipfile.ZipFile(launch.collect()) as z:
            self.assertIn('outputs/result.json',z.namelist());self.assertIn('review/01.json',z.namelist())
            self.assertFalse(any('/models/' in n or '/features/' in n or '/index/' in n for n in z.namelist()))
    def test_bundle_manifest_hashes(self):
        self.put('outputs/run.log','actual log')
        with zipfile.ZipFile(launch.collect()) as z:
            for row in json.loads(z.read('RETURN_MANIFEST.json')):
                b=z.read(row['path']);self.assertEqual(len(b),row['bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),row['sha256'])
    def test_bundle_preserves_external_symlink_target(self):
        external=self.put('other.txt','outside');(self.root/'outputs/linked.log').symlink_to(external)
        with zipfile.ZipFile(launch.collect()) as z:self.assertNotIn('outputs/linked.log',z.namelist())
        self.assertEqual(external.read_text(),'outside')
    def test_checksum_failure_stops(self):
        self.put('worker.py','changed');self.put('MANIFEST.sha256',hashlib.sha256(b'expected').hexdigest()+'  worker.py\n')
        with self.assertRaisesRegex(ValueError,'Package file changed'):launch.package_check()
    def test_missing_package_file_stops(self):
        self.put('MANIFEST.sha256',hashlib.sha256(b'x').hexdigest()+'  missing.py\n')
        with self.assertRaises(FileNotFoundError):launch.package_check()
