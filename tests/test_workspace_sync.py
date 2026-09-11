"""Offline preservation, transfer, and Git synchronization regression tests."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
import unittest
import warnings
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "sync_otto_workspace.py"
SPEC = importlib.util.spec_from_file_location("otto_workspace_sync_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def make_zip(path: Path, data: dict[str, bytes] | None = None) -> Path:
    payload = data if data is not None else {"reports/result.json": b'{"passed": true}\n'}
    manifest = "".join(f"{sync.sha_bytes(value)}  {name}\n" for name, value in payload.items())
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in payload.items():
            archive.writestr(name, value)
        archive.writestr("MANIFEST.sha256", manifest)
    return path


class RemoteError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.created = 0
        self.gets = 0
        self.public_block = True
        self.fail_key: str | None = None

    def get_public_access_block(self, **_kwargs: Any) -> dict[str, Any]:
        return {"PublicAccessBlockConfiguration": {
            "BlockPublicAcls": self.public_block,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        }}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["ExpectedBucketOwner"] == sync.ACCOUNT
        key = kwargs["Key"]
        if key == self.fail_key:
            raise RemoteError("AccessDenied")
        if key in self.objects:
            raise RemoteError("PreconditionFailed")
        self.objects[key] = kwargs["Body"]
        self.created += 1
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["ExpectedBucketOwner"] == sync.ACCOUNT
        self.gets += 1
        value = self.objects[kwargs["Key"]]
        return {"Body": io.BytesIO(value), "ContentLength": len(value)}


class FakeSTS:
    def __init__(self, account: str | None = None) -> None:
        self.account = account if account is not None else sync.ACCOUNT

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": self.account}


def session(home: Path, mode: str = "archive", s3: FakeS3 | None = None) -> Any:
    instance = sync.Session.__new__(sync.Session)
    instance.home = home
    instance.output = home / "otto_workspace_sync"
    instance.output.mkdir(parents=True, exist_ok=True)
    instance.started = time.monotonic()
    instance.stage = "test"
    instance.calls = 0
    instance.transfer_bytes = 0
    instance.report = {"schema_version": 1, "mode": mode, "status": "STARTED", "files": []}
    instance.s3 = s3 if s3 is not None else FakeS3()
    instance.sts = FakeSTS()
    instance.done = threading.Event()
    instance.thread = threading.Thread(target=lambda: None)
    instance.thread.start()
    return instance


class WorkspaceSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_relative_path_accepts_nested_regular_file(self) -> None:
        self.assertEqual(str(sync.safe_relative("reports/a.json")), "reports/a.json")

    def test_relative_path_rejects_unsafe_variants(self) -> None:
        for name in ("../a", "/a", "a/../b", "a\\b", "a//b", "a/./b", ".", "", "a\x00b"):
            with self.subTest(name=name), self.assertRaises(sync.StopReview):
                sync.safe_relative(name)

    def test_preserve_write_reuses_identical_bytes(self) -> None:
        path = self.root / "evidence" / "a.txt"
        self.assertTrue(sync.preserve_write(path, b"original"))
        self.assertFalse(sync.preserve_write(path, b"original"))

    def test_preserve_write_rejects_different_bytes(self) -> None:
        path = self.root / "a.txt"
        path.write_bytes(b"original")
        with self.assertRaises(sync.StopReview):
            sync.preserve_write(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")

    def test_preserve_write_rejects_symlink(self) -> None:
        path = self.root / "a.txt"
        path.symlink_to(self.root / "elsewhere")
        with self.assertRaises(sync.StopReview):
            sync.preserve_write(path, b"bad")

    def test_manifest_and_replay(self) -> None:
        path = make_zip(self.root / "bundle.zip")
        self.assertEqual(len(sync.inspect_zip(path)), 1)
        first = sync.extract_verified(path, self.root / "extracted")
        second = sync.extract_verified(path, self.root / "extracted")
        self.assertEqual(first["created_files"], 2)
        self.assertEqual(second["created_files"], 0)

    def test_tampered_file_is_rejected(self) -> None:
        path = self.root / "bundle.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.txt", b"bad")
            archive.writestr("MANIFEST.sha256", f"{sync.sha_bytes(b'good')}  a.txt\n")
        with self.assertRaises(sync.StopReview):
            sync.inspect_zip(path)

    def test_unlisted_member_is_rejected(self) -> None:
        path = make_zip(self.root / "bundle.zip")
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("extra.txt", b"unknown")
        with self.assertRaises(sync.StopReview):
            sync.inspect_zip(path)

    def test_duplicate_member_is_rejected(self) -> None:
        path = make_zip(self.root / "bundle.zip", {"a.txt": b"a"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("a.txt", b"a")
        with self.assertRaises(sync.StopReview):
            sync.inspect_zip(path)

    def test_zip_symlink_is_rejected(self) -> None:
        path = self.root / "bundle.zip"
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(info, "destination")
            archive.writestr("MANIFEST.sha256", "")
        with self.assertRaises(sync.StopReview):
            sync.inspect_zip(path)

    def test_oversized_archive_is_rejected(self) -> None:
        path = make_zip(self.root / "bundle.zip")
        with patch.object(sync, "MAX_ARCHIVE_BYTES", 1), self.assertRaises(sync.StopReview):
            sync.inspect_zip(path)

    def test_expansion_limit_is_enforced(self) -> None:
        path = make_zip(self.root / "bundle.zip")
        with patch.object(sync, "MAX_EXPANDED_BYTES", 1), self.assertRaises(sync.StopReview):
            sync.inspect_zip(path)

    def test_extraction_prevalidates_all_existing_files(self) -> None:
        path = make_zip(self.root / "bundle.zip", {"first": b"a", "second": b"b"})
        target = self.root / "extract"
        target.mkdir()
        (target / "second").write_bytes(b"preserve me")
        with self.assertRaises(sync.StopReview):
            sync.extract_verified(path, target)
        self.assertFalse((target / "first").exists())
        self.assertEqual((target / "second").read_bytes(), b"preserve me")

    def test_extraction_rejects_symlink_parent(self) -> None:
        target = self.root / "target"
        target.mkdir()
        (target / "reports").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(sync.StopReview):
            sync.extract_verified(make_zip(self.root / "bundle.zip"), target)

    def test_only_expected_git_origins_accepted(self) -> None:
        self.assertTrue(sync.allowed_origin(sync.REPO_URL))
        self.assertTrue(sync.allowed_origin(sync.REPO_URL.removesuffix(".git")))
        self.assertFalse(sync.allowed_origin("https://github.com/other/project.git"))

    def test_restore_rejects_cloudshell(self) -> None:
        with self.assertRaises(sync.StopReview):
            sync.check_space(self.root / "cloudshell-user", sync.SPACE)

    def test_restore_rejects_wrong_space_confirmation(self) -> None:
        with self.assertRaises(sync.StopReview):
            sync.check_space(self.root, "nfl-dev")

    def test_restore_rejects_wrong_runtime_domain(self) -> None:
        with (
            patch.dict(os.environ, {"SAGEMAKER_DOMAIN_ID": "different"}),
            self.assertRaises(sync.StopReview),
        ):
            sync.check_space(self.root, sync.SPACE)

    def test_wrong_account_stops_before_transfer(self) -> None:
        worker = session(self.root)
        worker.sts = FakeSTS("wrong-account")
        with self.assertRaises(sync.StopReview):
            worker.identity()
        self.assertEqual(worker.s3.created, 0)

    def test_public_bucket_gate_stops_before_transfer(self) -> None:
        worker = session(self.root)
        worker.s3.public_block = False
        with self.assertRaises(sync.StopReview):
            worker.identity()
        self.assertEqual(worker.s3.created, 0)

    def test_remote_create_readback_and_reuse(self) -> None:
        worker = session(self.root)
        first = worker.put("test/key", b"verified bytes")
        second = worker.put("test/key", b"verified bytes")
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(worker.s3.created, 1)
        self.assertEqual(worker.s3.gets, 2)

    def test_conflicting_remote_key_preserved(self) -> None:
        worker = session(self.root)
        worker.s3.objects["test/key"] = b"existing"
        with self.assertRaises(sync.StopReview):
            worker.put("test/key", b"different")
        self.assertEqual(worker.s3.objects["test/key"], b"existing")

    def test_call_budget_is_enforced(self) -> None:
        worker = session(self.root)
        worker.calls = sync.MAX_CALLS
        with self.assertRaises(sync.StopReview):
            worker.identity()

    def test_transfer_budget_is_enforced(self) -> None:
        worker = session(self.root)
        with self.assertRaises(sync.StopReview):
            worker.transferred(sync.MAX_TRANSFER_BYTES + 1)

    def test_elapsed_budget_is_enforced(self) -> None:
        worker = session(self.root)
        worker.started -= sync.WORK_SECONDS + 1
        with self.assertRaises(sync.StopReview):
            worker.identity()

    def test_partial_archive_report_and_successful_replay(self) -> None:
        specs = []
        for index in range(3):
            path = make_zip(self.root / f"bundle{index}.zip")
            specs.append({"name": path.name, "folder": ".", "label": str(index),
                          "bytes": path.stat().st_size, "sha256": sync.sha_file(path)})
        remote = FakeS3()
        remote.fail_key = f"{sync.PREFIX}/bundles/{specs[1]['name']}"
        first = session(self.root, s3=remote)
        with patch.object(sync, "FILES", tuple(specs)):
            with self.assertRaises(RemoteError):
                first.archive()
            report = json.loads((first.output / "archive_report.json").read_text())
            self.assertEqual(len(report["files"]), 1)
            self.assertEqual(remote.created, 1)
            remote.fail_key = None
            second = session(self.root, s3=remote)
            second.archive()
            self.assertEqual(second.report["status"], "AWS_ARCHIVE_VERIFIED")
            self.assertEqual(remote.created, 5)
            third = session(self.root, s3=remote)
            third.archive()
            self.assertEqual(remote.created, 5)
            self.assertTrue(all(not item["created"] for item in third.report["files"]))

    def git_fixture(self) -> tuple[Path, Path]:
        remote = self.root / "remote.git"
        working = self.root / "seed"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        sync.git(None, "init", "-b", "main", str(working))
        sync.git(working, "config", "user.name", "Offline test")
        sync.git(working, "config", "user.email", "offline@example.invalid")
        (working / "scripts").mkdir()
        (working / "scripts/sync_otto_workspace.py").write_bytes(SOURCE.read_bytes())
        report = working / sync.PUBLIC_REPORT
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"source_bundle_sha256": sync.FILES[-1]["sha256"]}))
        sync.git(working, "add", ".")
        sync.git(working, "commit", "-m", "offline fixture")
        sync.git(working, "remote", "add", "origin", str(remote))
        sync.git(working, "push", "origin", "main")
        return remote, working

    def test_git_clone_and_replay_match_remote_main(self) -> None:
        remote, _working = self.git_fixture()
        home = self.root / "home"
        home.mkdir()
        worker = session(home, "restore")
        with patch.object(sync, "REPO_URL", str(remote)):
            repo = worker.synchronize_git()
            head = sync.git(repo, "rev-parse", "HEAD")
            worker.synchronize_git()
        self.assertEqual(head, worker.report["github"]["head"])
        self.assertEqual(sync.git(repo, "status", "--porcelain"), "")

    def test_git_dirty_checkout_is_preserved(self) -> None:
        remote, _working = self.git_fixture()
        home = self.root / "home"
        home.mkdir()
        worker = session(home, "restore")
        with patch.object(sync, "REPO_URL", str(remote)):
            repo = worker.synchronize_git()
            (repo / "local_work.txt").write_text("preserve me")
            with self.assertRaises(sync.StopReview):
                worker.synchronize_git()
            self.assertEqual((repo / "local_work.txt").read_text(), "preserve me")

    def test_git_rejects_unpublished_helper(self) -> None:
        remote, working = self.git_fixture()
        (working / "scripts/sync_otto_workspace.py").write_text("# different\n")
        sync.git(working, "add", ".")
        sync.git(working, "commit", "-m", "different helper")
        sync.git(working, "push", "origin", "main")
        home = self.root / "home"
        home.mkdir()
        with patch.object(sync, "REPO_URL", str(remote)), self.assertRaises(sync.StopReview):
            session(home, "restore").synchronize_git()

    def test_git_fast_forward_retains_published_addition(self) -> None:
        remote, working = self.git_fixture()
        home = self.root / "home"
        home.mkdir()
        worker = session(home, "restore")
        with patch.object(sync, "REPO_URL", str(remote)):
            repo = worker.synchronize_git()
            (working / "note.txt").write_text("new upstream evidence")
            sync.git(working, "add", ".")
            sync.git(working, "commit", "-m", "upstream change")
            sync.git(working, "push", "origin", "main")
            worker.synchronize_git()
        self.assertEqual((repo / "note.txt").read_text(), "new upstream evidence")

    def test_full_restore_and_replay_with_local_git_and_fake_s3(self) -> None:
        remote, _working = self.git_fixture()
        home = self.root / "home"
        home.mkdir()
        store = FakeS3()
        specs = []
        for index in range(3):
            archive = make_zip(self.root / f"artifact{index}.zip")
            specs.append({"name": archive.name, "folder": ".", "label": str(index),
                          "bytes": archive.stat().st_size, "sha256": sync.sha_file(archive)})
            store.objects[f"{sync.PREFIX}/bundles/{archive.name}"] = archive.read_bytes()
        # Publish the fixture's exact source-bundle identity before restoration.
        report = _working / sync.PUBLIC_REPORT
        report.write_text(json.dumps({"source_bundle_sha256": specs[-1]["sha256"]}))
        sync.git(_working, "add", ".")
        sync.git(_working, "commit", "-m", "fixture bundle identity")
        sync.git(_working, "push", "origin", "main")
        with (
            patch.object(sync, "REPO_URL", str(remote)),
            patch.object(sync, "FILES", tuple(specs)),
            patch.object(sync, "check_space", return_value={"identity_basis": "test fixture"}),
        ):
            first = session(home, "restore", store)
            first.restore(sync.SPACE)
            self.assertEqual(first.report["status"], "AWS_GITHUB_WORKSPACE_SYNCED")
            self.assertTrue(first.report["s3_sync_receipt"]["read_back_verified"])
            gets = store.gets
            second = session(home, "restore", store)
            second.restore(sync.SPACE)
            self.assertTrue(all(item["created_files"] == 0 for item in second.report["files"]))
            # Replay reads back only its new receipt, not the three existing bundles.
            self.assertEqual(store.gets - gets, 1)
            result_zip = second.finish()
            self.assertTrue(result_zip.is_file())

    def test_git_local_commit_is_not_discarded(self) -> None:
        remote, _working = self.git_fixture()
        home = self.root / "home"
        home.mkdir()
        worker = session(home, "restore")
        with patch.object(sync, "REPO_URL", str(remote)):
            repo = worker.synchronize_git()
            sync.git(repo, "config", "user.name", "Offline test")
            sync.git(repo, "config", "user.email", "offline@example.invalid")
            (repo / "mine.txt").write_text("keep local commit")
            sync.git(repo, "add", ".")
            sync.git(repo, "commit", "-m", "local-only")
            local_head = sync.git(repo, "rev-parse", "HEAD")
            with self.assertRaises(sync.StopReview):
                worker.synchronize_git()
        self.assertEqual(sync.git(repo, "rev-parse", "HEAD"), local_head)


if __name__ == "__main__":
    unittest.main()
