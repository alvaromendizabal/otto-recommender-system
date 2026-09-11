"""Owner-run OTTO archive and workspace synchronization; never trains or pushes Git.

Run ``archive`` in Oregon CloudShell, publish this package through a reviewed
GitHub pull request, then run ``restore --confirm-space otto-dev`` in that space.
Only three hash-pinned result ZIPs are archived. No recursive home upload exists.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ACCOUNT = "560403859723"
REGION = "us-west-2"
BUCKET = "otto-recsys-560403859723-us-west-2"
DOMAIN = "d-njhxv1erusdc"
SPACE = "otto-dev"
REPO_URL = "https://github.com/alvaromendizabal/otto-recommender-system.git"
PREFIX = "manual/workspace-sync/af39235b19df"
PUBLIC_REPORT = "reports/research/shared_feature_confirmation_preflight.json"
MAX_ARCHIVE_BYTES = 64 * 1024**2
MAX_EXPANDED_BYTES = 128 * 1024**2
MAX_CALLS = 64
MAX_TRANSFER_BYTES = 128 * 1024**2
WORK_SECONDS = 280
FILES: tuple[dict[str, Any], ...] = (
    {
        "name": "otto_recovery_bundle.zip",
        "folder": "otto_manual_recovery",
        "label": "recovery",
        "bytes": 6731874,
        "sha256": "ee9ed3727c0eca4e96705636c687a01a6703c03e768f3dc454c2a1ed2a244c69",
    },
    {
        "name": "otto_confirmation_preflight_bundle.zip",
        "folder": "otto_confirmation_preflight",
        "label": "preflight",
        "bytes": 11418845,
        "sha256": "df42a2a5460d094c0fe042b59521b9a75be824843db1e2c7d91f84c381eb1856",
    },
    {
        "name": "otto_shared_feature_smoke_inputs_bundle.zip",
        "folder": "otto_shared_feature_smoke_inputs",
        "label": "smoke_inputs",
        "bytes": 27322879,
        "sha256": "af39235b19dfd9226268e0236ce52b8cd4bc565f7fc0d1925a022b74349a9ae7",
    },
)


class StopReview(RuntimeError):
    """A failed integrity, identity, preservation, or resource gate."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_archive_keys(digest: str) -> tuple[str, str]:
    """Version helper bytes without overwriting old source or duplicating evidence."""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise StopReview("Source identity must be a lowercase SHA-256 digest")
    root = f"{PREFIX}/source/{digest}"
    return f"{root}/sync_otto_workspace.py", f"{root}/manifest.json"


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or "\x00" in name
        or str(path) != name
        or path == PurePosixPath(".")
    ):
        raise StopReview(f"Unsafe archive member: {name!r}")
    return path


def safe_destination(root: Path, relative: str) -> Path:
    parts = safe_relative(relative).parts
    if root.is_symlink():
        raise StopReview("Destination root is a symlink")
    destination = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise StopReview(f"Destination contains a symlink: {current}")
    if not destination.resolve().is_relative_to(root.resolve()):
        raise StopReview("Destination escapes its root")
    return destination


def preserve_write(path: Path, data: bytes) -> bool:
    """Create a regular file or reuse identical bytes; never overwrite differences."""
    if path.is_symlink():
        raise StopReview(f"Refusing symlink: {path}")
    if path.exists():
        if not path.is_file() or sha_file(path) != sha_bytes(data):
            raise StopReview(f"Existing file differs; preserved untouched: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically and exclusively. Only this function's temporary file is removed.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def inspect_zip(path: Path) -> dict[str, str]:
    """Verify every listed file, membership, CRCs, and extraction safety."""
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise StopReview("Archive exceeds the size limit")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) > 5000:
            raise StopReview("Duplicate archive entries or excessive member count")
        if sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
            raise StopReview("Archive expansion exceeds limit")
        for entry in entries:
            safe_relative(entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                raise StopReview("Only regular archive files are accepted")
        if "MANIFEST.sha256" not in names:
            raise StopReview("Archive has no checksum manifest")
        expected: dict[str, str] = {}
        for line in archive.read("MANIFEST.sha256").decode().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.strip()
            safe_relative(name)
            if name in expected or len(digest) != 64:
                raise StopReview("Invalid checksum manifest")
            expected[name] = digest
        if set(names) != set(expected) | {"MANIFEST.sha256"}:
            raise StopReview("Archive membership differs from checksum manifest")
        for name, wanted in expected.items():
            if sha_bytes(archive.read(name)) != wanted:
                raise StopReview(f"Archive checksum mismatch: {name}")
        return expected


def extract_verified(path: Path, destination: Path) -> dict[str, int]:
    manifest = inspect_zip(path)
    created = 0
    with zipfile.ZipFile(path) as archive:
        # Validate ALL destinations before changing any file.
        targets = {
            name: safe_destination(destination, name) for name in archive.namelist()
        }
        for name, target in targets.items():
            if target.exists() and sha_file(target) != sha_bytes(archive.read(name)):
                raise StopReview(f"Existing extracted evidence differs: {target}")
        for name, target in targets.items():
            created += int(preserve_write(target, archive.read(name)))
    return {"manifest_files_verified": len(manifest), "created_files": created}


def find_bundle(home: Path, spec: dict[str, Any]) -> Path:
    candidates = [home / spec["folder"] / spec["name"], home / spec["name"]]
    matching = []
    for path in candidates:
        if (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == spec["bytes"]
            and sha_file(path) == spec["sha256"]
        ):
            matching.append(path)
    if not matching:
        raise StopReview(f"Missing or changed reviewed ZIP: {candidates[0]}")
    return matching[0]


def allowed_origin(origin: str) -> bool:
    return origin.strip() in {
        REPO_URL,
        REPO_URL.removesuffix(".git"),
        "git@github.com:alvaromendizabal/otto-recommender-system.git",
        "ssh://git@github.com/alvaromendizabal/otto-recommender-system.git",
    }


def git(repo: Path | None, *args: str, timeout: int = 75) -> str:
    prefix = ["-C", str(repo)] if repo is not None else []
    command = ["git", *prefix, *args]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    if result.returncode:
        raise StopReview(f"Git command failed ({args[0]}): {result.stderr[-2000:]}")
    return result.stdout.strip()


def find_repository(home: Path) -> Path:
    candidates: set[Path] = set()
    # Bounded search, no recursive scan of project data or virtual environments.
    parents = [home, home / "projects", home / "SageMaker", home / "workspaces"]
    for parent in parents:
        if not parent.is_dir() or parent.is_symlink():
            continue
        for path in parent.iterdir():
            if "otto" not in path.name.lower() or not path.is_dir() or path.is_symlink():
                continue
            if (path / ".git").exists():
                origin = git(path, "remote", "get-url", "origin", timeout=10)
                if allowed_origin(origin):
                    candidates.add(path.resolve())
    if len(candidates) > 1:
        raise StopReview(f"Multiple OTTO checkouts; preserve and review: {sorted(candidates)}")
    return next(iter(candidates)) if candidates else home / "otto-recommender-system"


def check_space(home: Path, confirmation: str) -> dict[str, Any]:
    if confirmation != SPACE or home.name == "cloudshell-user":
        raise StopReview("Restore must run in the otto-dev SageMaker terminal, not CloudShell")
    metadata_path = Path("/opt/ml/metadata/resource-metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    domain = metadata.get("DomainId") or os.environ.get("SAGEMAKER_DOMAIN_ID")
    space = metadata.get("SpaceName") or os.environ.get("SAGEMAKER_SPACE_NAME")
    if domain and domain != DOMAIN:
        raise StopReview("This terminal belongs to a different SageMaker domain")
    if space and space != SPACE:
        raise StopReview("This terminal belongs to a different SageMaker space")
    arn = str(metadata.get("ResourceArn", ""))
    if arn and (f":{ACCOUNT}:" not in arn or DOMAIN not in arn):
        raise StopReview("SageMaker resource ARN does not match this OTTO account/domain")
    if arn and f":app/{DOMAIN}/" in arn and f":app/{DOMAIN}/{SPACE}/" not in arn:
        raise StopReview("SageMaker app ARN identifies a different space")
    return {
        "expected_space": SPACE,
        "expected_domain": DOMAIN,
        "runtime_domain": domain,
        "runtime_space": space,
        "identity_basis": "runtime metadata" if domain and space else "owner confirmation",
        "home": str(home),
        "free_home_bytes": shutil.disk_usage(home).free,
        "cpu_count": os.cpu_count(),
    }


class Session:
    """Bounded AWS operations, heartbeat, and per-file verification receipts."""

    def __init__(self, home: Path, mode: str) -> None:
        self.home = home
        self.output = home / "otto_workspace_sync"
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.stage = "initializing"
        self.calls = 0
        self.transfer_bytes = 0
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "mode": mode,
            "started_at_utc": utc(),
            "status": "STARTED",
            "model_fits": 0,
            "jobs_launched": 0,
            "github_writes_by_script": 0,
            "scope": "three reviewed bundles, their source helper, and one GitHub checkout",
            "files": [],
        }
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        boto3 = importlib.import_module("boto3")
        config_module = importlib.import_module("botocore.config")
        config = config_module.Config(
            connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 1}
        )
        self.s3 = boto3.client("s3", region_name=REGION, config=config)
        self.sts = boto3.client("sts", region_name=REGION, config=config)

    def log(self, message: str) -> None:
        line = f"{utc()} {message}"
        print(line, flush=True)
        with (self.output / "sync.jsonl").open("a") as output:
            output.write(json.dumps({"utc": utc(), "message": message}) + "\n")

    def heartbeat(self) -> None:
        while not self.done.wait(15):
            self.log(
                f"HEARTBEAT stage={self.stage} elapsed={time.monotonic() - self.started:.1f}s "
                f"aws_calls={self.calls} transferred_bytes={self.transfer_bytes}"
            )

    def call(self, client: Any, operation: str, **params: Any) -> Any:
        if self.calls >= MAX_CALLS or time.monotonic() - self.started > WORK_SECONDS:
            raise StopReview("Bounded operation or time limit reached")
        self.calls += 1
        return getattr(client, operation)(**params)

    def transferred(self, count: int) -> None:
        self.transfer_bytes += count
        if self.transfer_bytes > MAX_TRANSFER_BYTES:
            raise StopReview("Transfer limit reached")

    def identity(self) -> None:
        self.stage = "identity_and_private_bucket"
        identity = self.call(self.sts, "get_caller_identity")
        if identity.get("Account") != ACCOUNT:
            raise StopReview("Wrong AWS account; no upload or restore was attempted")
        block = self.call(
            self.s3, "get_public_access_block", Bucket=BUCKET, ExpectedBucketOwner=ACCOUNT
        )["PublicAccessBlockConfiguration"]
        required = ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy",
                    "RestrictPublicBuckets")
        if not all(block.get(key) is True for key in required):
            raise StopReview("Required bucket public-access protections are not verified")
        self.report["account"] = ACCOUNT
        self.report["bucket_public_access_block"] = block

    def download(self, key: str, wanted: str, size: int) -> bytes:
        response = self.call(
            self.s3, "get_object", Bucket=BUCKET, Key=key, ExpectedBucketOwner=ACCOUNT
        )
        body = response["Body"]
        try:
            if response["ContentLength"] != size or size > MAX_ARCHIVE_BYTES:
                raise StopReview("Remote object size differs")
            chunks = []
            count = 0
            while True:
                chunk = body.read(1024**2)
                if not chunk:
                    break
                self.transferred(len(chunk))
                count += len(chunk)
                if count > size:
                    raise StopReview("Remote object exceeded declared size")
                chunks.append(chunk)
            data = b"".join(chunks)
        finally:
            body.close()
        if len(data) != size or sha_bytes(data) != wanted:
            raise StopReview("Remote bytes do not match the reviewed checksum")
        return data

    def put(self, key: str, data: bytes) -> dict[str, Any]:
        if len(data) > MAX_ARCHIVE_BYTES:
            raise StopReview("Upload exceeds limit")
        wanted = sha_bytes(data)
        created = False
        try:
            self.transferred(len(data))
            self.call(
                self.s3,
                "put_object",
                Bucket=BUCKET,
                Key=key,
                Body=data,
                ExpectedBucketOwner=ACCOUNT,
                IfNoneMatch="*",
                ChecksumSHA256=base64.b64encode(bytes.fromhex(wanted)).decode(),
                Metadata={"sha256": wanted},
            )
            created = True
        except Exception as error:
            response = getattr(error, "response", {})
            if response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
                raise
        # Independent full-byte read-back, including when reusing a remote key.
        self.download(key, wanted, len(data))
        return {"key": key, "bytes": len(data), "sha256": wanted,
                "created": created, "read_back_verified": True}

    def archive(self) -> None:
        self.identity()
        self.stage = "verify_local_bundles"
        paths = [(spec, find_bundle(self.home, spec)) for spec in FILES]
        for spec, path in paths:
            inspect_zip(path)
            self.log(f"LOCAL_VERIFIED {spec['name']}")
        self.stage = "archive_and_read_back"
        for spec, path in paths:
            result = self.put(f"{PREFIX}/bundles/{spec['name']}", path.read_bytes())
            self.report["files"].append(result)
            self.save_report()
            self.log(f"S3_VERIFIED {spec['name']}")
        source = Path(__file__).resolve().read_bytes()
        source_digest = sha_bytes(source)
        source_key, manifest_key = source_archive_keys(source_digest)
        self.report["source"] = self.put(source_key, source)
        manifest = {"schema_version": 1, "bundles": list(FILES),
                    "source_sha256": source_digest, "source_key": source_key}
        self.report["manifest"] = self.put(manifest_key, json_bytes(manifest))
        self.log(f"SOURCE_VERIFIED sha256={source_digest} key={source_key}")
        self.report["status"] = "AWS_ARCHIVE_VERIFIED"

    def synchronize_git(self) -> Path:
        self.stage = "synchronize_github_main"
        repo = find_repository(self.home)
        if not (repo / ".git").exists():
            if repo.exists():
                raise StopReview(f"Destination exists but is not a Git checkout: {repo}")
            git(None, "clone", "--depth", "1", "--single-branch", "--branch", "main",
                REPO_URL, str(repo))
        if not allowed_origin(git(repo, "remote", "get-url", "origin")):
            raise StopReview("Unexpected Git origin; no checkout update attempted")
        branch = git(repo, "branch", "--show-current")
        dirty = git(repo, "status", "--porcelain", "--untracked-files=normal")
        self.report["checkout_before"] = {
            "path": str(repo), "branch": branch, "head": git(repo, "rev-parse", "HEAD"),
            "status": dirty,
        }
        if branch != "main" or dirty:
            raise StopReview("Existing checkout is not clean main; changes preserved for review")
        git(repo, "fetch", "--no-tags", "origin", "main")
        remote = git(repo, "rev-parse", "FETCH_HEAD")
        published = json.loads(git(repo, "show", f"{remote}:{PUBLIC_REPORT}"))
        if published.get("source_bundle_sha256") != FILES[-1]["sha256"]:
            raise StopReview("Merge the reviewed synchronization PR before restoring")
        # This source comparison ensures the reviewed helper is actually in GitHub main.
        published_script = git(repo, "show", f"{remote}:scripts/sync_otto_workspace.py")
        if published_script != Path(__file__).read_text().strip():
            raise StopReview("Published helper differs from this reviewed helper")
        git(repo, "merge", "--ff-only", remote)
        if git(repo, "rev-parse", "HEAD") != remote:
            raise StopReview("Local branch is ahead/diverged; no reset or force push performed")
        self.report["github"] = {
            "repository": REPO_URL, "branch": "main", "head": remote,
            "tree": git(repo, "rev-parse", "HEAD^{tree}"), "path": str(repo),
        }
        return repo

    def restore(self, confirmation: str) -> None:
        self.report["space"] = check_space(self.home, confirmation)
        if shutil.disk_usage(self.home).free < 1024**3:
            raise StopReview("Need at least 1 GiB of workspace space; no files deleted")
        self.identity()
        root = self.home / "otto-artifacts" / "manual-sync" / "af39235b19df"
        self.report["artifact_root"] = str(root)
        self.stage = "restore_reviewed_evidence"
        for spec in FILES:
            key = f"{PREFIX}/bundles/{spec['name']}"
            path = safe_destination(root, f"bundles/{spec['name']}")
            if path.exists():
                if sha_file(path) != spec["sha256"]:
                    raise StopReview(f"Existing local archive differs: {path}")
            else:
                preserve_write(path, self.download(key, spec["sha256"], spec["bytes"]))
            receipt = extract_verified(path, root / spec["label"])
            self.report["files"].append({**spec, **receipt, "key": key, "verified": True})
            self.save_report()
            self.log(f"WORKSPACE_VERIFIED {spec['name']}")
        repo = self.synchronize_git()
        latest = git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
        if latest != git(repo, "rev-parse", "HEAD") or git(repo, "status", "--porcelain"):
            raise StopReview("GitHub advanced or checkout changed during restore; review required")
        self.report["status"] = "AWS_GITHUB_WORKSPACE_SYNCED"
        self.report["limitations"] = [
            "Only the three reviewed bundles and current published repository are synchronized.",
            "Full history tables remain in their original S3 locations, not copied to this space.",
            "ML environment dependencies and graph-building feasibility are not certified.",
            "Early-window wide graphs remain uncertified; no feature or model run is permitted.",
            "AWS app stopping is a separate owner action; this script does not stop the app.",
        ]
        self.save_report()
        receipt_bytes: bytes = json_bytes(self.report)
        key = f"{PREFIX}/receipts/{sha_bytes(receipt_bytes)}.json"
        self.report["s3_sync_receipt"] = self.put(key, receipt_bytes)

    def save_report(self) -> None:
        self.report["aws_calls"] = self.calls
        self.report["transfer_bytes"] = self.transfer_bytes
        self.report["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        destination = self.output / f"{self.report['mode']}_report.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(json_bytes(self.report))
        temporary.replace(destination)

    def finish(self) -> Path:
        self.done.set()
        self.thread.join(timeout=2)
        self.report["finished_at_utc"] = utc()
        self.save_report()
        destination = self.output / "otto_workspace_sync_result.zip"
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.output.glob("*_report.json")):
                archive.write(path, path.name)
            if (self.output / "sync.jsonl").is_file():
                archive.write(self.output / "sync.jsonl", "sync.jsonl")
        return destination


def timeout_handler(_signum: int, _frame: Any) -> None:
    raise StopReview("280-second work limit reached; completed evidence is preserved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("archive", "restore"))
    parser.add_argument("--confirm-space", default="")
    arguments = parser.parse_args()
    session: Session | None = None
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(WORK_SECONDS)
        session = Session(Path.home(), arguments.mode)
        if arguments.mode == "archive":
            session.archive()
        else:
            session.restore(arguments.confirm_space)
    except (Exception, KeyboardInterrupt) as error:
        if session is None:
            print(f"STOPPED_REVIEW_REQUIRED: {error}", flush=True)
            return 1
        session.report["status"] = "STOPPED_REVIEW_REQUIRED"
        session.report["error"] = f"{type(error).__name__}: {error}"
        session.log(session.report["error"])
    finally:
        with contextlib.suppress(AttributeError):
            signal.alarm(0)
    if session is None:
        return 1
    result_zip = session.finish()
    print(f"RESULT: {session.report['status']}")
    print(f"DOWNLOAD_THIS: {result_zip}")
    print("No models trained. No compute launched. No Git push. No existing project files deleted.")
    return int(session.report["status"] == "STOPPED_REVIEW_REQUIRED")


if __name__ == "__main__":
    raise SystemExit(main())
