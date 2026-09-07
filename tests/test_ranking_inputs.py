"""Regression coverage for an absent Studio feature cache and immutable recovery."""
from __future__ import annotations

import importlib.util
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from otto_recsys.cloud.ranking_checkpoints import S3FeatureCheckpoints
from otto_recsys.cloud.ranking_inputs import prepare_ranking_inputs, restore_observed_features
from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import FAMILIES, valid_part, write_json

LOGGER = logging.getLogger("ranking_input_contracts")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def close_cli_log_handlers():
    yield
    for name in ("ranking", "ranking_candidates"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    evidence = tmp_path / "evidence"
    remote = tmp_path / "remote"
    observed = tmp_path / "restored"
    covisit = tmp_path / "covisit"
    vectors = tmp_path / "vectors/items.kv"
    index = tmp_path / "index/items.index"
    for path in (cache, evidence, remote, covisit, vectors.parent, index.parent):
        path.mkdir(parents=True)
    for name in ("items", "examples", "labels"):
        pl.DataFrame({"session": [1, 2], "aid": [11, 12]}).write_parquet(
            cache / f"{name}.parquet"
        )
    ranking = {"input_id": "a" * 64, "validation_manifest_id": "b" * 64}
    ranking.update({f"{name}_sha256": sha256_file(cache / f"{name}.parquet")
                    for name in ("items", "examples", "labels")})
    write_json(cache / "manifest.json", ranking)
    for family in ("time", "type", "buy"):
        (covisit / f"{family}.parquet").write_bytes(b"synthetic prerequisite stub")
        write_json(covisit / f"{family}.json", {"synthetic": True})
    for path in (vectors, index):
        path.write_bytes(b"synthetic prerequisite stub")
        write_json(path.parent / "manifest.json", {"synthetic": True})
    contract = {
        "schema_version": 1, "buckets": 2, "folds": 3,
        "training_cache_input_id": ranking["input_id"],
        "validation_manifest_id": ranking["validation_manifest_id"],
        "input_sha256": {name: ranking[f"{name}_sha256"] for name in ("examples", "labels")},
    }
    identity = canonical_json_sha256(contract)
    write_json(evidence / "ranking_feature_contract.json", contract)
    shutil.copyfile(evidence / "ranking_feature_contract.json", remote / "feature_contract.json")
    receipts = []
    for bucket in range(2):
        part = remote / "parts" / f"part-{bucket:03d}"
        part.mkdir(parents=True)
        files = {}
        for family in FAMILIES:
            path = part / f"{family}.parquet"
            frame = pl.DataFrame({"session": [bucket * 2 + 1, bucket * 2 + 2],
                                  "value": [1, 2]})
            frame.write_parquet(path)
            files[family] = {"rows": frame.height, "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
        receipt = {"input_id": identity, "bucket": bucket, "files": files,
                   "compute_seconds": 0.25, "completed_at_utc": "2026-09-07T00:00:00+00:00"}
        write_json(part.with_suffix(".json"), receipt)
        receipts.append(receipt)
    report = {
        "status": "passed", "input_id": identity, "completed_buckets": 2,
        "contract_sha256": sha256_file(evidence / "ranking_feature_contract.json"),
        "parts_sha256": canonical_json_sha256([r["files"] for r in receipts]),
        "rows": {name: 4 for name in FAMILIES},
    }
    write_json(evidence / "ranking_features.json", report)
    write_json(evidence / "ranking_features_audit.json", {
        "status": "passed", "input_id": identity, "verified_buckets": 2,
        "feature_contract_sha256": report["contract_sha256"],
        "parts_sha256": report["parts_sha256"],
        "mismatches": {"sessions": 0, "items": 0, "queries": 0, "duplicate_keys": 0},
    })
    uri = f"s3://otto-test/ranking/features/{identity}"
    write_json(evidence / "ranking_features_publication.json", {
        "status": "passed", "input_id": identity, "uri": uri + "/",
    })
    state = SimpleNamespace(cache=cache, evidence=evidence, remote=remote, observed=observed,
                            covisit=covisit, vectors=vectors, index=index, identity=identity,
                            report=report, reads=[], fail_on=None, output=tmp_path / "run")

    def download_only(self, arguments, *, allow_missing=False):
        assert arguments[0] == "cp", "input restoration may only read exact S3 objects"
        assert arguments[1].startswith(uri + "/"), "never guess a new or latest namespace"
        relative = arguments[1][len(uri) + 1:]
        state.reads.append(relative)
        if state.fail_on == relative:
            raise OSError("simulated interrupted transfer")
        shutil.copyfile(remote / relative, arguments[2])
        return True

    def forbidden_rebuild(*args, **kwargs):
        raise AssertionError("restoring completed features must not recompute them")

    monkeypatch.setattr(S3FeatureCheckpoints, "run", download_only)
    monkeypatch.setattr("otto_recsys.ranking.feature_cache.build_feature_cache", forbidden_rebuild)
    return state


def restore(stored):
    return restore_observed_features(stored.observed, stored.evidence,
                                     region="us-west-2", logger=LOGGER)


def prepare(stored):
    return prepare_ranking_inputs(
        stored.cache, stored.observed, stored.covisit, stored.vectors, stored.index,
        evidence=stored.evidence, region="us-west-2", logger=LOGGER,
    )


def snapshots(root):
    return {path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
            for path in root.rglob("*.parquet")}


def test_absent_cache_restored_without_feature_computation(stored):
    assert not stored.observed.exists()
    result = prepare(stored)
    assert result["status"] == "passed"
    assert result["restored_buckets"] == 2
    assert result["downloaded_data_files"] == 6
    assert result["feature_computation_performed"] is False
    assert result["parts_sha256"] == stored.report["parts_sha256"]
    assert all(valid_part(stored.observed, bucket, stored.identity) for bucket in range(2))
    assert sha256_file(stored.observed / "feature_contract.json") == stored.report["contract_sha256"]


def test_complete_cache_reuse_needs_no_network_and_preserves_bytes_mtimes(stored):
    restore(stored)
    before = snapshots(stored.observed)
    stored.reads.clear()
    result = restore(stored)
    assert result["reused_buckets"] == 2
    assert result["downloaded_data_files"] == 0
    assert stored.reads == []
    assert snapshots(stored.observed) == before


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_one_damaged_file_does_not_redownload_good_files(stored, damage):
    restore(stored)
    relative = "parts/part-001/items.parquet"
    path = stored.observed / relative
    before = snapshots(stored.observed)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"bad")
    stored.reads.clear()
    result = restore(stored)
    assert result["downloaded_data_files"] == 1
    assert result["restored_buckets"] == result["reused_buckets"] == 1
    assert [name for name in stored.reads if name.endswith(".parquet")] == [relative]
    after = snapshots(stored.observed)
    assert after[relative][0] == before[relative][0]
    assert all(after[name] == value for name, value in before.items() if name != relative)


def test_missing_receipt_restores_metadata_without_recomputing_or_downloading_data(stored):
    restore(stored)
    (stored.observed / "parts/part-000.json").unlink()
    before = snapshots(stored.observed)
    stored.reads.clear()
    result = restore(stored)
    assert result["restored_buckets"] == 1
    assert result["downloaded_data_files"] == 0
    assert snapshots(stored.observed) == before


@pytest.mark.parametrize("damage", ["missing", "invalid_json"])
def test_missing_contract_with_valid_parts_does_not_fetch_again(stored, damage):
    restore(stored)
    path = stored.observed / "feature_contract.json"
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("{")
    stored.reads.clear()
    result = restore(stored)
    assert result["reused_buckets"] == 2
    assert stored.reads == []
    assert sha256_file(path) == stored.report["contract_sha256"]


def test_incompatible_local_work_is_preserved(stored):
    stored.observed.mkdir()
    path = stored.observed / "feature_contract.json"
    write_json(path, {"belongs_to_another_experiment": True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        restore(stored)
    assert path.read_bytes() == before
    assert stored.reads == []


@pytest.mark.parametrize("artifact", ["contract", "receipt", "data"])
def test_bad_remote_evidence_never_commits_a_bucket(stored, artifact):
    if artifact == "contract":
        (stored.remote / "feature_contract.json").write_text("{}")
    elif artifact == "receipt":
        path = stored.remote / "parts/part-000.json"
        value = json.loads(path.read_text())
        value["files"]["queries"]["sha256"] = "0" * 64
        write_json(path, value)
    else:
        path = stored.remote / "parts/part-000/queries.parquet"
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1  # Same byte size does not satisfy SHA-256.
        path.write_bytes(raw)
    with pytest.raises(ValueError):
        restore(stored)
    assert not (stored.observed / "parts/part-000.json").exists()
    assert valid_part(stored.observed, 0, stored.identity) is None


def test_interruption_preserves_completed_bucket_and_file_level_work(stored):
    stored.fail_on = "parts/part-001/queries.parquet"
    with pytest.raises(OSError, match="interrupted"):
        restore(stored)
    assert valid_part(stored.observed, 0, stored.identity) is not None
    assert valid_part(stored.observed, 1, stored.identity) is None
    before = snapshots(stored.observed)
    stored.reads.clear()
    stored.fail_on = None
    result = restore(stored)
    assert result["reused_buckets"] == 1
    assert result["downloaded_data_files"] == 1
    assert [name for name in stored.reads if name.endswith(".parquet")] == [
        "parts/part-001/queries.parquet"
    ]
    after = snapshots(stored.observed)
    assert all(after[name] == value for name, value in before.items())


@pytest.mark.parametrize("damage", ["audit", "published_identity", "cache_identity", "input_bytes"])
def test_incompatible_provenance_rejected_before_network(stored, damage):
    if damage == "audit":
        path = stored.evidence / "ranking_features_audit.json"
        value = json.loads(path.read_text())
        value["mismatches"]["queries"] = 1
    elif damage == "published_identity":
        path = stored.evidence / "ranking_features_publication.json"
        value = json.loads(path.read_text())
        value["input_id"] = "f" * 64
    elif damage == "cache_identity":
        path = stored.cache / "manifest.json"
        value = json.loads(path.read_text())
        value["input_id"] = "f" * 64
    else:
        path = stored.cache / "labels.parquet"
        path.write_bytes(b"corrupt frozen input")
        value = None
    if value is not None:
        write_json(path, value)
    with pytest.raises(ValueError):
        prepare(stored)
    assert stored.reads == []
    assert not stored.observed.exists()


def test_preflight_reports_all_missing_upstream_paths_together(stored):
    missing = [stored.cache / "items.parquet", stored.covisit / "time.parquet", stored.vectors]
    for path in missing:
        path.unlink()
    with pytest.raises(ValueError, match="Frozen ranking inputs are missing") as failure:
        prepare(stored)
    assert all(str(path) in str(failure.value) for path in missing)
    assert stored.reads == []


def load_cli():
    spec = importlib.util.spec_from_file_location("ranking_cli_regression", ROOT / "scripts/run_ranking.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(stored, stage):
    return ["--checkpoint-uri", "s3://otto-test/ranking", "--stage", stage,
            "--ranking-cache", str(stored.cache), "--observed-features", str(stored.observed),
            "--feature-evidence", str(stored.evidence), "--covisit-dir", str(stored.covisit),
            "--vectors", str(stored.vectors), "--index", str(stored.index),
            "--output-dir", str(stored.output)]


def test_real_cli_preflight_from_absent_cache_never_starts_training(stored, monkeypatch, capsys):
    cli = load_cli()

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight cannot materialize candidates or fit models")

    monkeypatch.setattr(cli, "build_candidates", forbidden)
    monkeypatch.setattr(cli, "run_ranking", forbidden)
    assert not stored.observed.exists()
    assert cli.main(arguments(stored, "preflight")) == 0
    report = json.loads((stored.output / "input_readiness.json").read_text())
    assert report["restored_buckets"] == 2
    assert report["feature_computation_performed"] is False
    assert "OTTO_RANKING_PREFLIGHT_PASSED" in capsys.readouterr().out


def test_real_cli_restores_prerequisites_before_candidate_generation(stored, monkeypatch):
    cli = load_cli()

    def reached_candidates(*paths, **kwargs):
        assert paths[1] == stored.observed
        assert all(valid_part(paths[1], bucket, stored.identity) for bucket in range(2))
        raise RuntimeError("candidate stage reached after verified restoration")

    monkeypatch.setattr(cli, "build_candidates", reached_candidates)
    assert not stored.observed.exists()
    with pytest.raises(RuntimeError, match="candidate stage reached after verified restoration"):
        cli.main(arguments(stored, "all"))
    assert json.loads((stored.output / "input_readiness.json").read_text())["status"] == "passed"
