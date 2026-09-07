"""Behavioral contracts for observed features, query grouping and durable resume."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest

from otto_recsys.cloud.ranking_checkpoints import S3FeatureCheckpoints
from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import configure_logging
from otto_recsys.ranking.feature_cache import (
    build_feature_cache,
    make_contract,
    valid_part,
    validate_examples,
    workspace_lock,
)
from otto_recsys.ranking.features import (
    FEATURE_COLUMNS,
    candidate_features,
    observed_features,
    query_ledger,
)
from otto_recsys.ranking.splits import inner_partition, split_role
from otto_recsys.ranking.training_cache import build_ranking_training_cache

LOGGER = logging.getLogger("features-test")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "manifest.json").write_text(json.dumps({"manifest_id": "a" * 64}))
    sessions, labels = [], []
    for session in range(100):
        sessions.append(
            {
                "session": session,
                "events": [
                    {"aid": 1, "type": "clicks", "ts": 1000},
                    {"aid": 2, "type": "carts", "ts": 2000},
                    {"aid": 1, "type": "orders", "ts": 4000},
                ],
            }
        )
        labels.append(
            {
                "session": session,
                "labels": {
                    "clicks": 99,
                    "carts": list(range(50, 72)),
                    "orders": [1],
                },
            }
        )
    for name, rows in (("test_sessions", sessions), ("test_labels", labels)):
        (raw / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "cache"
    build_ranking_training_cache(raw, output, logger=LOGGER, folds=5, buckets=2)
    return output


def frames(cache: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    contract = make_contract(cache, inner_seed=20260907)
    examples = validate_examples(pl.read_parquet(cache / "examples.parquet"), contract)
    return (
        pl.read_parquet(cache / "events.parquet"),
        examples,
        pl.read_parquet(cache / "labels.parquet"),
    )


def run(cache: Path, output: Path, **kwargs: object) -> dict:
    return build_feature_cache(cache, output, logger=LOGGER, **kwargs)


def test_observed_values_and_full_denominators(cache: Path) -> None:
    events, examples, labels = frames(cache)
    sessions, items = observed_features(events, examples)
    row = sessions.filter(pl.col("session") == 0).row(0, named=True)
    assert (row["session_events"], row["session_unique_items"], row["session_duration_ms"]) == (
        3,
        2,
        3000,
    )
    first = items.filter((pl.col("session") == 0) & (pl.col("aid") == 1)).row(0, named=True)
    assert first["item_events"] == 2
    assert first["item_age_ms"] == 0
    assert first["item_last_type"] == 2
    assert first["item_event_share"] == pytest.approx(2 / 3)
    assert 99 not in items["aid"]
    ledger = query_ledger(examples, labels)
    assert ledger.height == 300
    assert ledger["query_id"].n_unique() == 300
    assert ledger.filter(pl.col("objective") == "carts")["true_items"].sum() == 2200
    assert ledger.filter(pl.col("objective") == "carts")["recall_denominator"].sum() == 2000


def candidates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source": ["time", "item2vec", "time"],
            "objective": ["clicks"] * 3,
            "session": [0] * 3,
            "aid": [1, 1, 42],
            "score": [4.0, 0.7, 2.0],
            "source_rank": [1, 2, 2],
        }
    )


def test_candidates_never_insert_labels_or_leak_targets(cache: Path) -> None:
    events, examples, labels = frames(cache)
    sessions, items = observed_features(events, examples)
    original = candidate_features(candidates(), sessions, items, labels)
    changed = labels.with_columns(
        pl.when(pl.col("objective") == "clicks")
        .then(pl.lit(1))
        .otherwise(pl.col("aid"))
        .alias("aid")
    )
    alternative = candidate_features(candidates().reverse(), sessions, items, changed)
    assert original.drop("target").equals(alternative.drop("target"))
    assert original["aid"].to_list() == [1, 42]
    assert original["target"].to_list() == [0, 0]
    assert alternative["target"].to_list() == [1, 0]
    assert original["source_count"].to_list() == [2, 1]
    assert original["reciprocal_rank_sum"].to_list() == [1.5, 0.5]
    assert original["item_age_ms"].to_list() == [0, None]
    assert "target" not in FEATURE_COLUMNS
    assert "session" not in FEATURE_COLUMNS


@pytest.mark.parametrize(
    "column,value",
    [
        ("score", float("nan")),
        ("score", float("inf")),
        ("source_rank", 0),
        ("source", "future_labels"),
    ],
)
def test_invalid_source_evidence_is_rejected(cache: Path, column: str, value: object) -> None:
    events, examples, labels = frames(cache)
    sessions, items = observed_features(events, examples)
    with pytest.raises(ValueError, match="invalid source"):
        candidate_features(
            candidates().head(1).with_columns(pl.lit(value).alias(column)), sessions, items, labels
        )


def test_duplicate_sources_and_future_events_are_rejected(cache: Path) -> None:
    events, examples, labels = frames(cache)
    sessions, items = observed_features(events, examples)
    with pytest.raises(ValueError, match="duplicate"):
        candidate_features(pl.concat([candidates(), candidates()]), sessions, items, labels)
    with pytest.raises(ValueError, match="boundaries"):
        observed_features(events.with_columns(pl.col("ts") + 1), examples)
    with pytest.raises(ValueError, match="chronological"):
        observed_features(events.with_columns(-pl.col("ts")), examples)


def test_split_roles_are_stable_disjoint_and_do_not_claim_temporal_holdout(cache: Path) -> None:
    contract = make_contract(cache, inner_seed=20260907)
    assert contract["untouched_temporal_holdout"] is False
    assert contract["retriever_fit_provenance_certified"] is False
    _, examples, _ = frames(cache)
    for outer in range(5):
        groups = {role: set() for role in ("fit", "inner", "outer")}
        for session, fold, inner in examples.select(
            "session", "fold", "inner_partition"
        ).iter_rows():
            assert inner == inner_partition(session, seed=20260907)
            groups[split_role(fold, inner, outer_fold=outer)].add(session)
        assert all(groups.values())
        assert set.union(*groups.values()) == set(range(100))
        assert sum(map(len, groups.values())) == 100
        assert not groups["fit"] & groups["outer"]
        assert not groups["inner"] & groups["outer"]
    with pytest.raises(ValueError, match="outer fold"):
        split_role(6, 0, outer_fold=0)
    with pytest.raises(ValueError, match="fold assignment"):
        validate_examples(examples.with_columns((pl.col("fold") + 1) % 5), contract)


def test_cache_resume_preserves_good_parts_and_rebuilds_corrupt_bucket(
    cache: Path, tmp_path: Path
) -> None:
    output = tmp_path / "features"
    first = run(cache, output)
    assert first["rows"] == {"sessions": 100, "items": 200, "queries": 300}
    part = output / "parts/part-000/items.parquet"
    modified = part.stat().st_mtime_ns
    second = run(cache, output)
    assert second["reused_buckets"] == 2
    assert second["parts_sha256"] == first["parts_sha256"]
    assert second["retained_compute_seconds"] == first["retained_compute_seconds"]
    assert part.stat().st_mtime_ns == modified
    (output / "parts/part-001/items.parquet").write_bytes(b"interrupted data")
    third = run(cache, output)
    assert third["reused_buckets"] == 1
    assert third["parts_sha256"] == first["parts_sha256"]
    assert part.stat().st_mtime_ns == modified
    assert valid_part(output, 1, first["input_id"]) is not None


def test_incompatible_inputs_fail_before_reuse(cache: Path, tmp_path: Path) -> None:
    output = tmp_path / "features"
    run(cache, output)
    with pytest.raises(ValueError, match="contract mismatch"):
        run(cache, output, inner_seed=11)
    (cache / "events.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(cache, output)


def test_concurrent_writers_are_rejected(tmp_path: Path) -> None:
    with (
        workspace_lock(tmp_path),
        pytest.raises(RuntimeError, match="active writer"),
        workspace_lock(tmp_path),
    ):
        pass


def test_progress_log_has_utc_duration_and_total(cache: Path, tmp_path: Path) -> None:
    output = tmp_path / "features"
    logger = configure_logging("ranking_features", log_dir=output / "logs")
    build_feature_cache(cache, output, logger=logger, heartbeat_seconds=0.001)
    records = [
        json.loads(line)
        for line in (output / "logs/ranking_features.jsonl").read_text().splitlines()
    ]
    assert all(row["timestamp"].endswith("+00:00") for row in records)
    assert any(row["message"] == "heartbeat" for row in records)
    completed = [row for row in records if row["message"] == "ranking_features_complete"]
    assert completed[0]["elapsed_seconds"] >= 0


class Remote:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fail_receipt = False
        self.writes: list[str] = []

    def run(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["aws", "--region", "us-west-2", "s3"]
        assert kwargs["timeout"] == 300
        operation, source, target = command[4:7]
        if operation == "sync":
            remote = self.root / source.removeprefix("s3://")
            if remote.exists():
                for path in remote.rglob("*.json"):
                    if path.name == "feature_contract.json" or path.parent.name == "parts":
                        destination = Path(target) / path.relative_to(remote)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, destination)
        else:
            assert operation == "cp"
            if target.startswith("s3://"):
                if self.fail_receipt and target.endswith("part-001.json"):
                    return subprocess.CompletedProcess(command, 1, "", "simulated disconnect")
                destination = self.root / target.removeprefix("s3://")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                self.writes.append(target)
            else:
                remote_file = self.root / source.removeprefix("s3://")
                if not remote_file.is_file():
                    return subprocess.CompletedProcess(command, 1, "", "HeadObject (404)")
                shutil.copyfile(remote_file, target)
        return subprocess.CompletedProcess(command, 0, "", "")


def store() -> S3FeatureCheckpoints:
    return S3FeatureCheckpoints("s3://bucket/ranking", region="us-west-2", logger=LOGGER)


def test_interrupted_upload_and_fresh_workspace_resume(
    cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = Remote(tmp_path / "remote")
    monkeypatch.setattr("otto_recsys.cloud.ranking_checkpoints.subprocess.run", remote.run)
    output = tmp_path / "features"
    remote.fail_receipt = True
    with pytest.raises(RuntimeError, match="disconnect"):
        run(cache, output, checkpoints=store())
    assert (output / "parts/part-001.json").is_file()
    remote.fail_receipt = False
    # Local completed computation survives failed publication.
    summary = run(cache, output, checkpoints=store())
    assert summary["reused_buckets"] == 2
    published = remote.root / "bucket/ranking" / summary["input_id"]
    assert (published / "manifest.json").exists()
    fresh = tmp_path / "fresh"
    restored = run(cache, fresh, checkpoints=store())
    assert restored["reused_buckets"] == 2
    assert restored["parts_sha256"] == summary["parts_sha256"]
    for stem in ("part-000", "part-001"):
        receipt_index = next(i for i, p in enumerate(remote.writes) if p.endswith(f"{stem}.json"))
        for family in ("sessions", "items", "queries"):
            assert any(
                p.endswith(f"{stem}/{family}.parquet") for p in remote.writes[:receipt_index]
            )
    (published / "parts/part-001/items.parquet").write_bytes(b"corrupt remote bucket")
    recovered = run(cache, tmp_path / "another", checkpoints=store())
    assert recovered["reused_buckets"] == 1
    assert recovered["parts_sha256"] == summary["parts_sha256"]
    (published / "parts/part-001/items.parquet").unlink()
    missing = run(cache, tmp_path / "missing", checkpoints=store())
    assert missing["reused_buckets"] == 1
    assert missing["parts_sha256"] == summary["parts_sha256"]


def test_remote_contract_and_access_errors_are_not_ignored(
    cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = Remote(tmp_path / "remote")
    monkeypatch.setattr("otto_recsys.cloud.ranking_checkpoints.subprocess.run", remote.run)
    result = run(cache, tmp_path / "features", checkpoints=store())
    contract = remote.root / "bucket/ranking" / result["input_id"] / "feature_contract.json"
    contract.write_text("{}")
    with pytest.raises(ValueError, match="remote feature contract"):
        run(cache, tmp_path / "fresh", checkpoints=store())
    monkeypatch.setattr(
        "otto_recsys.cloud.ranking_checkpoints.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "AccessDenied"),
    )
    with pytest.raises(RuntimeError, match="AccessDenied"):
        run(cache, tmp_path / "denied", checkpoints=store())


def test_input_hashes_are_real_content_hashes(cache: Path) -> None:
    contract = make_contract(cache, inner_seed=20260907)
    assert contract["input_sha256"]["events"] == sha256_file(cache / "events.parquet")


def test_independent_audit_detects_semantic_corruption(cache: Path, tmp_path: Path) -> None:
    import runpy

    from otto_recsys.experiments.manifest import canonical_json_sha256

    audit = runpy.run_path("scripts/audit_ranking_features.py")["audit"]
    output = tmp_path / "features"
    result = run(cache, output)
    assert audit(cache, output)["status"] == "passed"
    part = output / "parts/part-000/sessions.parquet"
    pl.read_parquet(part).with_columns(pl.col("session_duration_ms") + 1).write_parquet(part)
    receipt_path = output / "parts/part-000.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["sessions"].update({"sha256": sha256_file(part), "bytes": part.stat().st_size})
    receipt_path.write_text(json.dumps(receipt))
    receipts = [
        json.loads(p.read_text())["files"] for p in sorted((output / "parts").glob("*.json"))
    ]
    result["parts_sha256"] = canonical_json_sha256(receipts)
    (output / "manifest.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="reconciliation failed"):
        audit(cache, output)


def test_portfolio_notebook_is_executed_with_embedded_figures() -> None:
    import base64

    notebook = json.loads(Path("notebooks/07_ranking_features.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) >= 6
    images = []
    for cell in cells:
        assert cell["execution_count"] is not None
        for output in cell["outputs"]:
            assert output["output_type"] != "error"
            assert output.get("name") != "stderr"
            if "image/png" in output.get("data", {}):
                images.append(base64.b64decode(output["data"]["image/png"]))
    assert "notebook_complete" in str(cells[-1]["outputs"])
    for name in ("ranking_splits.png", "ranking_session_lengths.png"):
        assert (Path("reports/figures") / name).read_bytes() in images
