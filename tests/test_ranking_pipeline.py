"""Real Parquet/SQL/LightGBM integration on explicitly synthetic test sessions."""
from __future__ import annotations

import fnmatch
import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import duckdb
import polars as pl
import pytest

from otto_recsys.cloud.ranking_stage import S3CandidateCheckpoints, S3ModelCheckpoints
from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.candidates import (
    CandidateConfig,
    compress_sources,
    valid_part,
    write_candidate_parts,
)
from otto_recsys.ranking.feature_cache import write_json
from otto_recsys.ranking.features import (
    FEATURE_COLUMNS,
    OBJECTIVES,
    observed_features,
    query_ledger,
)
from otto_recsys.ranking.lambdarank import RankerConfig
from otto_recsys.ranking.pipeline import (
    pool_metrics,
    role_filter,
    run_ranking,
    training_memory_guard,
)

LOGGER = logging.getLogger("ranking_integration")


def fixture_frames():
    examples = pl.DataFrame({
        "session": list(range(1, 91)), "fold": [sid % 3 for sid in range(1, 91)],
        "bucket": [sid % 2 for sid in range(1, 91)],
        "inner_partition": [(sid // 3) % 5 for sid in range(1, 91)],
        "observed_events": [1] * 90, "observed_unique_items": [1] * 90,
        "first_ts": [1000] * 90, "last_ts": [1000] * 90,
    })
    events = examples.select("session", "fold", "bucket").with_columns(
        (pl.col("session") % 30).alias("aid"), pl.lit(1000).alias("ts"),
        pl.lit(0).alias("event_type"), pl.lit(0).alias("event_index"),
    )
    labels = examples.select("session", "fold", "bucket").join(
        pl.DataFrame({"objective": OBJECTIVES}), how="cross"
    ).with_columns((pl.col("session") % 30).alias("aid"))
    extra = labels.filter(pl.col("session") == 1).with_columns(pl.lit(999).alias("aid"))
    labels = pl.concat([labels, extra], how="vertical_relaxed")
    rows = [(source, objective, sid, aid,
             float(30 - aid if source == "revisit" else -abs(aid - sid % 30)), aid + 1)
            for sid in range(1, 90) for objective in OBJECTIVES
            for aid in range(30) for source in ("revisit", "item2vec")]
    sources = pl.DataFrame(rows, schema=["source", "objective", "session", "aid",
                                        "score", "source_rank"], orient="row")
    return examples, events, labels, sources


def write_observed(path, examples, events, labels):
    path.mkdir(parents=True)
    sessions, items = observed_features(events, examples)
    sessions.write_parquet(path / "sessions.parquet")
    items.write_parquet(path / "items.parquet")
    query_ledger(examples, labels).write_parquet(path / "queries.parquet")


def materialized_cache(tmp_path, *, candidate_k=30):
    examples, events, labels, sources = fixture_frames()
    cache = tmp_path / "candidates"
    cache.mkdir()
    contract = {"schema_version": 1, "buckets": 2, "folds": 3,
                "config": {"candidate_k": candidate_k}, "feature_names": list(FEATURE_COLUMNS),
                "validation_scope": "synthetic contract test; not an OTTO performance result",
                "neural_candidates": False}
    write_json(cache / "candidate_contract.json", contract)
    input_id = canonical_json_sha256(contract)
    for bucket in range(2):
        observed = tmp_path / "observed" / str(bucket)
        subset = examples.filter(pl.col("bucket") == bucket)
        part_labels = labels.filter(pl.col("bucket") == bucket)
        write_observed(observed, subset, events.filter(pl.col("bucket") == bucket), part_labels)
        root = cache / "parts" / f"part-{bucket:03d}"
        with duckdb.connect() as connection:
            connection.register(
                "input_sources", sources.filter(pl.col("session") % 2 == bucket).to_arrow()
            )
            connection.execute("CREATE TABLE source_candidates AS SELECT * FROM input_sources")
            files = write_candidate_parts(connection, observed, part_labels, root,
                                          config=CandidateConfig(
                                              candidate_k=candidate_k, batch_sessions=7
                                          ))
        write_json(root.with_suffix(".json"), {"input_id": input_id, "bucket": bucket,
                                             "files": files, "compute_seconds": 0.01})
    return cache, input_id


@pytest.mark.parametrize("changes", [
    {"candidate_k": 0}, {"candidate_k": 10001}, {"threads": True},
    {"batch_sessions": -1}, {"candidate_k": 1000, "batch_sessions": 101},
    {"ef_search": 0}, {"source_k": 1.5},
])
def test_invalid_candidate_config(changes):
    with pytest.raises(ValueError):
        replace(CandidateConfig(), **changes).validate()


def test_default_candidate_config():
    CandidateConfig().validate()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_compression_rejects_invalid_budget(budget):
    with pytest.raises(ValueError):
        compress_sources(None, budget)


def test_compression_ties_and_label_blindness():
    with duckdb.connect() as connection:
        connection.execute("""CREATE TABLE source_candidates AS SELECT * FROM
            (VALUES ('revisit','clicks',1,9,1.0,1), ('revisit','clicks',1,4,1.0,1))
            AS t(source,objective,session,aid,score,source_rank)""")
        compress_sources(connection, 1)
        assert connection.execute("SELECT aid FROM selected_candidates").fetchall() == [(4,)]
        assert connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='vlabels'"
        ).fetchone()[0] == 0


def test_streamed_features_have_full_queries_and_no_inserted_positive(tmp_path):
    cache, identity = materialized_cache(tmp_path)
    for bucket in range(2):
        receipt = valid_part(cache, bucket, identity)
        assert receipt is not None
        root = cache / "parts" / f"part-{bucket:03d}"
        queries = pl.read_parquet(root / "queries.parquet")
        assert queries.height == 45 * 3
        for objective in OBJECTIVES:
            rows = pl.read_parquet(root / f"{objective}.parquet")
            assert not rows.select("session", "aid").is_duplicated().any()
            assert rows.filter(pl.col("aid") == 999).is_empty()
            assert rows.filter(pl.col("session") == 90).is_empty()
            assert set(rows["objective"].unique()) == {objective}
            assert max(rows.group_by("session").len()["len"]) <= 30
            assert all(rows.schema[name] == pl.Float32 for name in FEATURE_COLUMNS)


@pytest.mark.parametrize("damage", ["file", "receipt", "missing", "rows"])
def test_candidate_receipts_reject_corruption(tmp_path, damage):
    cache, identity = materialized_cache(tmp_path)
    root = cache / "parts" / "part-000"
    if damage == "file":
        (root / "clicks.parquet").write_bytes(b"corrupt")
    elif damage == "missing":
        (root / "queries.parquet").unlink()
    elif damage == "receipt":
        root.with_suffix(".json").write_text("{")
    else:
        receipt = json.loads(root.with_suffix(".json").read_text())
        receipt["files"]["clicks"]["rows"] += 1
        write_json(root.with_suffix(".json"), receipt)
    assert valid_part(cache, 0, identity) is None
    assert valid_part(cache, 1, identity) is not None


def test_complete_query_labels_cannot_change_candidate_membership(tmp_path):
    examples, events, labels, sources = fixture_frames()
    observed = tmp_path / "observed"
    write_observed(observed, examples, events, labels)
    results = []
    targets = (labels, labels.with_columns((pl.col("aid") + 10000).alias("aid")))
    for index, target in enumerate(targets):
        with duckdb.connect() as connection:
            connection.register("source_input", sources.to_arrow())
            connection.execute("CREATE TABLE source_candidates AS SELECT * FROM source_input")
            write_candidate_parts(connection, observed, target, tmp_path / str(index),
                                  config=CandidateConfig(candidate_k=21, batch_sessions=10))
            results.append(pl.read_parquet(tmp_path / str(index) / "orders.parquet"))
    assert results[0].drop("target").equals(results[1].drop("target"))
    assert results[0]["target"].sum() > 0
    assert results[1]["target"].sum() == 0


def test_memory_guard_does_not_silently_sample():
    with pytest.raises(MemoryError, match="No rows were silently sampled"):
        training_memory_guard(10**12, 30, 1)
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            training_memory_guard(100, 30, invalid)


def test_roles_are_disjoint_and_exhaustive():
    examples, _, _, _ = fixture_frames()
    roles = [set(examples.filter(role_filter(role, 0))["session"])
             for role in ("fit", "inner", "outer")]
    assert all(roles)
    assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    assert set.union(*roles) == set(examples["session"])
    with pytest.raises(ValueError):
        role_filter("test", 0)


def test_rankers_execute_and_reuse_all_three_objectives(tmp_path, monkeypatch):
    cache, _ = materialized_cache(tmp_path)
    output = tmp_path / "ranking"
    config = RankerConfig(rounds=6, patience=3, checkpoint_every=2,
                          num_leaves=4, min_data_in_leaf=2, threads=1)
    report = run_ranking(cache, output, outer_folds=(0,), config=config,
                         logger=LOGGER, max_memory_gib=1)
    assert report["status"] == "passed"
    assert report["untouched_temporal_holdout"] is False
    assert report["learned"]["objectives"]["orders"]["denominator"] == 30
    assert report["folds"][0]["objectives"]["orders"]["learned"]["all_queries"] == 30
    assert report["learned"]["weighted_recall_at_20"] <= 29 / 30 + 1e-12
    saved = {str(path): sha256_file(path) for path in output.glob("fold-*/*/model.txt")}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("completed objectives must not retrain")

    monkeypatch.setattr("otto_recsys.ranking.pipeline.fit_ranker", forbidden_fit)
    resumed = run_ranking(cache, output, outer_folds=(0,), config=config,
                          logger=LOGGER, max_memory_gib=1)
    assert resumed["learned"] == report["learned"]
    assert all(sha256_file(Path(path)) == digest for path, digest in saved.items())
    with pytest.raises(ValueError, match="contract mismatch"):
        run_ranking(cache, output, outer_folds=(0,), config=replace(config, rounds=7),
                    logger=LOGGER, max_memory_gib=1)


def test_pooled_metrics_use_query_and_label_denominators():
    def part(hits, denominator, queries, score):
        return {"hits": hits, "denominator": denominator, "labeled_queries": queries,
                "all_queries": queries, "candidate_rows": queries * 20,
                "ndcg_at_20": score, "mrr_at_20": score, "hit_rate_at_20": score,
                "candidate_ceiling_at_20": score}
    pooled = pool_metrics([part(1, 2, 1, 0.5), part(9, 18, 9, 1.0)])
    assert pooled["recall_at_20"] == 0.5
    assert pooled["ndcg_at_20"] == pytest.approx(0.95)
    with pytest.raises(ValueError):
        pool_metrics([])


class FakeTransfers:
    def upload(self, path, relative):
        self.upload_order.append(relative)
        target = self.remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    def run(self, arguments, *, allow_missing=False):
        if arguments[0] == "sync":
            destination = Path(arguments[2])
            patterns = [arguments[i + 1] for i, value in enumerate(arguments)
                        if value == "--include"]
            for source in self.remote.rglob("*"):
                relative = source.relative_to(self.remote).as_posix()
                if source.is_file() and any(
                    fnmatch.fnmatch(relative, pattern) for pattern in patterns
                ):
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        else:
            relative = arguments[1].removeprefix(self.uri + "/")
            source = self.remote / relative
            if not source.exists() and allow_missing:
                return False
            shutil.copyfile(source, arguments[2])
        return True


class CandidateStore(FakeTransfers, S3CandidateCheckpoints):
    def __init__(self, remote):
        super().__init__("s3://otto-test/ranking/candidates", region="us-west-2", logger=LOGGER)
        self.remote = remote
        self.remote.mkdir(exist_ok=True)
        self.upload_order = []


class ModelStore(FakeTransfers, S3ModelCheckpoints):
    def __init__(self, remote):
        super().__init__("s3://otto-test/ranking/models", region="us-west-2", logger=LOGGER)
        self.remote = remote
        self.remote.mkdir(exist_ok=True)
        self.upload_order = []


def test_candidate_s3_restore_is_receipt_last_and_checksum_verified(tmp_path):
    cache, identity = materialized_cache(tmp_path)
    store = CandidateStore(tmp_path / "remote")
    store.restore(cache, identity)
    store.publish_part(cache, 0, identity)
    assert store.upload_order[-1] == "parts/part-000.json"
    assert all(store.upload_order.index(f"parts/part-000/{family}.parquet")
               < store.upload_order.index("parts/part-000.json")
               for family in (*OBJECTIVES, "queries"))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    shutil.copyfile(cache / "candidate_contract.json", fresh / "candidate_contract.json")
    store.restore(fresh, identity)
    assert valid_part(fresh, 0, identity) is not None
    (store.remote / "parts/part-000/orders.parquet").write_bytes(b"corrupt")
    other = tmp_path / "other"
    other.mkdir()
    shutil.copyfile(cache / "candidate_contract.json", other / "candidate_contract.json")
    store.restore(other, identity)
    assert valid_part(other, 0, identity) is None


def test_model_s3_roundtrip_reuses_models_without_retraining(tmp_path, monkeypatch):
    cache, _ = materialized_cache(tmp_path)
    config = RankerConfig(rounds=4, patience=2, checkpoint_every=2,
                          num_leaves=4, min_data_in_leaf=2, threads=1)
    store = ModelStore(tmp_path / "remote")
    result = run_ranking(cache, tmp_path / "original", outer_folds=(0,), config=config,
                         logger=LOGGER, checkpoints=store, max_memory_gib=1)
    prefix = "fold-0/clicks/"
    assert (store.upload_order.index(prefix + "model.txt")
            < store.upload_order.index(prefix + "evaluation_receipt.json"))

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("fresh workspace must restore completed objectives")

    monkeypatch.setattr("otto_recsys.ranking.pipeline.fit_ranker", forbidden_fit)
    restored = run_ranking(cache, tmp_path / "fresh", outer_folds=(0,), config=config,
                           logger=LOGGER, checkpoints=ModelStore(store.remote), max_memory_gib=1)
    assert restored["learned"] == result["learned"]
    assert len(list((tmp_path / "fresh").glob("fold-*/*/model.txt"))) == 3
