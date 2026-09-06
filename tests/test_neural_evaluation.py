from __future__ import annotations

import numpy as np
import pytest

from otto_recsys.retrieval.neural_evaluation import session_counts, summarize_counts


def test_ceiling_caps_hits_but_preserves_raw_exclusive_counts() -> None:
    counts = session_counts(set(range(50)), set(range(25)), list(range(25, 50)), (20, 25))
    assert counts == [20, 20, 20, 20, 20, 20, 20, 25]
    assert counts[3] - counts[1] == 0  # all top-20 capacity already covered by base


def test_missing_catalogue_positives_remain_in_denominator() -> None:
    assert session_counts({3, 999}, {3}, [1, 2, 3], (3,)) == [2, 1, 1, 1, 0]


def test_duplicate_predictions_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        session_counts({1}, set(), [1, 1], (2,))


def test_paired_bootstrap_matches_known_gain_and_is_reproducible() -> None:
    counts = np.tile(np.array([2, 1, 1, 2, 1]), (20, 3, 1))
    result = summarize_counts(counts, depths=(20,), iterations=50, seed=9)
    row = result["points"][0]
    assert row["weighted_base_ceiling"] == 0.5
    assert row["weighted_union_ceiling"] == 1.0
    assert row["weighted_incremental_ci95"] == [0.5, 0.5]
    assert result == summarize_counts(counts, depths=(20,), iterations=50, seed=9)


def test_missing_objective_is_not_silently_scored_zero() -> None:
    with pytest.raises(ValueError, match="each objective"):
        summarize_counts(np.zeros((10, 3, 5)), depths=(20,))


@pytest.mark.parametrize("search_method", ["exhaustive_inner_product", "faiss_ivfflat"])
def test_comparison_runs_real_baseline_and_resumes_verified_parts(
    tmp_path, monkeypatch, search_method
) -> None:
    import json
    import logging

    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gensim.models import KeyedVectors
    from test_comparison_checkpoints import LocalS3

    from otto_recsys.cloud.comparison_checkpoints import S3ComparisonCheckpoints
    from otto_recsys.experiments.manifest import sha256_file
    from otto_recsys.retrieval import neural_evaluation as module

    ranking, predictions, graphs, vectors_dir, index_dir, output = (
        tmp_path / name
        for name in ("ranking", "predictions", "graphs", "vectors", "index", "output")
    )
    for path in (ranking, predictions, graphs, vectors_dir, index_dir):
        path.mkdir()
    sessions = [10, 12, 13]
    items = pa.table(
        {
            "session": sessions,
            "aid": [1, 2, 3],
            "ts": [1000] * 3,
            "event_type": [0] * 3,
            "event_index": [0] * 3,
            "recency_rank": [1] * 3,
            "bucket": [0, 0, 1],
            "fold": [0, 0, 1],
        }
    )
    pq.write_table(items, ranking / "items.parquet")
    pq.write_table(pa.table({"session": sessions, "fold": [0, 0, 1]}), ranking / "examples.parquet")
    labels = [
        {
            "session": session,
            "objective": objective,
            "aid": 900,
            "bucket": session % 2,
            "fold": int(session == 13),
        }
        for session in sessions
        for objective in module.OBJECTIVES
    ]
    pq.write_table(pa.Table.from_pylist(labels), ranking / "labels.parquet")
    manifest = {
        "config": {"buckets": 2},
        "validation_manifest_id": "validation",
        **{
            name + "_sha256": sha256_file(ranking / (name + ".parquet"))
            for name in ("items", "labels", "examples")
        },
    }
    (ranking / "manifest.json").write_text(json.dumps(manifest))
    parts = []
    for objective in module.OBJECTIVES:
        for bucket in range(2):
            path = predictions / f"predictions/{objective}/part-{bucket:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            cohort = [10, 12] if bucket == 0 else []
            aids = [[900, *range(799)]] * len(cohort)
            pq.write_table(
                pa.table(
                    {
                        "session": pa.array(cohort, type=pa.int64()),
                        "aids": pa.array(aids, type=pa.list_(pa.int32())),
                    }
                ),
                path,
            )
            parts.append(
                {
                    "path": str(path.relative_to(predictions)),
                    "input_id": "prediction",
                    "sha256": sha256_file(path),
                }
            )
    (predictions / "prediction_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "ranking_manifest": manifest,
                "search": {"k": 800, "method": search_method},
                "validation_fold": 0,
                "input_id": "prediction",
                "sessions": 2,
                "parts": parts,
            }
        )
    )
    for graph in ("time", "type", "buy"):
        rows = [
            {"objective": objective, "source_aid": aid, "target_aid": 4, "score": 1.0}
            for objective in (("all",) if graph == "time" else module.OBJECTIVES)
            for aid in (1, 2, 3)
        ]
        pq.write_table(pa.Table.from_pylist(rows), graphs / (graph + ".parquet"))
        (graphs / (graph + ".json")).write_text("{}")
    vectors = KeyedVectors(vector_size=4)
    matrix = np.random.default_rng(2).normal(size=(5, 4)).astype(np.float32)
    faiss.normalize_L2(matrix)
    vectors.add_vectors(list(range(5)), matrix)
    vectors_path = vectors_dir / "vectors.kv"
    vectors.save(str(vectors_path))
    index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(4, 4, faiss.METRIC_INNER_PRODUCT))
    index.add_with_ids(matrix, np.arange(5, dtype=np.int64))
    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    for path in (vectors_dir, index_dir):
        (path / "manifest.json").write_text("{}")
    arguments = (ranking, predictions, graphs, vectors_path, index_path, output)
    remote = LocalS3(tmp_path / "remote")
    monkeypatch.setattr("otto_recsys.cloud.comparison_checkpoints.subprocess.run", remote.run)
    checkpoint_store = S3ComparisonCheckpoints(
        "s3://bucket/comparison", region="us-west-2", logger=logging.getLogger("test")
    )
    result = module.evaluate_neural_retrieval(
        *arguments,
        logger=logging.getLogger("test"),
        ann_k=5,
        source_k=5,
        iterations=10,
        memory_limit="256MB",
        checkpoint_store=checkpoint_store,
    )
    assert result["sessions"] == 2
    assert result["prediction_search"]["method"] == search_method
    assert result["points"][0]["weighted_base_ceiling"] == 0
    assert result["points"][0]["weighted_union_ceiling"] == 1
    receipt_time = (output / "parts/part-000.npz").stat().st_mtime_ns

    def no_recomputation(*args, **kwargs):
        raise AssertionError("completed baseline recomputed")

    monkeypatch.setattr(module, "create_covisit_source_candidates", no_recomputation)
    # A different local output directory must recover from S3 without rebuilding
    # any already completed baseline bucket, including an empty bucket.
    recovered = tmp_path / "recovered"
    second = module.evaluate_neural_retrieval(
        *arguments[:-1],
        recovered,
        logger=logging.getLogger("test"),
        ann_k=5,
        source_k=5,
        iterations=10,
        memory_limit="256MB",
        checkpoint_store=checkpoint_store,
    )
    assert second["points"] == result["points"]
    assert sha256_file(recovered / "parts/part-000.npz") == sha256_file(
        output / "parts/part-000.npz"
    )
    assert (output / "parts/part-000.npz").stat().st_mtime_ns == receipt_time
    (predictions / parts[0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        module.evaluate_neural_retrieval(
            *arguments,
            logger=logging.getLogger("test"),
            ann_k=5,
            source_k=5,
            iterations=10,
            memory_limit="256MB",
        )
