from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models import KeyedVectors

from otto_recsys.retrieval.embedding_evaluation import evaluate_embedding_retrieval


def test_embedding_evaluation_smoke(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()

    items = pa.table(
        {
            "session": pa.array([1, 1], type=pa.int32()),
            "aid": pa.array([10, 20], type=pa.int32()),
            "ts": pa.array([1, 2], type=pa.int64()),
            "event_type": pa.array([0, 1], type=pa.int8()),
            "event_index": pa.array([0, 1], type=pa.uint16()),
            "recency_rank": pa.array([2, 1], type=pa.uint16()),
            "bucket": pa.array([0, 0], type=pa.uint16()),
        }
    )
    labels = pa.table(
        {
            "session": pa.array([1], type=pa.int32()),
            "objective": pa.array(["clicks"], type=pa.string()),
            "aid": pa.array([30], type=pa.int32()),
            "bucket": pa.array([0], type=pa.uint16()),
        }
    )
    pq.write_table(items, cache / "items.parquet")
    pq.write_table(labels, cache / "labels.parquet")

    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        [10, 20, 30],
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    vectors_path = tmp_path / "vectors.kv"
    vectors.save(str(vectors_path))

    base = faiss.IndexHNSWFlat(2, 2, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIDMap2(base)
    matrix = np.asarray(vectors.vectors, dtype=np.float32).copy()
    faiss.normalize_L2(matrix)
    index.add_with_ids(matrix, np.asarray([10, 20, 30], dtype=np.int64))
    index_path = tmp_path / "item.index"
    faiss.write_index(index, str(index_path))

    result = evaluate_embedding_retrieval(
        cache,
        vectors_path,
        index_path,
        tmp_path / "evaluation",
        logger=logging.getLogger("test"),
        buckets=1,
        ks=(1, 3),
        ann_k=3,
        ef_search=16,
        heartbeat_seconds=10.0,
    )

    assert result.sessions == 1
    assert result.metrics["clicks.recall_3"] == 1.0
