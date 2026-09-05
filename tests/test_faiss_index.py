from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
from gensim.models import KeyedVectors

from otto_recsys.retrieval.faiss_index import FaissConfig, build_faiss_index


def test_faiss_hnsw_index_returns_item_ids(tmp_path: Path) -> None:
    vectors = KeyedVectors(vector_size=3)
    vectors.add_vectors(
        [10, 20, 30],
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    path = tmp_path / "vectors.kv"
    vectors.save(str(path))

    manifest = build_faiss_index(
        path,
        tmp_path / "index",
        logger=logging.getLogger("test"),
        config=FaissConfig(
            m=2,
            ef_construction=16,
            ef_search=16,
            batch_size=2,
            threads=1,
        ),
        heartbeat_seconds=10.0,
    )

    index = faiss.read_index(str(tmp_path / "index" / "item.index"))
    query = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    faiss.normalize_L2(query)
    _, ids = index.search(query, 1)

    assert manifest.items == 3
    assert int(ids[0, 0]) == 10
