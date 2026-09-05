from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.retrieval.bulk_evaluation import evaluate_covisit_retrieval


def _write_validation_cache(root: Path) -> None:
    root.mkdir(parents=True)

    items = pa.table(
        {
            "session": pa.array([1], type=pa.int32()),
            "aid": pa.array([10], type=pa.int32()),
            "ts": pa.array([100], type=pa.int64()),
            "event_type": pa.array([0], type=pa.int8()),
            "event_index": pa.array([0], type=pa.uint16()),
            "recency_rank": pa.array([1], type=pa.uint16()),
            "bucket": pa.array([0], type=pa.uint16()),
        }
    )
    labels = pa.table(
        {
            "session": pa.array([1, 1, 1], type=pa.int32()),
            "objective": ["clicks", "carts", "orders"],
            "aid": pa.array([20, 30, 40], type=pa.int32()),
            "bucket": pa.array([0, 0, 0], type=pa.uint16()),
        }
    )

    pq.write_table(items, root / "items.parquet")
    pq.write_table(labels, root / "labels.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "validation_manifest_id": "validation",
                "buckets": 1,
                "sessions": 1,
                "item_rows": 1,
                "label_rows": 3,
                "items_sha256": "items",
                "labels_sha256": "labels",
            }
        ),
        encoding="utf-8",
    )


def _write_matrix(
    root: Path,
    name: str,
    rows: list[tuple[str, int, int, float, int]],
) -> None:
    path = root / f"{name}.parquet"
    manifest = root / f"{name}.json"

    table = pa.table(
        {
            "objective": [row[0] for row in rows],
            "source_aid": pa.array(
                [row[1] for row in rows], type=pa.int32()
            ),
            "target_aid": pa.array(
                [row[2] for row in rows], type=pa.int32()
            ),
            "score": pa.array(
                [row[3] for row in rows], type=pa.float64()
            ),
            "rank": pa.array([row[4] for row in rows], type=pa.int64()),
        }
    )

    pq.write_table(table, path)
    manifest.write_text(
        json.dumps({"rows": len(rows), "name": name}),
        encoding="utf-8",
    )


def test_bulk_evaluation_recovers_targets(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    _write_validation_cache(validation)

    covisit = tmp_path / "covisit"
    covisit.mkdir()

    _write_matrix(
        covisit,
        "time",
        [
            ("all", 10, 20, 5.0, 1),
            ("all", 10, 30, 4.0, 2),
            ("all", 10, 40, 3.0, 3),
        ],
    )
    _write_matrix(
        covisit,
        "type",
        [
            ("clicks", 10, 20, 5.0, 1),
            ("carts", 10, 30, 5.0, 1),
            ("orders", 10, 40, 5.0, 1),
        ],
    )
    _write_matrix(
        covisit,
        "buy",
        [
            ("carts", 10, 30, 5.0, 1),
            ("orders", 10, 40, 5.0, 1),
        ],
    )

    result = evaluate_covisit_retrieval(
        validation,
        covisit,
        tmp_path / "evaluation",
        logger=logging.getLogger("test"),
        buckets=1,
        ks=(20,),
        threads=1,
        memory_limit="512MB",
        temp_root=tmp_path / "duckdb-evaluation",
        heartbeat_seconds=10.0,
    )

    assert result.completed_buckets == 1
    assert result.metrics["revisit.clicks.recall_20"] == 0.0
    assert result.metrics["revisit_time.clicks.recall_20"] == 1.0
    assert result.metrics["revisit_time.carts.recall_20"] == 1.0
    assert result.metrics["revisit_time.orders.recall_20"] == 1.0
    assert result.metrics["full_covisit.weighted_recall_20"] == 1.0
