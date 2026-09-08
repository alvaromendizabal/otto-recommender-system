"""Independent chronology and provenance contracts for the research corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from otto_recsys.research.protocol import TemporalProtocol, build_temporal_corpus


def make_source(root: Path, *, late_aid: int = 9000) -> Path:
    root.mkdir()
    records = []
    sessions = {
        1: [(1, 10, 0), (2, 30, 1), (1, 50, 2)],
        2: [(2, 110, 0), (3, 120, 0), (3, 130, 1), (3, 130, 2)],
        3: [(1, 140, 0), (3, 150, 1), (3, 160, 2)],
        4: [(1, 210, 0), (3, 220, 0), (3, 250, 1), (3, 260, 2)],
        5: [(1, 310, 0), (late_aid, 320, 0), (late_aid, 330, 1), (late_aid, 330, 2)],
        6: [(1, 170, 0), (2, 200, 0)],  # crossing fit end leaves no future event
        7: [(1, 60, 0), (2, 150, 0)],  # history session cannot migrate into fit
    }
    for session, events in sessions.items():
        for index, (aid, ts, kind) in enumerate(events):
            records.append((session, aid, ts, kind, index))
    pl.DataFrame(
        records, schema=["session", "aid", "ts", "event_type", "event_index"], orient="row"
    ).write_parquet(root / "part-000000.parquet")
    return root


def build(source: Path, output: Path, **kwargs: int) -> dict:
    return build_temporal_corpus(
        source,
        output,
        TemporalProtocol(100, 200, 300, 400, **kwargs),
        logger=logging.getLogger("protocol-test"),
        threads=1,
        memory_gib=1,
    )


def test_temporal_labels_match_raw_future_and_preserve_unseen_items(tmp_path):
    source = make_source(tmp_path / "source")
    output = tmp_path / "corpus"
    report = build(source, output)
    raw = pl.read_parquet(source / "part-000000.parquet")
    observed = pl.read_parquet(output / "observed.parquet")
    queries = pl.read_parquet(output / "queries.parquet")
    labels = pl.read_parquet(output / "labels.parquet")
    assert set(queries["session"]) == {2, 3, 4, 5}
    assert pl.read_parquet(output / "history.parquet")["ts"].max() < 100
    assert 9000 in labels["aid"]
    for q in queries.iter_rows(named=True):
        prefix = observed.filter(pl.col("session") == q["session"])
        assert prefix["event_index"].max() == q["observed_last_index"]
        future = raw.filter(
            (pl.col("session") == q["session"])
            & (pl.col("event_index") > q["observed_last_index"])
            & (pl.col("ts") < q["period_end"])
        ).sort("event_index")
        action = {"clicks": 0, "carts": 1, "orders": 2}[q["objective"]]
        target = future.filter(pl.col("event_type") == action)
        expected = set(target["aid"][:1] if action == 0 else target["aid"])
        actual = labels.filter(
            (pl.col("session") == q["session"]) & (pl.col("objective") == q["objective"])
        )
        assert set(actual["aid"]) == expected
        assert q["recall_denominator"] == min(20, len(expected))
        assert (actual["label_ts"] >= q["query_ts"]).all()
    assert report["roles"]["evaluation"]["sessions"] == 1


def test_future_item_changes_cannot_change_sampling_or_fit_inputs(tmp_path):
    first = make_source(tmp_path / "a")
    second = make_source(tmp_path / "b", late_aid=8888)
    build(first, tmp_path / "aout", fit_sessions=1)
    build(second, tmp_path / "bout", fit_sessions=1)
    for name in ("observed", "labels", "queries"):
        a = pl.read_parquet(tmp_path / f"aout/{name}.parquet").filter(pl.col("split_role") == "fit")
        b = pl.read_parquet(tmp_path / f"bout/{name}.parquet").filter(pl.col("split_role") == "fit")
        assert a.equals(b)


def test_resume_verifies_files_and_rejects_changed_protocol(tmp_path):
    source = make_source(tmp_path / "source")
    output = tmp_path / "output"
    first = build(source, output)
    path = output / "history.parquet"
    modified = path.stat().st_mtime_ns
    assert build(source, output) == first
    assert path.stat().st_mtime_ns == modified
    with pytest.raises(ValueError, match="different protocol"):
        build(source, output, seed=123)
    assert path.stat().st_mtime_ns == modified
    inventory = [{"path": "part-000000.parquet", "sha256": "0" * 64}]
    (source / "download_inventory.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="download inventory"):
        build(source, tmp_path / "bad")


def test_invalid_boundaries_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        TemporalProtocol(100, 200, 200, 400).validate()
