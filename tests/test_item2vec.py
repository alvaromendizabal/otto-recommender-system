from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from gensim.models import KeyedVectors

import otto_recsys.retrieval.item2vec as module
from otto_recsys.retrieval.item2vec import (
    Item2VecConfig,
    SessionSequenceCorpus,
    train_item2vec,
)


def _write_training_data(path: Path) -> None:
    rows = [
        {
            "session": 1,
            "events": [
                {"aid": 10, "ts": 1, "type": "clicks"},
                {"aid": 20, "ts": 2, "type": "clicks"},
                {"aid": 30, "ts": 3, "type": "carts"},
            ],
        },
        {
            "session": 2,
            "events": [
                {"aid": 10, "ts": 4, "type": "clicks"},
                {"aid": 20, "ts": 5, "type": "orders"},
            ],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_session_corpus_streams_integer_sequences(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_training_data(source)

    corpus = SessionSequenceCorpus(source)
    assert list(corpus) == [[10, 20, 30], [10, 20]]
    assert corpus.sessions == 2
    assert corpus.tokens == 5


def test_item2vec_small_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "train.jsonl"
    _write_training_data(source)

    validation_manifest = tmp_path / "manifest.json"
    validation_manifest.write_text(
        json.dumps({"manifest_id": "abc"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "validate_training_resources", lambda: None)

    manifest = train_item2vec(
        source,
        validation_manifest,
        tmp_path / "output",
        logger=logging.getLogger("test"),
        config=Item2VecConfig(
            vector_size=8,
            window=2,
            negative=2,
            epochs=1,
            workers=1,
            seed=42,
        ),
        heartbeat_seconds=10.0,
    )

    vectors = KeyedVectors.load(
        str(tmp_path / "output" / "item_vectors.kv"),
        mmap="r",
    )

    assert manifest.vocabulary_size == 3
    assert manifest.corpus_sessions == 2
    assert manifest.corpus_tokens == 5
    assert len(vectors) == 3
