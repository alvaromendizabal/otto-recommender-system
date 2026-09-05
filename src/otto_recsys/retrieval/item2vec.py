from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import psutil
from gensim.models import KeyedVectors, Word2Vec  # type: ignore[import-untyped]
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.runtime import Heartbeat

_MIN_TOTAL_RAM_GIB = 16.0
_MIN_AVAILABLE_RAM_GIB = 12.0


@dataclass(frozen=True)
class Item2VecConfig:
    vector_size: int = 128
    window: int = 10
    negative: int = 10
    epochs: int = 5
    workers: int = 4
    seed: int = 42
    sample: float = 1e-4
    ns_exponent: float = 0.75


@dataclass(frozen=True)
class Item2VecManifest:
    validation_manifest_id: str
    config: Item2VecConfig
    corpus_sessions: int
    corpus_tokens: int
    vocabulary_size: int
    vector_size: int
    model_family_sha256: str
    vectors_family_sha256: str
    elapsed_seconds: float


class SessionSequenceCorpus:
    """Streaming OTTO session corpus that never materializes the source file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.pass_index = 0
        self.sessions = 0
        self.tokens = 0

    def __iter__(self) -> Iterator[list[int]]:
        self.pass_index += 1
        self.sessions = 0
        self.tokens = 0

        with self.path.open("rb") as handle:
            for line in handle:
                record = orjson.loads(line)
                events = record.get("events")

                if not isinstance(events, list) or not events:
                    continue

                sequence = [int(event["aid"]) for event in events]
                self.sessions += 1
                self.tokens += len(sequence)
                yield sequence

    def progress(self) -> dict[str, int]:
        return {
            "pass_index": self.pass_index,
            "sessions": self.sessions,
            "events": self.tokens,
        }


def _gib(value: int) -> float:
    return value / (1024**3)


def validate_training_resources() -> None:
    memory = psutil.virtual_memory()
    total_gib = _gib(memory.total)
    available_gib = _gib(memory.available)

    if total_gib < _MIN_TOTAL_RAM_GIB:
        raise RuntimeError(
            f"Item2Vec training requires at least {_MIN_TOTAL_RAM_GIB:.0f} GiB "
            f"host RAM; observed {total_gib:.1f} GiB."
        )

    if available_gib < _MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError(
            f"Item2Vec training requires at least {_MIN_AVAILABLE_RAM_GIB:.0f} GiB "
            f"available RAM; observed {available_gib:.1f} GiB."
        )


def _family_hash(prefix: Path) -> str:
    files = sorted(
        path
        for path in prefix.parent.glob(prefix.name + "*")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(prefix)

    payload = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return canonical_json_sha256(payload)


def _validation_manifest_id(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_id = payload.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("validation manifest does not contain manifest_id")
    return manifest_id


class EpochLogger(CallbackAny2Vec):
    """Emit explicit epoch timing and loss telemetry."""

    def __init__(self, logger: logging.Logger, epochs: int) -> None:
        self.logger = logger
        self.epochs = epochs
        self.epoch = 0
        self._epoch_started = 0.0
        self._previous_loss = 0.0

    def on_epoch_begin(self, model: Word2Vec) -> None:
        del model
        self._epoch_started = time.perf_counter()
        self.logger.info(
            "item2vec_epoch_start",
            extra={
                "event": "item2vec_epoch_start",
                "stage": "item2vec_train",
                "epoch": self.epoch + 1,
                "epochs": self.epochs,
            },
        )

    def on_epoch_end(self, model: Word2Vec) -> None:
        cumulative_loss = float(model.get_latest_training_loss())
        epoch_loss = cumulative_loss - self._previous_loss
        self._previous_loss = cumulative_loss
        elapsed = round(time.perf_counter() - self._epoch_started, 3)

        self.logger.info(
            "item2vec_epoch_complete",
            extra={
                "event": "item2vec_epoch_complete",
                "stage": "item2vec_train",
                "status": "passed",
                "epoch": self.epoch + 1,
                "epochs": self.epochs,
                "epoch_loss": round(epoch_loss, 3),
                "elapsed_seconds": elapsed,
            },
        )
        self.epoch += 1


def train_item2vec(
    train_sessions_path: str | Path,
    validation_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    config: Item2VecConfig,
    heartbeat_seconds: float = 30.0,
) -> Item2VecManifest:
    """Train streaming skip-gram Item2Vec and persist mmap-friendly vectors."""
    validate_training_resources()

    if config.vector_size <= 0:
        raise ValueError("vector_size must be positive")
    if config.window <= 0:
        raise ValueError("window must be positive")
    if config.negative <= 0:
        raise ValueError("negative must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.workers <= 0:
        raise ValueError("workers must be positive")

    source = Path(train_sessions_path).resolve()
    validation_manifest = Path(validation_manifest_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    validation_id = _validation_manifest_id(validation_manifest)
    model_path = destination / "item2vec.model"
    vectors_path = destination / "item_vectors.kv"
    manifest_path = destination / "manifest.json"

    corpus = SessionSequenceCorpus(source)
    started = time.perf_counter()

    logger.info(
        "item2vec_vocab_start",
        extra={
            "event": "item2vec_vocab_start",
            "stage": "item2vec_vocab",
            "vector_size": config.vector_size,
            "window": config.window,
        },
    )

    model = Word2Vec(
        vector_size=config.vector_size,
        window=config.window,
        min_count=1,
        sample=config.sample,
        seed=config.seed,
        workers=config.workers,
        sg=1,
        hs=0,
        negative=config.negative,
        ns_exponent=config.ns_exponent,
        sorted_vocab=1,
        compute_loss=True,
        shrink_windows=True,
    )

    with Heartbeat(
        logger,
        stage="item2vec_vocab",
        interval_seconds=heartbeat_seconds,
        progress_provider=corpus.progress,
    ):
        model.build_vocab(corpus_iterable=corpus)

    if model.corpus_count <= 0 or len(model.wv) <= 0:
        raise RuntimeError("Item2Vec vocabulary is empty")

    logger.info(
        "item2vec_vocab_complete",
        extra={
            "event": "item2vec_vocab_complete",
            "stage": "item2vec_vocab",
            "sessions": int(model.corpus_count),
            "events": int(model.corpus_total_words),
            "vocabulary_size": len(model.wv),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )

    epoch_logger = EpochLogger(logger, config.epochs)

    with Heartbeat(
        logger,
        stage="item2vec_train",
        interval_seconds=heartbeat_seconds,
        progress_provider=corpus.progress,
    ):
        model.train(
            corpus_iterable=corpus,
            total_examples=model.corpus_count,
            epochs=config.epochs,
            callbacks=[epoch_logger],
        )

    logger.info(
        "item2vec_persist_start",
        extra={
            "event": "item2vec_persist_start",
            "stage": "item2vec_persist",
        },
    )

    model.save(str(model_path), sep_limit=10 * 1024 * 1024)
    model.wv.save(str(vectors_path), sep_limit=10 * 1024 * 1024)

    # Verify the mmap representation immediately; FAISS build uses this form.
    vectors = KeyedVectors.load(str(vectors_path), mmap="r")
    if len(vectors) != len(model.wv):
        raise RuntimeError("persisted Item2Vec vocabulary size changed")
    if int(vectors.vector_size) != config.vector_size:
        raise RuntimeError("persisted Item2Vec dimension changed")

    elapsed = round(time.perf_counter() - started, 3)

    manifest = Item2VecManifest(
        validation_manifest_id=validation_id,
        config=config,
        corpus_sessions=int(model.corpus_count),
        corpus_tokens=int(model.corpus_total_words),
        vocabulary_size=len(model.wv),
        vector_size=config.vector_size,
        model_family_sha256=_family_hash(model_path),
        vectors_family_sha256=_family_hash(vectors_path),
        elapsed_seconds=elapsed,
    )

    temp_manifest = destination / ".manifest.json.tmp"
    temp_manifest.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_manifest, manifest_path)

    logger.info(
        "item2vec_complete",
        extra={
            "event": "item2vec_complete",
            "stage": "item2vec_train",
            "status": "passed",
            "sessions": manifest.corpus_sessions,
            "events": manifest.corpus_tokens,
            "elapsed_seconds": elapsed,
            "vocabulary_size": manifest.vocabulary_size,
        },
    )

    return manifest
