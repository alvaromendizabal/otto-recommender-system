"""Cutoff-safe Item2Vec representations and observed-session similarity features."""

from __future__ import annotations

import json
import logging
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from gensim.models import Word2Vec  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.research.features import Prefix
from otto_recsys.research.protocol import atomic_json
from otto_recsys.runtime import Heartbeat

FAMILIES = ("all", "intent")
WINDOWS = (5, 20, 50)
STATISTICS = ("coverage", "last", "mean", "max", "centroid", "recency")
ACTIONS = ("all", "clicks", "carts", "orders")


def stable_hash(value: str) -> int:
    return zlib.adler32(value.encode("utf-8"))


def feature_names(family: str) -> tuple[str, ...]:
    if family not in FAMILIES:
        raise ValueError("unknown representation family")
    return (
        f"embedding_{family}_known",
        f"embedding_{family}_log_frequency",
        *(
            f"embedding_{family}_{action}_n{window}_{statistic}"
            for action in ACTIONS
            for window in WINDOWS
            for statistic in STATISTICS
        ),
    )


def session_sequences(batches: Iterable[Any]) -> Iterator[tuple[list[str], list[str]]]:
    """Keep at most 100 tokens per family, including sessions spanning Arrow batches."""
    current = -1
    all_actions: deque[str] = deque(maxlen=100)
    intent: deque[str] = deque(maxlen=100)
    for batch in batches:
        sessions = batch.column("session").to_numpy()
        aids = batch.column("aid").to_numpy()
        kinds = batch.column("event_type").to_numpy()
        if not sessions.size:
            continue
        if sessions[0] < current or (np.diff(sessions) < 0).any():
            raise ValueError("embedding history stream must be sorted by session")
        starts = np.r_[0, np.flatnonzero(np.diff(sessions)) + 1]
        stops = np.r_[starts[1:], sessions.size]
        for start, stop in zip(starts, stops, strict=True):
            session = int(sessions[start])
            if session != current:
                if all_actions:
                    yield list(all_actions), list(intent)
                current = session
                all_actions.clear()
                intent.clear()
            # Discard long-session prefixes before string conversion; both buffers stay bounded.
            all_actions.extend(str(int(v)) for v in aids[max(start, stop - 100) : stop])
            selected = aids[start:stop][kinds[start:stop] > 0]
            intent.extend(str(int(v)) for v in selected[-100:])
    if all_actions:
        yield list(all_actions), list(intent)


def prepare_sequences(
    history: Path,
    output: Path,
    *,
    cutoff: int,
    expected_sha256: str,
    threads: int,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Only historical events enter either corpus; reject rather than silently trim leaks."""
    if sha256_file(history) != expected_sha256:
        raise ValueError("embedding history checksum mismatch")
    contract = {
        "history_sha256": expected_sha256,
        "cutoff_exclusive_ms": cutoff,
        "maximum_events_per_session": 100,
        "families": {"all": [0, 1, 2], "intent": [1, 2]},
        "preparation": "external sort, Arrow batches, bounded per-session deques",
        "sort_memory_gb": 8,
        "sort_threads": min(4, int(threads)),
        "code_sha256": sha256_file(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["contract"] != contract:
            raise ValueError("sequence workspace belongs to a different history")
        if all(
            (output / name).is_file() and sha256_file(output / name) == digest
            for name, digest in saved["files"].items()
        ):
            return saved
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={min(4, int(threads))}")
        connection.execute("SET memory_limit='8GB'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET temp_directory=?", [str(output / "temporary")])
        summary = connection.execute(
            "SELECT min(ts), max(ts), count(*) FROM read_parquet(?)", [str(history)]
        ).fetchone()
        if summary is None or summary[2] == 0 or summary[1] >= cutoff:
            raise ValueError("embedding training must contain events strictly before history_end")
        progress = {"sessions": 0, "history_events": int(summary[2])}
        with (
            Heartbeat(
                logger,
                stage="embedding_sequences",
                interval_seconds=15,
                progress_provider=progress.copy,
            ),
            (output / "all.tmp").open("w", buffering=1024**2) as all_stream,
            (output / "intent.tmp").open("w", buffering=1024**2) as intent_stream,
        ):
            reader = connection.execute(
                "SELECT session, aid, event_type FROM read_parquet(?) "
                "ORDER BY session, event_index",
                [str(history)],
            ).to_arrow_reader(batch_size=262144)
            for all_actions, intent in session_sequences(reader):
                all_stream.write(" ".join(all_actions) + "\n")
                if intent:
                    intent_stream.write(" ".join(intent) + "\n")
                progress["sessions"] += 1
        for family in FAMILIES:
            destination = output / f"{family}.txt"
            (output / f"{family}.tmp").replace(destination)
            if publish:
                publish(destination)
        result = {
            "contract": contract,
            "history_events": int(summary[2]),
            "history_max_ts": int(summary[1]),
            "sessions": progress["sessions"],
            "files": {
                f"{family}.txt": sha256_file(output / f"{family}.txt") for family in FAMILIES
            },
        }
        atomic_json(path, result)
        if publish:
            publish(path)
        return result
    finally:
        connection.close()


def train_representation(
    corpus: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Recover verified native optimizer state at epoch boundaries; pin realized vectors."""
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "corpus_sha256": sha256_file(corpus),
        "config": config,
        "code_sha256": sha256_file(Path(__file__)),
        "reproducibility": (
            "pinned artifact hashes; multithreaded Word2Vec is not bit deterministic"
        ),
    }
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("representation checkpoint contract differs")
    atomic_json(contract_path, contract)
    if publish:
        publish(contract_path)
    input_id = canonical_json_sha256(contract)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if (
            saved["input_id"] == input_id
            and (output / "vectors.npz").is_file()
            and sha256_file(output / "vectors.npz") == saved["sha256"]
        ):
            return saved
    if any(int(config[k]) < 1 for k in ("dimensions", "window", "negative", "epochs", "workers")):
        raise ValueError("representation dimensions and resource settings must be positive")
    checkpoint = output / "epochs"
    checkpoint.mkdir(exist_ok=True)
    model = None
    completed = 0
    retained_seconds = 0.0
    for receipt in sorted(checkpoint.glob("*.json"), reverse=True):
        saved = json.loads(receipt.read_text())
        native = receipt.with_suffix(".pkl.gz")
        if (
            saved["input_id"] == input_id
            and native.is_file()
            and sha256_file(native) == saved["sha256"]
        ):
            # Only our own byte-verified native checkpoint is deserialized.
            model = Word2Vec.load(str(native))
            completed = int(saved["epoch"])
            retained_seconds = float(saved["retained_seconds"])
            break
    started = time.perf_counter()
    if model is None:
        model = Word2Vec(
            vector_size=int(config["dimensions"]),
            window=int(config["window"]),
            negative=int(config["negative"]),
            workers=int(config["workers"]),
            seed=int(config["seed"]),
            sg=1,
            hs=0,
            min_count=1,
            sample=1e-4,
            ns_exponent=0.75,
            sorted_vocab=1,
            shrink_windows=True,
            hashfxn=stable_hash,
        )
        with Heartbeat(logger, stage=f"embedding_vocabulary_{corpus.stem}", interval_seconds=15):
            model.build_vocab(corpus_file=str(corpus))
    epochs = int(config["epochs"])
    for epoch in range(completed, epochs):
        with Heartbeat(
            logger, stage=f"embedding_{corpus.stem}_epoch_{epoch + 1}", interval_seconds=15
        ):
            model.train(
                corpus_file=str(corpus),
                total_words=model.corpus_total_words,
                total_examples=model.corpus_count,
                epochs=1,
                start_alpha=0.025 - 0.0249 * epoch / epochs,
                end_alpha=0.025 - 0.0249 * (epoch + 1) / epochs,
            )
            native = checkpoint / f"{epoch + 1:04d}.pkl.gz"
            temporary = checkpoint / f"{epoch + 1:04d}.tmp.gz"
            model.save(str(temporary), separately=[])
            temporary.replace(native)
            receipt = checkpoint / f"{epoch + 1:04d}.json"
            atomic_json(
                receipt,
                {
                    "input_id": input_id,
                    "epoch": epoch + 1,
                    "sha256": sha256_file(native),
                    "retained_seconds": retained_seconds + time.perf_counter() - started,
                },
            )
            if publish:
                publish(native)
                publish(receipt)
        logger.info("embedding_epoch_complete", extra={"family": corpus.stem, "epoch": epoch + 1})
    ids = np.asarray([int(k) for k in model.wv.index_to_key], dtype=np.int64)
    order = np.argsort(ids)
    vectors = np.asarray(model.wv.vectors[order], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    counts = np.asarray([model.wv.get_vecattr(str(k), "count") for k in ids[order]], dtype=np.int64)
    destination = output / "vectors.npz"
    temporary = output / "vectors.tmp.npz"
    np.savez(temporary, ids=ids[order], vectors=vectors, counts=counts)
    temporary.replace(destination)
    result = {
        "input_id": input_id,
        "sha256": sha256_file(destination),
        "status": "passed",
        "vocabulary": int(ids.size),
        "dimensions": int(config["dimensions"]),
        "epochs": epochs,
        "sessions": int(model.corpus_count),
        "tokens": int(model.corpus_total_words),
        "retained_seconds": retained_seconds + time.perf_counter() - started,
    }
    atomic_json(manifest_path, result)
    if publish:
        publish(destination)
        publish(manifest_path)
    return result


class Representation:
    def __init__(self, path: Path) -> None:
        receipt = json.loads(path.with_name("manifest.json").read_text())
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError("representation vectors failed checksum verification")
        with np.load(path, allow_pickle=False) as data:
            self.ids = data["ids"]
            self.vectors = data["vectors"]
            self.counts = data["counts"]
        if (
            self.ids.ndim != 1
            or not self.ids.size
            or (np.diff(self.ids) <= 0).any()
            or self.counts.shape != self.ids.shape
            or self.vectors.shape[0] != self.ids.size
            or not np.isfinite(self.vectors).all()
        ):
            raise ValueError("representation catalogue must be sorted, unique and finite")

    def lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        locations = np.searchsorted(self.ids, ids)
        locations = np.minimum(locations, self.ids.size - 1)
        known = self.ids[locations] == ids
        vectors = self.vectors[locations].copy()
        vectors[~known] = 0
        counts = np.where(known, self.counts[locations], 0)
        return vectors, known, counts

    def transform(self, prefix: Prefix, candidates: np.ndarray) -> np.ndarray:
        """No candidate-set normalization: fitting negative samples cannot alter features."""
        prefix.validate()
        vectors, known, counts = self.lookup(candidates)
        observed, observed_known, _ = self.lookup(prefix.aid[-50:])
        kinds = prefix.kind[-50:]
        times = prefix.ts[-50:]
        similarities = vectors @ observed.T
        columns = [known.astype(np.float32), np.log1p(counts).astype(np.float32)]
        for action_index, action in enumerate(ACTIONS):
            for window in WINDOWS:
                selected = np.arange(max(0, observed.shape[0] - window), observed.shape[0])
                if action != "all":
                    selected = selected[kinds[selected] == action_index - 1]
                valid = selected[observed_known[selected]]
                coverage = float(valid.size / selected.size) if selected.size else 0.0
                zeros = np.zeros(candidates.size, dtype=np.float32)
                if not valid.size:
                    columns.extend([np.full_like(zeros, coverage), *[zeros] * 5])
                    continue
                values = similarities[:, valid]
                centroid = observed[valid].mean(axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                weights = np.exp(-np.maximum(0, times[-1] - times[valid]) / 21_600_000)
                weights /= weights.sum()
                columns.extend(
                    [
                        np.full_like(zeros, coverage),
                        similarities[:, selected[-1]],
                        values.mean(axis=1),
                        values.max(axis=1),
                        vectors @ centroid,
                        values @ weights.astype(np.float32),
                    ]
                )
        return np.column_stack(columns).astype(np.float32)
