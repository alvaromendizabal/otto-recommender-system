from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import OBJECTIVE_TO_ID


@dataclass(frozen=True)
class ItemVocabulary:
    item_ids: np.ndarray
    aid_to_index: np.ndarray
    vectors: np.ndarray
    unknown_index: int
    padding_index: int

    @classmethod
    def load(cls, root: Path) -> ItemVocabulary:
        item_ids = np.load(root / "item_ids.npy", mmap_mode="r")
        vectors = np.load(root / "item_vectors.npy", mmap_mode="r")
        aid_to_index = np.load(root / "aid_to_index.npy", mmap_mode="r")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if item_ids.ndim != 1 or vectors.ndim != 2:
            raise ValueError("invalid item vocabulary arrays")
        if vectors.shape[0] != item_ids.shape[0]:
            raise ValueError("item_ids and item_vectors row counts differ")
        if int(manifest["items"]) != item_ids.shape[0]:
            raise ValueError("item vocabulary manifest row count mismatch")
        unknown_index = int(item_ids.shape[0])
        padding_index = unknown_index + 1
        return cls(
            item_ids=item_ids,
            aid_to_index=aid_to_index,
            vectors=vectors,
            unknown_index=unknown_index,
            padding_index=padding_index,
        )

    def map_aids(self, aids: np.ndarray, *, allow_unknown: bool) -> np.ndarray:
        aids = np.asarray(aids, dtype=np.int64)
        result = np.full(aids.shape, self.unknown_index, dtype=np.int64)
        valid = (aids >= 0) & (aids < self.aid_to_index.shape[0])
        if np.any(valid):
            mapped = np.asarray(self.aid_to_index[aids[valid]], dtype=np.int64)
            known = mapped >= 0
            positions = np.flatnonzero(valid)
            result.flat[positions[known]] = mapped[known]
        if not allow_unknown and np.any(result == self.unknown_index):
            unknown = aids[result == self.unknown_index]
            raise ValueError(f"candidate aids missing from vocabulary: {unknown[:10].tolist()}")
        return result


@dataclass(frozen=True)
class SequenceBatch:
    item_indices: torch.Tensor
    event_types: torch.Tensor
    time_buckets: torch.Tensor
    mask: torch.Tensor


class PackedSessionStore:
    def __init__(
        self,
        *,
        session_ids: np.ndarray,
        offsets: np.ndarray,
        item_indices: np.ndarray,
        event_types: np.ndarray,
        timestamps: np.ndarray,
        max_seq_len: int,
        padding_index: int,
        time_buckets: int,
    ) -> None:
        self.session_ids = session_ids
        self.offsets = offsets
        self.item_indices = item_indices
        self.event_types = event_types
        self.timestamps = timestamps
        self.max_seq_len = max_seq_len
        self.padding_index = padding_index
        self.time_buckets = time_buckets
        self._session_to_row = {
            int(session): index for index, session in enumerate(session_ids.tolist())
        }

    @classmethod
    def from_parquet(
        cls,
        cache_dir: Path,
        vocabulary: ItemVocabulary,
        *,
        max_seq_len: int,
        time_buckets: int,
        selected_sessions: np.ndarray | None = None,
    ) -> PackedSessionStore:
        import pyarrow.parquet as pq

        events = pq.read_table(
            cache_dir / "events.parquet",
            columns=["session", "aid", "ts", "event_type", "event_index"],
            filters=[("session", "in", selected_sessions.tolist())]
            if selected_sessions is not None
            else None,
        )
        event_session = (
            events.column("session").to_numpy(zero_copy_only=False).astype(np.int64)
        )
        event_index = events.column("event_index").to_numpy(zero_copy_only=False).astype(np.int64)
        order = np.lexsort((event_index, event_session))
        event_session = event_session[order]
        aids = events.column("aid").to_numpy(zero_copy_only=False).astype(np.int64)[order]
        timestamps = (
            events.column("ts").to_numpy(zero_copy_only=False).astype(np.int64)[order]
        )
        event_types = (
            events.column("event_type").to_numpy(zero_copy_only=False).astype(np.int64)[order]
        )

        session_ids, counts = np.unique(event_session, return_counts=True)
        offsets = np.zeros(session_ids.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        item_indices = vocabulary.map_aids(aids, allow_unknown=True)
        return cls(
            session_ids=session_ids,
            offsets=offsets,
            item_indices=item_indices,
            event_types=event_types,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            padding_index=vocabulary.padding_index,
            time_buckets=time_buckets,
        )

    def batch(self, session_ids: np.ndarray, device: torch.device) -> SequenceBatch:
        batch_size = int(session_ids.shape[0])
        items = np.full(
            (batch_size, self.max_seq_len),
            self.padding_index,
            dtype=np.int64,
        )
        types = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        times = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        mask = np.zeros((batch_size, self.max_seq_len), dtype=np.bool_)

        for row, session in enumerate(session_ids.tolist()):
            session_row = self._session_to_row.get(int(session))
            if session_row is None:
                raise KeyError(f"session {session} is missing from the packed session store")
            start = int(self.offsets[session_row])
            end = int(self.offsets[session_row + 1])
            start = max(start, end - self.max_seq_len)
            length = end - start
            target_start = self.max_seq_len - length
            items[row, target_start:] = self.item_indices[start:end]
            types[row, target_start:] = self.event_types[start:end]
            mask[row, target_start:] = True

            ts = self.timestamps[start:end]
            last_ts = int(ts[-1])
            delta_seconds = np.maximum((last_ts - ts) // 1000, 0)
            buckets = np.floor(np.log2(delta_seconds + 1)).astype(np.int64)
            buckets = np.minimum(buckets, self.time_buckets - 1)
            times[row, target_start:] = buckets

        return SequenceBatch(
            item_indices=torch.from_numpy(items).to(device=device, non_blocking=True),
            event_types=torch.from_numpy(types).to(device=device, non_blocking=True),
            time_buckets=torch.from_numpy(times).to(device=device, non_blocking=True),
            mask=torch.from_numpy(mask).to(device=device, non_blocking=True),
        )


@dataclass(frozen=True)
class RetrievalBatch:
    session_ids: np.ndarray
    objective_ids: torch.Tensor
    positive_indices: torch.Tensor
    negative_indices: torch.Tensor
    positive_aids: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.session_ids.shape[0])


class HardNegativeBatchStream:
    def __init__(
        self,
        root: Path,
        vocabulary: ItemVocabulary,
        *,
        batch_size: int,
        validation_fold: int,
        seed: int,
    ) -> None:
        self.root = root
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.validation_fold = validation_fold
        self.seed = seed
        self.parts = sorted((root / "parts").glob("part-*.parquet"))
        if not self.parts:
            raise FileNotFoundError(f"no hard-negative parts found under {root / 'parts'}")

    def iter_batches(
        self,
        *,
        epoch: int,
        training: bool,
        device: torch.device,
        max_rows: int | None = None,
        start_batch: int = 0,
    ) -> Iterator[tuple[int, RetrievalBatch]]:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        rng = np.random.default_rng(self.seed + epoch * 1009 + (1 if training else 0))
        parts = list(self.parts)
        if training:
            rng.shuffle(parts)

        emitted_rows = 0
        logical_batch = 0
        for part in parts:
            table = pq.read_table(
                part,
                columns=[
                    "session",
                    "fold",
                    "objective",
                    "positive_aid",
                    "hard_negative_aids",
                ],
            )
            fold = table.column("fold")
            condition = pc.not_equal(fold, self.validation_fold) if training else pc.equal(
                fold, self.validation_fold
            )
            table = table.filter(condition)
            if table.num_rows == 0:
                continue

            indices = np.arange(table.num_rows)
            if training:
                rng.shuffle(indices)

            sessions_all = (
                table.column("session").to_numpy(zero_copy_only=False).astype(np.int64)
            )
            objectives_all = table.column("objective").to_pylist()
            positives_all = (
                table.column("positive_aid").to_numpy(zero_copy_only=False).astype(np.int64)
            )
            negatives_all = table.column("hard_negative_aids").to_pylist()

            for start in range(0, table.num_rows, self.batch_size):
                if max_rows is not None and emitted_rows >= max_rows:
                    return
                selection = indices[start : start + self.batch_size]
                if max_rows is not None:
                    selection = selection[: max_rows - emitted_rows]
                if selection.size == 0:
                    return

                if logical_batch < start_batch:
                    logical_batch += 1
                    emitted_rows += int(selection.size)
                    continue

                sessions = sessions_all[selection]
                objective_ids = np.asarray(
                    [OBJECTIVE_TO_ID[str(objectives_all[index])] for index in selection],
                    dtype=np.int64,
                )
                positive_aids = positives_all[selection]
                positive_indices = self.vocabulary.map_aids(
                    positive_aids,
                    allow_unknown=False,
                )
                negative_aids = np.asarray(
                    [negatives_all[index] for index in selection],
                    dtype=np.int64,
                )
                negative_indices = self.vocabulary.map_aids(
                    negative_aids,
                    allow_unknown=False,
                )
                yield logical_batch, RetrievalBatch(
                    session_ids=sessions,
                    objective_ids=torch.from_numpy(objective_ids).to(
                        device=device,
                        non_blocking=True,
                    ),
                    positive_indices=torch.from_numpy(positive_indices).to(
                        device=device,
                        non_blocking=True,
                    ),
                    negative_indices=torch.from_numpy(negative_indices).to(
                        device=device,
                        non_blocking=True,
                    ),
                    positive_aids=torch.from_numpy(positive_aids).to(
                        device=device,
                        non_blocking=True,
                    ),
                )
                logical_batch += 1
                emitted_rows += int(selection.size)


def writable_vectors(vectors: np.ndarray) -> torch.Tensor:
    """Own NumPy storage before exposing it to a mutable tensor."""
    return torch.from_numpy(np.array(vectors, dtype=np.float32, copy=True))
