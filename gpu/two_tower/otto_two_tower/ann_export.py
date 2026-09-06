"""Durable full-fold ANN predictions and official ranking metrics."""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .ann_search import build_index, search, write_npz
from .benchmark_artifacts import BenchmarkArtifacts
from .catalogue import Catalogue
from .config import ModelConfig
from .data import ItemVocabulary, PackedSessionStore, writable_vectors
from .evaluation import read_json
from .model import TwoTowerModel
from .ranking_metrics import OBJECTIVES, ranking_counts


def load_encoder(
    args: argparse.Namespace, sessions: np.ndarray
) -> tuple[TwoTowerModel, PackedSessionStore, torch.device]:
    training = read_json(args.model_dir / "training_manifest.json")
    config = ModelConfig(**training["config"]["model"])
    sequence_length = training["config"]["data"]["max_seq_len"]
    vocabulary = ItemVocabulary.load(args.item_data)
    store = PackedSessionStore.from_parquet(
        args.ranking_cache,
        vocabulary,
        max_seq_len=sequence_length,
        time_buckets=config.time_buckets,
        selected_sessions=sessions,
    )
    model = TwoTowerModel(
        writable_vectors(vocabulary.vectors),
        padding_index=vocabulary.padding_index,
        config=config,
        max_seq_len=sequence_length,
    )
    model.load_state_dict(
        torch.load(args.model_dir / "best_model.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, store, device


def export_fold(
    args: argparse.Namespace,
    reference: dict[str, Any],
    sessions: np.ndarray,
    catalogue: Catalogue,
    chosen: int,
    truth: dict[str, dict[int, set[int]]],
    artifacts: BenchmarkArtifacts,
    progress: dict[str, Any],
    load_vectors: Callable[[str], np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    buckets = int(reference["ranking_manifest"]["config"]["buckets"])
    all_counts = np.zeros((len(sessions), 3, 7))
    model, store, device = load_encoder(args, sessions)
    receipts = []
    started = time.perf_counter()
    for objective_id, objective in enumerate(OBJECTIVES):
        vectors = load_vectors(objective)
        index, _ = build_index(vectors, catalogue.item_ids, args, artifacts, objective)
        index.nprobe = chosen
        for bucket in range(buckets):
            subset = sessions[sessions % buckets == bucket]
            progress.update(
                stage="full_fold_ann_export", objective=objective, bucket=bucket, examples=0
            )
            name = f"prediction_export/predictions/{objective}/part-{bucket:03d}.parquet"

            def predict(
                path: Path,
                subset: np.ndarray = subset,
                objective_id: int = objective_id,
                vectors: np.ndarray = vectors,
                index: Any = index,
                model: TwoTowerModel = model,
                store: PackedSessionStore = store,
                device: torch.device = device,
            ) -> None:
                predictions, similarities = [], []
                with torch.inference_mode():
                    for begin in range(0, len(subset), args.batch_size):
                        batch = subset[begin : begin + args.batch_size]
                        query = (
                            model.encode_session(
                                store.batch(batch, device),
                                torch.full((len(batch),), objective_id, device=device),
                            )
                            .float()
                            .cpu()
                            .numpy()
                        )
                        scores, aids = search(
                            index, query, vectors, catalogue, args.candidate_depth
                        )
                        predictions.extend(aids.astype(np.int32).tolist())
                        similarities.extend(scores.tolist())
                        progress.update(examples=begin + len(batch))
                pq.write_table(
                    pa.table(
                        {
                            "session": pa.array(subset),
                            "aids": pa.array(predictions, type=pa.list_(pa.int32())),
                            "scores": pa.array(similarities, type=pa.list_(pa.float32())),
                        }
                    ),
                    path,
                    compression="zstd",
                )

            part = artifacts.produce(name, predict)
            table = pq.read_table(part)
            if not np.array_equal(table["session"].to_numpy(), subset):
                raise ValueError("full-fold prediction checkpoint session mismatch")
            rows = table["aids"].to_pylist()
            if any(len(row) != args.candidate_depth for row in rows):
                raise ValueError("full-fold prediction checkpoint depth mismatch")
            values = np.array(
                [
                    ranking_counts(truth[objective].get(int(s), set()), row[:20])
                    for s, row in zip(subset, rows, strict=True)
                ]
            ).reshape(-1, 7)
            all_counts[np.searchsorted(sessions, subset), objective_id] = values
            counts_name = f"prediction_export/counts/{objective}/part-{bucket:03d}.npz"

            def save_counts(
                path: Path, subset: np.ndarray = subset, values: np.ndarray = values
            ) -> None:
                write_npz(path, sessions=subset, counts=values)

            artifacts.produce(counts_name, save_counts)
            receipts.append(
                {
                    "path": name.removeprefix("prediction_export/"),
                    **artifacts.used[name],
                    "rows": len(subset),
                }
            )
        del vectors, index
        gc.collect()
    del predict, model, store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = artifacts.json(
        "prediction_export/prediction_manifest.json",
        lambda: {
            "schema_version": 1,
            "status": "passed",
            "input_id": args.run_id,
            "training_input_id": reference["training_input_id"],
            "code_commit": args.code_commit,
            "validation_fold": reference["validation_fold"],
            "ranking_manifest": reference["ranking_manifest"],
            "reference_input_id": reference["input_id"],
            "sessions": len(sessions),
            "catalogue_items": len(catalogue.item_ids),
            "parts": receipts,
            "search": {
                "method": "faiss_ivfflat",
                "nlist": args.nlist,
                "nprobe": chosen,
                "k": args.candidate_depth,
                "dtype": "float32",
                "tie_break": "ascending aid within returned candidate pool",
            },
            "elapsed_seconds_this_attempt": time.perf_counter() - started,
        },
    )
    return manifest, all_counts
