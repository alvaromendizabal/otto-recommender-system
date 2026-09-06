"""Managed full-catalogue ANN experiment over a frozen two-tower checkpoint."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch

from .ann_export import export_fold, load_encoder
from .ann_search import build_index, evaluate_queries, latency, write_npz
from .benchmark_artifacts import BenchmarkArtifacts
from .catalogue import Catalogue
from .data import ItemVocabulary
from .evaluation import identity, read_json, sha256_file, verified_part
from .ranking_metrics import OBJECTIVES, paired_recall_interval, ranking_counts, summarize_ranking


def _check_inputs(args: argparse.Namespace, progress: dict[str, Any]) -> dict[str, Any]:
    path = args.reference_dir / "prediction_manifest.json"
    if sha256_file(path) != args.reference_manifest_sha256:
        raise ValueError("reference manifest checksum mismatch")
    reference = read_json(path)
    if reference["status"] != "passed" or reference["input_id"] != args.reference_input_id:
        raise ValueError("reference prediction identity mismatch")
    if reference["search"]["k"] < args.candidate_depth:
        raise ValueError("saved reference does not cover requested candidate depth")
    if reference["search"]["method"] != "exhaustive_inner_product":
        raise ValueError("ANN fidelity requires an exact reference")
    ranking = read_json(args.ranking_cache / "manifest.json")
    training = read_json(args.model_dir / "training_manifest.json")
    items = read_json(args.item_data / "manifest.json")
    if (
        ranking != reference["ranking_manifest"]
        or training["input_id"] != reference["training_input_id"]
        or training["validation_fold"] != reference["validation_fold"]
    ):
        raise ValueError("training/ranking reference mismatch")
    export_contract = read_json(args.reference_dir / "evaluation_contract.json")
    if (
        identity(export_contract) != reference["input_id"]
        or export_contract["item_manifest"] != items
    ):
        raise ValueError("export contract mismatch")
    for root, manifest, suffix, names in (
        (args.ranking_cache, ranking, ".parquet", ("events", "examples", "labels")),
        (args.item_data, items, ".npy", ("item_ids", "item_vectors", "aid_to_index")),
    ):
        for name in names:
            progress.update(stage="verify_input", file=name)
            if sha256_file(root / (name + suffix)) != manifest[name + "_sha256"]:
                raise ValueError(f"input checksum mismatch: {name}")
    if sha256_file(args.model_dir / "best_model.pt") != reference["model_sha256"]:
        raise ValueError("saved model checksum mismatch")
    return reference


def _reference_counts(
    args: argparse.Namespace,
    reference: dict[str, Any],
    sessions: np.ndarray,
    selected: np.ndarray,
    truth: dict[str, dict[int, set[int]]],
    artifacts: BenchmarkArtifacts,
    progress: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    counts = np.zeros((len(sessions), 3, 7))
    selected_ids, selected_scores = {}, {}
    bucket_count = reference["ranking_manifest"]["config"]["buckets"]
    receipts = {row["path"]: row for row in reference["parts"]}
    if len(receipts) != 3 * bucket_count:
        raise ValueError("incomplete reference prediction manifest")
    position = {int(s): i for i, s in enumerate(selected)}
    for i, objective in enumerate(OBJECTIVES):
        selected_ids[objective] = np.full((len(selected), args.candidate_depth), -1, np.int64)
        selected_scores[objective] = np.empty((len(selected), 20), np.float32)
        for bucket in range(bucket_count):
            progress.update(stage="official_reference_metrics", objective=objective, bucket=bucket)
            relative = f"predictions/{objective}/part-{bucket:03d}.parquet"
            path = args.reference_dir / relative
            if sha256_file(path) != receipts[relative]["sha256"]:
                raise ValueError("reference prediction checksum mismatch")
            table = pq.read_table(path)
            actual = table["session"].to_numpy()
            expected = sessions[sessions % bucket_count == bucket]
            if not np.array_equal(actual, expected):
                raise ValueError("reference session coverage mismatch")
            aids = table["aids"].combine_chunks()
            scores = table["scores"].combine_chunks()
            k = reference["search"]["k"]
            if np.any(np.diff(aids.offsets.to_numpy()) != k):
                raise ValueError("reference depth mismatch")
            matrix = aids.values.to_numpy().reshape(len(actual), k)
            similarity = scores.values.to_numpy().reshape(len(actual), k)

            def save(
                part: Path,
                actual: np.ndarray = actual,
                matrix: np.ndarray = matrix,
                objective: str = objective,
            ) -> None:
                values = np.array(
                    [
                        ranking_counts(truth[objective].get(int(s), set()), row[:20].tolist())
                        for s, row in zip(actual, matrix, strict=True)
                    ]
                ).reshape(-1, 7)
                write_npz(part, sessions=actual, counts=values)

            part = artifacts.produce(f"reference/{objective}/part-{bucket:03d}.npz", save)
            with np.load(part, allow_pickle=False) as payload:
                if not np.array_equal(payload["sessions"], actual):
                    raise ValueError("reference metric checkpoint misaligned")
                counts[np.searchsorted(sessions, actual), i] = payload["counts"]
            for row in np.flatnonzero(np.isin(actual, selected)):
                idx = position[int(actual[row])]
                selected_ids[objective][idx] = matrix[row, : args.candidate_depth]
                selected_scores[objective][idx] = similarity[row, :20]
        if np.any(selected_ids[objective] < 0):
            raise ValueError("reference sample is incomplete")
    return counts, selected_ids, selected_scores


def _embeddings(
    args: argparse.Namespace, reference: dict[str, Any], objective: str, progress: dict[str, Any]
) -> np.ndarray:
    shape = (reference["catalogue_items"], reference["model_config"]["embedding_dim"])
    cache = args.output_dir / "cache" / f"{objective}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=shape)
    chunk = reference["search"]["chunk_size"]
    for start in range(0, shape[0], chunk):
        progress.update(stage="verify_embeddings", objective=objective, examples=start)
        path = args.reference_dir / f"embeddings/{objective}/part-{start:08d}.npy"
        receipt = verified_part(path, reference["input_id"])
        if receipt is None:
            raise ValueError("saved candidate embedding checksum mismatch")
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        end = min(start + chunk, shape[0])
        if value.shape != (end - start, shape[1]) or not np.isfinite(value).all():
            raise ValueError("invalid saved candidate embeddings")
        matrix[start:end] = value
    matrix.flush()
    return matrix


def run_benchmark(
    args: argparse.Namespace,
    artifacts: BenchmarkArtifacts,
    logger: logging.Logger,
    progress: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("managed benchmark requires CUDA for query encoding")
    faiss.omp_set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = _check_inputs(args, progress)
    vocabulary = ItemVocabulary.load(args.item_data)
    catalogue = Catalogue(vocabulary.item_ids, vocabulary.aid_to_index)
    item_ids = catalogue.item_ids
    logger.info("catalogue_validated", extra={"examples": len(item_ids)})
    fold = reference["validation_fold"]
    table = pq.read_table(
        args.ranking_cache / "examples.parquet", filters=[("fold", "=", fold)], columns=["session"]
    )
    sessions = np.sort(table["session"].to_numpy().astype(np.int64))
    if (
        len(sessions) != reference["sessions"]
        or len(np.unique(sessions)) != len(sessions)
        or args.sample_sessions > len(sessions)
    ):
        raise ValueError("invalid benchmark cohort size")
    shuffled = np.random.default_rng(args.seed).permutation(sessions)[: args.sample_sessions]
    half = len(shuffled) // 2
    tune, confirm = np.sort(shuffled[:half]), np.sort(shuffled[half:])
    selected = np.concatenate((tune, confirm))
    settings = {
        k: v
        for k, v in vars(args).items()
        if k
        not in {
            "ranking_cache",
            "item_data",
            "model_dir",
            "reference_dir",
            "output_dir",
            "checkpoint_uri",
            "heartbeat_seconds",
            "region",
            "probes",
        }
    }
    settings["probes"] = list(args.probes)
    contract = {
        "schema_version": 1,
        "settings": settings,
        "cohort_sha256": identity({"tuning": tune.tolist(), "confirmation": confirm.tolist()}),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "faiss": faiss.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cuda_runtime": torch.version.cuda,
        "encoder_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "instance_type": os.environ.get("OTTO_INSTANCE_TYPE", "local-contract-test"),
    }
    saved_contract = artifacts.json("contract.json", lambda: contract)
    if saved_contract != contract:
        raise ValueError("benchmark runtime or configuration changed; use a new run identity")
    artifacts.json(
        "cohort.json", lambda: {"tuning": tune.tolist(), "confirmation": confirm.tolist()}
    )
    completed = artifacts.get("metrics.json")
    if completed is not None:
        logger.info("benchmark_result_reused")
        return read_json(completed)
    truth: dict[str, dict[int, set[int]]] = {o: {} for o in OBJECTIVES}
    labels = pq.read_table(
        args.ranking_cache / "labels.parquet",
        filters=[("fold", "=", fold)],
        columns=["session", "objective", "aid"],
    )
    for row in labels.to_pylist():
        truth[row["objective"]].setdefault(int(row["session"]), set()).add(int(row["aid"]))
    full_counts, exact_ids, exact_scores = _reference_counts(
        args, reference, sessions, selected, truth, artifacts, progress
    )
    exact_counts = full_counts[np.searchsorted(sessions, selected)]
    # Fail before indexing if the frozen random sample cannot support every metric.
    summarize_ranking(exact_counts[:half])
    summarize_ranking(exact_counts[half:])
    query_paths = {o: artifacts.get(f"queries/{o}.npz") for o in OBJECTIVES}
    if any(path is None for path in query_paths.values()):
        model, store, device = load_encoder(args, selected)
        for i, objective in enumerate(OBJECTIVES):
            progress.update(stage="encode_queries", objective=objective)

            def encode(path: Path, i: int = i) -> None:
                chunks = []
                with torch.inference_mode():
                    for begin in range(0, len(selected), args.batch_size):
                        batch = selected[begin : begin + args.batch_size]
                        chunks.append(
                            model.encode_session(
                                store.batch(batch, device),
                                torch.full((len(batch),), i, device=device),
                            )
                            .float()
                            .cpu()
                            .numpy()
                        )
                        progress.update(examples=begin + len(batch))
                write_npz(path, sessions=selected, embeddings=np.concatenate(chunks))

            query_paths[objective] = artifacts.produce(f"queries/{objective}.npz", encode)
        del model, store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tuning: dict[int, dict[str, Any]] = {p: {} for p in args.probes}
    confirmation: dict[str, Any] = {}
    builds, exact_latency = {}, {}
    # Evaluate tuning settings first. Confirmation runs only after selection.
    for objective in OBJECTIVES:
        vectors = _embeddings(args, reference, objective, progress)
        query_path = query_paths[objective]
        if query_path is None:
            raise RuntimeError("query encoding checkpoint is missing")
        with np.load(query_path, allow_pickle=False) as payload:
            if not np.array_equal(payload["sessions"], selected):
                raise ValueError("saved query/session alignment mismatch")
            queries = payload["embeddings"]
        if queries.shape != (len(selected), vectors.shape[1]) or not np.isfinite(queries).all():
            raise ValueError("invalid query embeddings")
        positions = catalogue.rows(exact_ids[objective][:, :20])
        replay = np.einsum("nd,nkd->nk", queries, vectors[positions], optimize=False)
        if not np.allclose(replay, exact_scores[objective], atol=2e-5, rtol=0):
            raise ValueError("query/candidate scores do not reproduce the saved exact reference")
        progress.update(stage="build_index", objective=objective)
        index, builds[objective] = build_index(vectors, item_ids, args, artifacts, objective)

        def exact_timing(
            vectors: np.ndarray = vectors, queries: np.ndarray = queries
        ) -> dict[str, Any]:
            flat = faiss.IndexFlatIP(vectors.shape[1])
            flat.add(vectors)
            return latency(flat, queries[:half], vectors, catalogue, args, positional=True)

        exact_latency[objective] = artifacts.json(
            f"reference/{objective}/cpu_latency.json", exact_timing
        )
        for probe in args.probes:
            index.nprobe = probe
            progress.update(stage=f"tuning_nprobe_{probe}", objective=objective)
            tuning[probe][objective] = evaluate_queries(
                index,
                vectors,
                catalogue,
                tune,
                queries[:half],
                exact_ids[objective][:half],
                truth[objective],
                args,
                artifacts,
                f"tuning/probe-{probe}/{objective}",
            )
        del vectors, index
        gc.collect()
    eligible = [
        p
        for p in args.probes
        if all(
            tuning[p][o]["fidelity"][str(args.candidate_depth)] >= args.target_overlap
            for o in OBJECTIVES
        )
    ]
    chosen = min(eligible) if eligible else None
    artifacts.json(
        "selection.json",
        lambda: {
            "selected_nprobe": chosen,
            "criterion": (
                "smallest nprobe meeting strict top-K overlap target for every tuning objective"
            ),
            "target_overlap": args.target_overlap,
            "confirmation_used_for_selection": False,
        },
    )
    if chosen is not None:
        for objective in OBJECTIVES:
            vectors = _embeddings(args, reference, objective, progress)
            index, _ = build_index(vectors, item_ids, args, artifacts, objective)
            index.nprobe = chosen
            query_path = query_paths[objective]
            if query_path is None:
                raise RuntimeError("query encoding checkpoint is missing")
            with np.load(query_path, allow_pickle=False) as payload:
                queries = payload["embeddings"][half:]
            progress.update(stage="confirmation", objective=objective)
            confirmation[objective] = evaluate_queries(
                index,
                vectors,
                catalogue,
                confirm,
                queries,
                exact_ids[objective][half:],
                truth[objective],
                args,
                artifacts,
                f"confirmation/probe-{chosen}/{objective}",
            )
            del vectors, index
            gc.collect()

    def summarize(rows: dict[str, Any], expected: np.ndarray) -> dict[str, Any]:
        actual = np.stack([rows[o]["ranking_counts"] for o in OBJECTIVES], axis=1)
        return {
            "ranking": summarize_ranking(actual),
            "exact_ranking": summarize_ranking(expected),
            "paired_uncertainty": paired_recall_interval(expected, actual, seed=args.seed),
            "search": {
                o: {k: v for k, v in rows[o].items() if k != "ranking_counts"} for o in OBJECTIVES
            },
        }

    tuning_report = {str(p): summarize(tuning[p], exact_counts[:half]) for p in args.probes}
    confirmed = summarize(confirmation, exact_counts[half:]) if confirmation else None
    accepted = chosen is not None and all(
        confirmation[o]["fidelity"][str(args.candidate_depth)] >= args.target_overlap
        for o in OBJECTIVES
    )
    full_ann_ranking, full_ann_interval, prediction_export = None, None, None
    if chosen is not None and accepted and args.export_fold_predictions == "true":
        manifest, ann_counts = export_fold(
            args,
            reference,
            sessions,
            catalogue,
            chosen,
            truth,
            artifacts,
            progress,
            lambda objective: _embeddings(args, reference, objective, progress),
        )
        full_ann_ranking = summarize_ranking(ann_counts)
        full_ann_interval = paired_recall_interval(full_counts, ann_counts, seed=args.seed)
        prediction_export = {
            "input_id": manifest["input_id"],
            "sessions": manifest["sessions"],
            "parts": len(manifest["parts"]),
            "manifest": "prediction_export/prediction_manifest.json",
        }
    result = {
        "schema_version": 1,
        "status": "passed",
        "input_id": args.run_id,
        "code_commit": args.code_commit,
        "reference_input_id": args.reference_input_id,
        "validation_fold": fold,
        "contract": contract,
        "catalogue_items": len(item_ids),
        "full_reference_ranking": summarize_ranking(full_counts),
        "full_ann_ranking": full_ann_ranking,
        "full_ann_paired_uncertainty": full_ann_interval,
        "prediction_export": prediction_export,
        "tuning_sessions": len(tune),
        "confirmation_sessions": len(confirm),
        "tuning": tuning_report,
        "confirmation": confirmed,
        "selected_nprobe": chosen,
        "confirmation_fidelity_passed": accepted,
        "selection_rule": "smallest tuning nprobe with per-objective top-K overlap >= target",
        "index_builds": builds,
        "exact_cpu_latency_on_tuning_queries": exact_latency,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_seconds_this_attempt": round(time.perf_counter() - started, 3),
        "retained_artifact_compute_seconds": sum(
            float(r["elapsed_seconds"]) for r in artifacts.used.values()
        ),
        "metric_notes": (
            "Recall@20 uses official OTTO weighting; candidate ceilings and ANN overlap "
            "are separate."
        ),
        "latency_notes": (
            "CPU search with precomputed queries; batch-1 percentiles exclude "
            "encoder/network/loading. GPU only encodes queries."
        ),
        "generalization": (
            "Fold 0 selected the model checkpoint; exploratory validation. ANN "
            "confirmation is disjoint from ANN tuning, not an untouched model test."
        ),
        "base_union_status": (
            "ANN incremental gain over the frozen base remains pending; exact-positive "
            "retention is not base-exclusive-positive retention."
        ),
        "tie_policy": (
            "FP32 rerank returned candidates by score then aid; unreturned boundary ties "
            "can affect strict ID overlap."
        ),
        "next_decision": (
            "Review fidelity, official metric change and latency before scaling; no "
            "additional folds launched."
        ),
    }
    return artifacts.json("metrics.json", lambda: result)
