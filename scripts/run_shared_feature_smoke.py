"""Bounded early-window reconstruction smoke for the frozen OTTO 102/134 schemas.

This stage never fits a model. It reuses the certified Aug-16 corpus/retrieval,
rebuilds the wider symmetric/forward graphs at the same cutoff, verifies the
existing candidate identities, and materializes complete 400-candidate feature
matrices for 256 fit and 256 selection sessions. A second full replay must be
byte-identical at the matrix level before the result can pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.research.dataset import OBJECTIVES, Queries, sampled_rows
from otto_recsys.research.directed_retrievers import build_retrievers
from otto_recsys.research.domain_features import (
    DOMAIN_FAMILIES,
    NormalizedGraphSignals,
    episode_features,
    funnel_features,
)
from otto_recsys.research.domain_features import feature_names as domain_feature_names
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals
from otto_recsys.research.protocol import atomic_json

FROZEN_HISTORY_END = 1660687200000
SESSION_COUNT = 256
CANDIDATE_BUDGET = 400


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def matrix_digest(aids: np.ndarray, target: np.ndarray, matrix: np.ndarray) -> str:
    payload = b"".join(
        (
            np.asarray(aids, dtype="<i8").tobytes(order="C"),
            np.asarray(target, dtype="<i1").tobytes(order="C"),
            np.asarray(matrix, dtype="<f4").tobytes(order="C"),
        )
    )
    return sha256_bytes(payload)


def selected_sessions(preflight: Path, role: str) -> np.ndarray:
    path = preflight / f"early_{role}_queries.parquet"
    frame = pl.read_parquet(path, columns=["session"]).unique().sort("session")
    ids = frame["session"].to_numpy().astype(np.int64)
    if ids.size != SESSION_COUNT or np.unique(ids).size != SESSION_COUNT:
        raise ValueError(f"{role} smoke cohort must contain exactly {SESSION_COUNT} sessions")
    return ids


def role_indices(queries: Queries, ids: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(queries.session, ids)
    if (
        indices.size != ids.size
        or (indices >= queries.session.size).any()
        or not np.array_equal(queries.session[indices], ids)
    ):
        raise ValueError("frozen smoke sessions differ from the complete query ledger")
    return indices.astype(np.int64)


def cached_metadata(cache: Path, ids: np.ndarray) -> dict[int, pl.DataFrame]:
    paths = sorted(cache.glob("part-*.parquet"))
    if not paths:
        raise ValueError(f"candidate cache has no parts: {cache}")
    frame = (
        pl.scan_parquet([str(path) for path in paths])
        .filter(pl.col("session").is_in(ids.tolist()))
        .select(
            "session",
            "aid",
            "candidate_position",
            *[f"target_{objective}" for objective in OBJECTIVES],
        )
        .collect()
        .sort("session", "candidate_position")
    )
    groups: dict[int, pl.DataFrame] = {}
    for key, group in frame.partition_by("session", as_dict=True).items():
        session = int(key[0] if isinstance(key, tuple) else key)
        groups[session] = group.sort("candidate_position")
    if set(groups) != set(map(int, ids.tolist())):
        raise ValueError("candidate cache is missing one or more frozen smoke sessions")
    return groups


def baseline_fusion(source_ranks: np.ndarray) -> np.ndarray:
    evidence = np.where(source_ranks > 0, 1 / (20 + source_ranks), 0)
    weights = np.array(
        [
            [1.0, 0.1, 0.1, 1.0, 0.01],
            [0.3, 1.0, 0.3, 1.0, 0.01],
            [0.2, 0.5, 1.0, 1.0, 0.01],
        ],
        dtype=np.float32,
    )
    return (evidence @ weights.T).astype(np.float32)


def domain_matrix(
    graphs: NormalizedGraphSignals, prefix: Any, aids: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks = {
        "funnel": funnel_features(prefix, aids),
        "episode": episode_features(prefix, aids),
        **graphs.transform(prefix, aids),
    }
    names = tuple(
        name for family in DOMAIN_FAMILIES for name in domain_feature_names(family)
    )
    matrix = np.column_stack([blocks[family] for family in DOMAIN_FAMILIES]).astype(
        np.float32
    )
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise ValueError("domain feature matrix is incomplete or non-finite")
    return matrix, names


def verify_cached_query(
    stored: pl.DataFrame,
    *,
    session: int,
    aids: np.ndarray,
    target: np.ndarray,
    chosen: np.ndarray,
) -> None:
    expected = pl.DataFrame(
        {
            "session": np.full(chosen.size, session, dtype=np.int32),
            "aid": aids[chosen].astype(np.int32),
            "candidate_position": chosen.astype(np.int16),
            **{
                f"target_{objective}": target[chosen, j]
                for j, objective in enumerate(OBJECTIVES)
            },
        }
    ).sort("candidate_position")
    left = stored.select(expected.columns).sort("candidate_position")
    if not left.equals(expected):
        raise ValueError(f"candidate identity/target parity failed for session {session}")


def diagnostics(
    names: tuple[str, ...], matrix: np.ndarray, baseline_count: int
) -> pl.DataFrame:
    standard = matrix.std(axis=0)
    nonzero = np.mean(matrix != 0, axis=0)
    max_corr = np.zeros(len(names), dtype=np.float64)
    base = matrix[:, :baseline_count].astype(np.float64)
    base_std = base.std(axis=0)
    valid_base = base_std > 1e-12
    if valid_base.any():
        base_centered = base[:, valid_base] - base[:, valid_base].mean(axis=0)
        base_centered /= np.maximum(base[:, valid_base].std(axis=0), 1e-12)
        for j in range(baseline_count, len(names)):
            if standard[j] <= 1e-12:
                continue
            column = matrix[:, j].astype(np.float64)
            column = (column - column.mean()) / column.std()
            max_corr[j] = np.max(
                np.abs(column @ base_centered / max(1, column.size - 1))
            )
    return pl.DataFrame(
        {
            "feature": list(names),
            "std": standard.astype(np.float64),
            "nonzero_share": nonzero.astype(np.float64),
            "max_abs_corr_with_frozen_baseline": max_corr,
            "family": [
                "baseline" if i < baseline_count else "shared_added"
                for i in range(len(names))
            ],
        }
    )


def run_role(
    role: str,
    *,
    corpus: Path,
    retrieval: Path,
    cache: Path,
    preflight: Path,
    graphs: NormalizedGraphSignals,
    baseline_names: tuple[str, ...],
    added_names: tuple[str, ...],
    negative_budget: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    queries = Queries(corpus, role)
    ids = selected_sessions(preflight, role)
    indices = role_indices(queries, ids)
    stored = cached_metadata(cache, ids)
    engine = FeatureEngine(retrieval)
    if engine.cutoff != FROZEN_HISTORY_END:
        raise ValueError("baseline retrieval cutoff differs from frozen Aug-16 history")
    all_domain_names: tuple[str, ...] | None = None
    added_indices: list[int] | None = None
    names = (*baseline_names, *added_names)
    rows: list[pl.DataFrame] = []
    hashes: dict[int, str] = {}
    matrices: list[np.ndarray] = []
    candidate_rows = 0
    denominator = np.zeros(3, dtype=np.int64)
    for ordinal, index in enumerate(indices):
        prefix = queries.prefix(int(index))
        candidates = engine.candidates(prefix, CANDIDATE_BUDGET)
        target = queries.targets(prefix.session, candidates)
        base = engine.transform(prefix, candidates, baseline_names).astype(np.float32)
        domain, domain_names = domain_matrix(graphs, prefix, candidates.aid)
        if all_domain_names is None:
            all_domain_names = domain_names
            missing = [name for name in added_names if name not in domain_names]
            if missing:
                raise ValueError(
                    f"frozen added features missing from domain catalog: {missing[:3]}"
                )
            added_indices = [domain_names.index(name) for name in added_names]
        elif domain_names != all_domain_names:
            raise ValueError("domain feature schema changed within one smoke run")
        assert added_indices is not None
        matrix = np.column_stack((base, domain[:, added_indices])).astype(np.float32)
        if matrix.shape != (candidates.aid.size, len(names)) or not np.isfinite(matrix).all():
            raise ValueError("frozen 102/134 matrix shape or finiteness failed")
        chosen = (
            np.arange(candidates.aid.size, dtype=np.int64)
            if role == "selection"
            else sampled_rows(
                target, candidates.aid, prefix.session, negative_budget, seed
            )
        )
        verify_cached_query(
            stored[int(prefix.session)],
            session=int(prefix.session),
            aids=candidates.aid,
            target=target,
            chosen=np.asarray(chosen, dtype=np.int64),
        )
        denominator += queries.denominators[int(index)]
        hashes[int(prefix.session)] = matrix_digest(candidates.aid, target, matrix)
        matrices.append(matrix)
        fusion = baseline_fusion(candidates.source_ranks)
        rows.append(
            pl.DataFrame(matrix, schema=list(names)).with_columns(
                pl.Series(
                    "session",
                    np.full(candidates.aid.size, prefix.session, dtype=np.int32),
                ),
                pl.Series("aid", candidates.aid.astype(np.int32)),
                pl.Series(
                    "candidate_position",
                    np.arange(candidates.aid.size, dtype=np.int16),
                ),
                *[
                    pl.Series(f"target_{objective}", target[:, j])
                    for j, objective in enumerate(OBJECTIVES)
                ],
                *[
                    pl.Series(f"baseline_{objective}", fusion[:, j])
                    for j, objective in enumerate(OBJECTIVES)
                ],
            )
        )
        candidate_rows += candidates.aid.size
        if (ordinal + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "stage": f"{role}_features",
                        "completed_sessions": ordinal + 1,
                        "total_sessions": SESSION_COUNT,
                    }
                ),
                flush=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    frame = pl.concat(rows)
    path = output / f"early_{role}_shared_smoke.parquet"
    frame.write_parquet(path, compression="zstd")
    if role == "fit":
        diagnostics(names, np.vstack(matrices), len(baseline_names)).write_parquet(
            output / "early_fit_feature_diagnostics.parquet", compression="zstd"
        )
    return {
        "sessions": SESSION_COUNT,
        "candidate_rows": candidate_rows,
        "denominators": denominator.tolist(),
        "session_hashes": {str(k): v for k, v in sorted(hashes.items())},
        "parquet": path.name,
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def replay_role(
    role: str,
    *,
    corpus: Path,
    retrieval: Path,
    preflight: Path,
    graphs: NormalizedGraphSignals,
    baseline_names: tuple[str, ...],
    added_names: tuple[str, ...],
    expected: dict[str, str],
) -> None:
    queries = Queries(corpus, role)
    ids = selected_sessions(preflight, role)
    indices = role_indices(queries, ids)
    engine = FeatureEngine(retrieval)
    added_indices: list[int] | None = None
    for ordinal, index in enumerate(indices):
        prefix = queries.prefix(int(index))
        candidates = engine.candidates(prefix, CANDIDATE_BUDGET)
        target = queries.targets(prefix.session, candidates)
        base = engine.transform(prefix, candidates, baseline_names).astype(np.float32)
        domain, names = domain_matrix(graphs, prefix, candidates.aid)
        if added_indices is None:
            added_indices = [names.index(name) for name in added_names]
        matrix = np.column_stack((base, domain[:, added_indices])).astype(np.float32)
        actual = matrix_digest(candidates.aid, target, matrix)
        if expected[str(int(prefix.session))] != actual:
            raise ValueError(
                f"deterministic replay differs for {role} session {prefix.session}"
            )
        if (ordinal + 1) % 64 == 0:
            print(
                json.dumps(
                    {
                        "stage": f"{role}_replay",
                        "completed_sessions": ordinal + 1,
                        "total_sessions": SESSION_COUNT,
                    }
                ),
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(
        (args.repo / "configs/shared_feature_validation.json").read_text()
    )
    retrieval_config = json.loads(
        (args.repo / "configs/retrieval_study.json").read_text()
    )["retrieval"]
    baseline_names = tuple(config["baseline_features"])
    added_names = tuple(config["added_features"])
    if (
        len(baseline_names) != 102
        or len(added_names) != 32
        or len(set((*baseline_names, *added_names))) != 134
    ):
        raise ValueError("frozen shared schema must remain exactly 102 + 32 unique features")
    corpus = args.inputs / "corpus"
    retrieval = args.inputs / "retrieval"
    corpus_manifest = json.loads((corpus / "manifest.json").read_text())
    if corpus_manifest["protocol"]["history_end"] != FROZEN_HISTORY_END:
        raise ValueError("early corpus is not the preregistered Aug-16 window")
    args.output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shared_feature_smoke")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    graph_reports = {}
    for family in FAMILIES:
        graph_reports[family] = build_retrievers(
            corpus,
            args.output / "graphs" / family,
            {**retrieval_config, "direction": family},
            logger=logger,
            threads=8,
            memory_gib=16,
        )
        if (
            graph_reports[family]["history_end"] != FROZEN_HISTORY_END
            or graph_reports[family]["query_labels_used"] is not False
            or graph_reports[family]["observed_history_max_ts"]
            >= FROZEN_HISTORY_END
            or graph_reports[family]["completed_parts"]
            != retrieval_config["source_partitions"]
        ):
            raise ValueError(f"{family} graph cutoff/completion contract failed")
    normalized = NormalizedGraphSignals(
        {
            family: GraphSignals(args.output / "graphs" / family, family)
            for family in FAMILIES
        }
    )
    fit_contract = json.loads((args.inputs / "fit_cache/contract.json").read_text())
    selection_contract = json.loads(
        (args.inputs / "selection_cache/contract.json").read_text()
    )
    if (
        fit_contract["candidate_budget"] != CANDIDATE_BUDGET
        or selection_contract["candidate_budget"] != CANDIDATE_BUDGET
    ):
        raise ValueError("existing candidate cache budget differs from frozen smoke budget")
    negative_budget = int(fit_contract["negative_budget"])
    seed = int(fit_contract["seed"])
    if (
        selection_contract["negative_budget"] is not None
        or int(selection_contract["seed"]) != seed
    ):
        raise ValueError("existing fit/selection candidate contracts are inconsistent")
    roles = {}
    for role in ("fit", "selection"):
        roles[role] = run_role(
            role,
            corpus=corpus,
            retrieval=retrieval,
            cache=args.inputs / f"{role}_cache",
            preflight=args.preflight,
            graphs=normalized,
            baseline_names=baseline_names,
            added_names=added_names,
            negative_budget=negative_budget,
            seed=seed,
            output=args.output,
        )
    for role in ("fit", "selection"):
        replay_role(
            role,
            corpus=corpus,
            retrieval=retrieval,
            preflight=args.preflight,
            graphs=normalized,
            baseline_names=baseline_names,
            added_names=added_names,
            expected=roles[role]["session_hashes"],
        )
    report = {
        "status": "EARLY_SHARED_FEATURE_SMOKE_PASSED",
        "history_end": FROZEN_HISTORY_END,
        "schema": {"baseline": 102, "added": 32, "total": 134},
        "candidate_budget": CANDIDATE_BUDGET,
        "graphs": {
            family: {
                "input_id": graph_reports[family]["input_id"],
                "completed_parts": graph_reports[family]["completed_parts"],
                "reused_parts": graph_reports[family]["reused_parts"],
                "elapsed_seconds": graph_reports[family]["elapsed_seconds"],
                "observed_history_max_ts": graph_reports[family][
                    "observed_history_max_ts"
                ],
            }
            for family in FAMILIES
        },
        "roles": roles,
        "deterministic_full_replay": True,
        "model_fits": 0,
        "feature_screening": (
            "none; diagnostics are unsupervised fitting-role smoke statistics only"
        ),
        "selection_feature_formulas_changed": False,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "Measure 1024 complete fitting queries before any ranker fit; then "
            "fitting-only screening/ablation."
        ),
    }
    atomic_json(args.output / "result.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_rss_mib": report["peak_rss_mib"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
