"""Measure the frozen 134-column representation on 1,024 early fitting sessions.

This gate reuses the already-certified Aug-16 symmetric/forward graphs and
baseline candidate policy. It never accesses selection/evaluation roles and
never fits a ranker. Exact candidate/target parity, streaming unsupervised
diagnostics, partition checksums and a second deterministic replay are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256
from otto_recsys.research.dataset import OBJECTIVES, Queries, sampled_rows
from otto_recsys.research.domain_features import (
    DOMAIN_FAMILIES,
    NormalizedGraphSignals,
    episode_features,
    funnel_features,
)
from otto_recsys.research.domain_features import feature_names as domain_feature_names
from otto_recsys.research.feature_diagnostics import StreamingFeatureDiagnostics
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals
from otto_recsys.research.protocol import atomic_json

FROZEN_HISTORY_END = 1660687200000
COHORT_SEED = 20260911
SESSION_COUNT = 1024
SMOKE_SESSION_COUNT = 256
CANDIDATE_BUDGET = 400
PART_SESSIONS = 256


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def matrix_digest(aids: np.ndarray, target: np.ndarray, matrix: np.ndarray) -> str:
    return hashlib.sha256(
        b"".join(
            (
                np.asarray(aids, dtype="<i8").tobytes(order="C"),
                np.asarray(target, dtype="<i1").tobytes(order="C"),
                np.asarray(matrix, dtype="<f4").tobytes(order="C"),
            )
        )
    ).hexdigest()


def fitting_ids(queries: Queries, preflight: Path) -> np.ndarray:
    order = queries.indices(COHORT_SEED)
    ids = queries.session[order[:SESSION_COUNT]].astype(np.int64)
    if ids.size != SESSION_COUNT or np.unique(ids).size != SESSION_COUNT:
        raise ValueError("frozen fitting scale cohort must contain 1,024 unique sessions")
    smoke = (
        pl.read_parquet(preflight / "early_fit_queries.parquet", columns=["session"])
        .unique()
        .sort("session")["session"]
        .to_numpy()
        .astype(np.int64)
    )
    if smoke.size != SMOKE_SESSION_COUNT or not np.array_equal(
        np.sort(ids[:SMOKE_SESSION_COUNT]), smoke
    ):
        raise ValueError("1,024 fitting cohort is not nested over the verified smoke cohort")
    return np.sort(ids)


def role_indices(queries: Queries, ids: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(queries.session, ids)
    if (
        (indices >= queries.session.size).any()
        or not np.array_equal(queries.session[indices], ids)
    ):
        raise ValueError("scale cohort differs from complete fitting query ledger")
    return indices.astype(np.int64)


def cached_metadata(
    cache: Path,
    ids: np.ndarray,
    *,
    corpus_id: str,
) -> tuple[dict[int, pl.DataFrame], dict[str, Any]]:
    manifest = json.loads((cache / "manifest.json").read_text())
    contract = json.loads((cache / "contract.json").read_text())
    if (
        manifest["status"] != "passed"
        or manifest["role"] != "fit"
        or contract["role"] != "fit"
        or manifest["input_id"] != canonical_json_sha256(contract)
        or contract["corpus_id"] != corpus_id
    ):
        raise ValueError("fitting cache manifest/contract lineage differs")
    paths = sorted(cache.glob("part-*.parquet"))
    if not paths:
        raise ValueError("fitting candidate cache has no staged parts")
    for path in paths:
        if path.name not in manifest["files"] or sha256_file(path) != manifest["files"][path.name]:
            raise ValueError(f"staged fitting cache checksum differs: {path.name}")
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
        raise ValueError("staged fitting cache is missing a scale-cohort session")
    return groups, contract


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
    if not stored.select(expected.columns).sort("candidate_position").equals(expected):
        raise ValueError(f"candidate identity/target parity failed for session {session}")


def graph_sources(graph_root: Path, evidence: dict[str, Any]) -> dict[str, GraphSignals]:
    expected = evidence["graphs"]
    sources: dict[str, GraphSignals] = {}
    for family in FAMILIES:
        manifest = json.loads((graph_root / family / "manifest.json").read_text())
        if (
            manifest["input_id"] != expected[family]["input_id"]
            or int(manifest["history_end"]) != FROZEN_HISTORY_END
            or int(manifest["observed_history_max_ts"]) >= FROZEN_HISTORY_END
            or manifest["query_labels_used"] is not False
            or int(manifest["completed_parts"]) != 64
        ):
            raise ValueError(f"{family} graph differs from the certified Aug-16 evidence")
        sources[family] = GraphSignals(graph_root / family, family)
    return sources


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
        raise ValueError("domain matrix is incomplete or non-finite")
    return matrix, names


def family_labels(
    baseline_names: tuple[str, ...],
    added_names: tuple[str, ...],
    blocks: dict[str, list[str]],
) -> tuple[str, ...]:
    owner: dict[str, str] = {}
    for family, values in blocks.items():
        for name in values:
            if name in owner:
                raise ValueError("added feature appears in multiple ablation blocks")
            owner[name] = family
    if set(owner) != set(added_names):
        raise ValueError("ablation blocks do not exactly cover the frozen additions")
    return tuple(["baseline"] * len(baseline_names) + [owner[n] for n in added_names])


def feature_matrix(
    engine: FeatureEngine,
    graphs: NormalizedGraphSignals,
    prefix: Any,
    baseline_names: tuple[str, ...],
    added_names: tuple[str, ...],
    added_indices: list[int] | None,
) -> tuple[Any, np.ndarray, tuple[str, ...], list[int]]:
    candidates = engine.candidates(prefix, CANDIDATE_BUDGET)
    base = engine.transform(prefix, candidates, baseline_names).astype(np.float32)
    domain, domain_names = domain_matrix(graphs, prefix, candidates.aid)
    if added_indices is None:
        missing = [name for name in added_names if name not in domain_names]
        if missing:
            raise ValueError(f"frozen additions missing from domain catalog: {missing[:3]}")
        added_indices = [domain_names.index(name) for name in added_names]
    matrix = np.column_stack((base, domain[:, added_indices])).astype(np.float32)
    if matrix.shape != (candidates.aid.size, len(baseline_names) + len(added_names)):
        raise ValueError("scale feature matrix has the wrong shape")
    if not np.isfinite(matrix).all():
        raise ValueError("scale feature matrix contains non-finite values")
    return candidates, matrix, domain_names, added_indices


def run(
    *,
    repo: Path,
    inputs: Path,
    preflight: Path,
    graph_root: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads((repo / "configs/shared_feature_validation.json").read_text())
    evidence = json.loads((repo / "reports/research/early_shared_feature_smoke.json").read_text())
    baseline_names = tuple(config["baseline_features"])
    added_names = tuple(config["added_features"])
    names = (*baseline_names, *added_names)
    if len(baseline_names) != 102 or len(added_names) != 32 or len(set(names)) != 134:
        raise ValueError("frozen scale schema must remain exactly 102 + 32 unique features")
    if evidence["status"] != "EARLY_SHARED_FEATURE_SMOKE_PASSED":
        raise ValueError("the prerequisite Aug-16 smoke evidence is not passed")

    corpus = inputs / "corpus"
    manifest = json.loads((corpus / "manifest.json").read_text())
    if int(manifest["protocol"]["history_end"]) != FROZEN_HISTORY_END:
        raise ValueError("scale corpus is not the certified Aug-16 corpus")
    queries = Queries(corpus, "fit")
    ids = fitting_ids(queries, preflight)
    indices = role_indices(queries, ids)
    stored, contract = cached_metadata(
        inputs / "fit_cache",
        ids,
        corpus_id=manifest["input_id"],
    )
    if (
        int(contract["candidate_budget"]) != CANDIDATE_BUDGET
        or int(contract["negative_budget"]) <= 0
    ):
        raise ValueError("fitting candidate-cache contract differs from frozen policy")
    negative_budget, candidate_seed = int(contract["negative_budget"]), int(contract["seed"])

    sources = graph_sources(graph_root, evidence)
    normalized = NormalizedGraphSignals(sources)
    engine = FeatureEngine(inputs / "retrieval")
    if engine.cutoff != FROZEN_HISTORY_END:
        raise ValueError("baseline retrieval cutoff differs from Aug-16")
    labels = family_labels(baseline_names, added_names, config["ablation_blocks"])
    diagnostics = StreamingFeatureDiagnostics(
        names, baseline_count=len(baseline_names), family=labels
    )

    output.mkdir(parents=True, exist_ok=True)
    part_rows: list[pl.DataFrame] = []
    part_sessions = 0
    part_index = 0
    part_receipts: list[dict[str, Any]] = []
    expected_hashes: dict[str, str] = {}
    denominator = np.zeros(3, dtype=np.int64)
    added_indices: list[int] | None = None
    feature_started = time.perf_counter()

    for ordinal, index in enumerate(indices):
        prefix = queries.prefix(int(index))
        candidates, matrix, _, added_indices = feature_matrix(
            engine, normalized, prefix, baseline_names, added_names, added_indices
        )
        target = queries.targets(prefix.session, candidates)
        chosen = sampled_rows(
            target, candidates.aid, prefix.session, negative_budget, candidate_seed
        )
        verify_cached_query(
            stored[int(prefix.session)],
            session=int(prefix.session),
            aids=candidates.aid,
            target=target,
            chosen=chosen,
        )
        diagnostics.update(matrix)
        denominator += queries.denominators[int(index)]
        expected_hashes[str(int(prefix.session))] = matrix_digest(
            candidates.aid, target, matrix
        )
        part_rows.append(
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
            )
        )
        part_sessions += 1

        if part_sessions == PART_SESSIONS:
            part = output / f"fit_complete_part-{part_index:04d}.parquet"
            frame = pl.concat(part_rows)
            frame.write_parquet(part, compression="zstd")
            part_receipts.append(
                {
                    "part": part.name,
                    "sessions": part_sessions,
                    "rows": frame.height,
                    "sha256": sha256_file(part),
                }
            )
            part_rows, part_sessions = [], 0
            part_index += 1
        if (ordinal + 1) % 64 == 0:
            print(
                json.dumps(
                    {
                        "stage": "fit_scale_features",
                        "completed_sessions": ordinal + 1,
                        "total_sessions": SESSION_COUNT,
                    }
                ),
                flush=True,
            )

    if part_rows or part_sessions:
        raise ValueError("1,024-session scale gate must finish on complete 256-session parts")
    feature_seconds = time.perf_counter() - feature_started

    diag = diagnostics.finalize()
    atomic_json(output / "fit_feature_diagnostics.json", diag)
    ledger = pl.DataFrame(
        {
            "session": ids.astype(np.int64),
            **{
                f"denominator_{objective}": queries.denominators[indices, j]
                for j, objective in enumerate(OBJECTIVES)
            },
        }
    ).sort("session")
    ledger_path = output / "fit_query_ledger.parquet"
    ledger.write_parquet(ledger_path, compression="zstd")

    replay_started = time.perf_counter()
    added_indices = None
    for ordinal, index in enumerate(indices):
        prefix = queries.prefix(int(index))
        candidates, matrix, _, added_indices = feature_matrix(
            engine, normalized, prefix, baseline_names, added_names, added_indices
        )
        target = queries.targets(prefix.session, candidates)
        actual = matrix_digest(candidates.aid, target, matrix)
        if expected_hashes[str(int(prefix.session))] != actual:
            raise ValueError(f"deterministic scale replay differs for {prefix.session}")
        if (ordinal + 1) % 128 == 0:
            print(
                json.dumps(
                    {
                        "stage": "fit_scale_replay",
                        "completed_sessions": ordinal + 1,
                        "total_sessions": SESSION_COUNT,
                    }
                ),
                flush=True,
            )
    replay_seconds = time.perf_counter() - replay_started

    added_rows = [
        row for row in diag["features"] if row["family"] != "baseline"
    ]
    result = {
        "status": "EARLY_SHARED_FEATURE_SCALE_PASSED",
        "history_end": FROZEN_HISTORY_END,
        "cohort_seed": COHORT_SEED,
        "role": "fit",
        "selection_access": False,
        "sessions": SESSION_COUNT,
        "candidate_budget": CANDIDATE_BUDGET,
        "candidate_rows": SESSION_COUNT * CANDIDATE_BUDGET,
        "schema": {"baseline": 102, "added": 32, "total": 134},
        "graphs": {
            family: {
                "input_id": sources[family].manifest["input_id"],
                "history_end": int(sources[family].manifest["history_end"]),
                "observed_history_max_ts": int(
                    sources[family].manifest["observed_history_max_ts"]
                ),
            }
            for family in FAMILIES
        },
        "pooled_denominators": denominator.tolist(),
        "parts": part_receipts,
        "query_ledger": {
            "rows": ledger.height,
            "sha256": sha256_file(ledger_path),
        },
        "diagnostics_sha256": sha256_file(output / "fit_feature_diagnostics.json"),
        "added_feature_diagnostics": {
            "constant": sum(bool(row["constant"]) for row in added_rows),
            "near_duplicate_baseline": sum(
                bool(row["near_duplicate_baseline"]) for row in added_rows
            ),
            "near_duplicate_added": sum(
                bool(row["near_duplicate_added"]) for row in added_rows
            ),
            "minimum_session_nonzero_share": min(
                float(row["session_nonzero_share"]) for row in added_rows
            ),
        },
        "feature_seconds": feature_seconds,
        "replay_seconds": replay_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "deterministic_full_replay": True,
        "model_fits": 0,
        "feature_retention_decisions": 0,
        "interpretation": (
            "Fitting-only engineering/throughput gate. Diagnostic flags do not select "
            "features and no ranking metric or promotion is produced."
        ),
        "next_gate": (
            "If support/resource diagnostics are acceptable, preregister a bounded "
            "fitting-only supervised screen/add-drop comparison; keep selection frozen."
        ),
    }
    atomic_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_rss_mib": result["peak_rss_mib"],
            }
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        repo=args.repo,
        inputs=args.inputs,
        preflight=args.preflight,
        graph_root=args.graphs,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
