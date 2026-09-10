"""Add domain features to immutable candidate rows with per-part recovery."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.research.dataset import Queries
from otto_recsys.research.domain_features import (
    DOMAIN_FAMILIES,
    NormalizedGraphSignals,
    episode_features,
    feature_names,
    funnel_features,
)
from otto_recsys.research.materialize import MODEL_METADATA
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import verified_file
from otto_recsys.runtime import Heartbeat


def augment_cache(
    source: Path,
    output: Path,
    queries: Queries,
    graphs: NormalizedGraphSignals,
    *,
    graph_hashes: dict[str, str],
    base_names: tuple[str, ...],
    logger: logging.Logger,
    publish: Callable[[Path], None],
) -> dict[str, Any]:
    original = json.loads((source / "manifest.json").read_text())
    if (
        original["status"] != "passed"
        or original["role"] != queries.role
        or queries.role not in ("fit", "selection")
        or not set(base_names).issubset(original["features"])
    ):
        raise ValueError("domain cache requires certified fitting or selection candidates")
    if any(g.cutoff != queries.manifest["protocol"]["history_end"] for g in graphs.graphs.values()):
        raise ValueError("domain graphs and query corpus have different cutoffs")
    names = (*base_names, *(name for family in DOMAIN_FAMILIES for name in feature_names(family)))
    if len(set(names)) != len(names):
        raise ValueError("domain columns must be unique and distinct from the baseline")
    contract = {
        "source_cache_id": original["input_id"],
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "corpus_id": queries.manifest["input_id"],
        "role": queries.role,
        "graphs": graph_hashes,
        "features": list(names),
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "domain_feature_cache.py",
                "domain_features.py",
                "graph_signals.py",
                "features.py",
            )
        },
        "candidate_policy": "exact existing candidates, order, labels, fusion scores and negatives",
        "time_unit": "milliseconds; episode thresholds converted from seconds",
    }
    identity = canonical_json_sha256(contract)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("domain cache contract differs")
    atomic_json(contract_path, contract)
    publish(contract_path)
    indices = {int(session): i for i, session in enumerate(queries.session)}
    count, observed_sessions = 0, []
    files = {}
    with Heartbeat(logger, stage=f"domain_features_{queries.role}", interval_seconds=15):
        for part in original["parts"]:
            source_path, destination = source / part, output / part
            if Path(part).name != part or sha256_file(source_path) != original["files"][part]:
                raise ValueError("domain source part checksum or path is invalid")
            frame = pl.read_parquet(source_path, columns=[*base_names, *MODEL_METADATA])
            sessions, aids = frame["session"].to_numpy(), frame["aid"].to_numpy()
            ids, starts, sizes = np.unique(sessions, return_index=True, return_counts=True)
            if (np.diff(sessions) < 0).any() or not set(ids).issubset(indices):
                raise ValueError("domain source groups differ from the prefix ledger")
            observed_sessions.extend(ids.tolist())
            if not verified_file(destination, identity):
                matrix = np.empty((frame.height, len(names) - len(base_names)), dtype=np.float32)
                for session, start, size in zip(ids, starts, sizes, strict=True):
                    prefix, candidates = (
                        queries.prefix(indices[int(session)]),
                        aids[start : start + size],
                    )
                    graph_values = graphs.transform(prefix, candidates)
                    blocks = {
                        "funnel": funnel_features(prefix, candidates),
                        "episode": episode_features(prefix, candidates),
                        **graph_values,
                    }
                    matrix[start : start + size] = np.column_stack(
                        [blocks[family] for family in DOMAIN_FAMILIES]
                    )
                if not np.isfinite(matrix).all():
                    raise ValueError("domain features must be finite before publication")
                augmented = frame.hstack(
                    pl.DataFrame(matrix, schema=list(names[len(base_names) :]))
                )
                temporary = destination.with_suffix(".parquet.tmp")
                augmented.write_parquet(temporary, compression="zstd")
                temporary.replace(destination)
                atomic_json(
                    destination.with_suffix(".json"),
                    {
                        "input_id": identity,
                        "sha256": sha256_file(destination),
                        "rows": frame.height,
                    },
                )
            checked = pl.read_parquet(destination, columns=[*base_names, *MODEL_METADATA])
            if not checked.equals(frame):
                raise ValueError("domain augmentation changed an original feature or candidate row")
            receipt = json.loads(destination.with_suffix(".json").read_text())
            count += int(receipt["rows"])
            files[part] = receipt["sha256"]
            publish(destination)
            publish(destination.with_suffix(".json"))
            logger.info(
                "domain_feature_part_complete",
                extra={
                    "role": queries.role,
                    "part": part,
                    "completed_rows": count,
                    "sessions": len(observed_sessions),
                },
            )
    if count != original["rows"] or not np.array_equal(
        np.asarray(observed_sessions), queries.session
    ):
        raise ValueError("domain cache must preserve every whole query exactly once")
    result = {
        "status": "passed",
        "input_id": identity,
        "role": queries.role,
        "features": list(names),
        "rows": count,
        "sessions": original["sessions"],
        "parts": original["parts"],
        "files": files,
    }
    atomic_json(output / "manifest.json", result)
    publish(output / "manifest.json")
    return result
