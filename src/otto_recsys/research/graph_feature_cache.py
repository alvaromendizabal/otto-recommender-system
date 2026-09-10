"""Preserve baseline candidate rows while adding complementary graph features."""

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
from otto_recsys.research.graph_signals import FAMILIES, GraphSignals, feature_names
from otto_recsys.research.materialize import MODEL_METADATA
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import verified_file
from otto_recsys.runtime import Heartbeat


def augment_cache(
    source: Path,
    output: Path,
    queries: Queries,
    graphs: dict[str, GraphSignals],
    *,
    graph_hashes: dict[str, str],
    base_names: tuple[str, ...],
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    original = json.loads((source / "manifest.json").read_text())
    if original["status"] != "passed" or original["role"] != queries.role:
        raise ValueError("graph signal input cache has an incorrect temporal role")
    if queries.role not in ("fit", "selection"):
        raise ValueError("graph signal development cannot accept evaluation queries")
    names = (*base_names, *(name for family in FAMILIES for name in feature_names(family)))
    contract = {
        "source_cache_id": original["input_id"],
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "corpus_id": queries.manifest["input_id"],
        "role": queries.role,
        "graphs": graph_hashes,
        "features": list(names),
        "code_sha256": sha256_file(Path(__file__)),
        "candidate_policy": "exact existing rows and order; never inject labels or candidates",
    }
    identity = canonical_json_sha256(contract)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("augmented cache contract differs")
    atomic_json(contract_path, contract)
    if publish:
        publish(contract_path)
    indices = {int(session): i for i, session in enumerate(queries.session)}
    count = 0
    files = {}
    with Heartbeat(logger, stage=f"graph_features_{queries.role}", interval_seconds=15):
        for name in original["parts"]:
            source_path, destination = source / name, output / name
            if sha256_file(source_path) != original["files"][name]:
                raise ValueError("source candidate feature cache checksum mismatch")
            if not verified_file(destination, identity):
                frame = pl.read_parquet(source_path, columns=[*base_names, *MODEL_METADATA])
                sessions = frame["session"].to_numpy()
                aids = frame["aid"].to_numpy()
                ids, starts, groups = np.unique(sessions, return_index=True, return_counts=True)
                if (np.diff(sessions) < 0).any() or not set(ids).issubset(indices):
                    raise ValueError("cache groups do not belong to the declared prefix ledger")
                matrix = np.empty((frame.height, len(names) - len(base_names)), dtype=np.float32)
                for session, start, size in zip(ids, starts, groups, strict=True):
                    prefix = queries.prefix(indices[int(session)])
                    matrix[start : start + size] = np.column_stack(
                        [
                            graphs[family].transform(prefix, aids[start : start + size])
                            for family in FAMILIES
                        ]
                    )
                frame = frame.hstack(pl.DataFrame(matrix, schema=list(names[len(base_names) :])))
                temporary = destination.with_suffix(".parquet.tmp")
                frame.write_parquet(temporary, compression="zstd")
                temporary.replace(destination)
                atomic_json(
                    destination.with_suffix(".json"),
                    {
                        "input_id": identity,
                        "sha256": sha256_file(destination),
                        "rows": frame.height,
                    },
                )
            receipt = json.loads(destination.with_suffix(".json").read_text())
            count += int(receipt["rows"])
            files[name] = receipt["sha256"]
            if publish:
                publish(destination)
                publish(destination.with_suffix(".json"))
            logger.info(
                "graph_feature_part_complete",
                extra={
                    "role": queries.role,
                    "part": name,
                    "completed_rows": count,
                },
            )
    if count != original["rows"]:
        raise ValueError("augmentation must preserve every candidate row")
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
    if publish:
        publish(output / "manifest.json")
    return result
