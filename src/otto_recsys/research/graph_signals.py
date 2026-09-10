"""Historical wide-graph affinities on an unchanged baseline candidate pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.features import Prefix

FAMILIES = ("symmetric", "forward")
CHANNELS = ("time", "cart", "order")
ACTIONS = (("all", None), ("clicks", 0), ("carts", 1), ("orders", 2))
LENGTHS = (1, 5, 20)


def feature_names(family: str) -> tuple[str, ...]:
    if family not in FAMILIES:
        raise ValueError("unknown historical graph family")
    return tuple(
        f"wide_{family}_{channel}_{action}_n{length}_{reduction}"
        for channel in CHANNELS
        for action, _ in ACTIONS
        for length in LENGTHS
        for reduction in ("sum", "max")
    )


class GraphSignals:
    """Read verified historical edges; never discover candidates or accept labels."""

    def __init__(self, directory: Path, family: str) -> None:
        self.names = feature_names(family)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.cutoff = int(self.manifest["history_end"])
        if (
            self.manifest["status"] != "passed"
            or self.manifest["query_labels_used"] is not False
            or self.manifest["observed_history_max_ts"] >= self.cutoff
        ):
            raise ValueError("graph must be certified before the fitting cutoff")
        parts = []
        for name, expected in sorted(self.manifest["files"].items()):
            if not name.startswith("parts/"):
                continue
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("graph part escapes its directory")
            if sha256_file(path) != expected:
                raise ValueError("graph part checksum mismatch")
            parts.append(path)
        if not parts:
            raise ValueError("graph has no verified partitions")
        frame = pl.read_parquet(parts)
        source = frame["source_aid"].to_numpy().astype(np.int64)
        target = frame["target_aid"].to_numpy().astype(np.int64)
        if (source < 0).any() or (target < 0).any():
            raise ValueError("graph item identifiers must be nonnegative")
        high = int(max(source.max(initial=0), target.max(initial=0))) + 1
        self.graphs = []
        for channel in CHANNELS:
            values = frame[f"{channel}_score"].to_numpy().astype(np.float32)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("graph weights must be finite and nonnegative")
            self.graphs.append(csr_matrix((values, (source, target)), shape=(high, high)))
        self.high = high

    def transform(self, prefix: Prefix, aids: np.ndarray) -> np.ndarray:
        prefix.validate()
        if prefix.ts[0] < self.cutoff:
            raise ValueError("observed prefix precedes the historical graph cutoff")
        if aids.ndim != 1 or (aids < 0).any() or len(np.unique(aids)) != len(aids):
            raise ValueError("candidate identifiers must be nonnegative and unique")
        seeds, kinds = prefix.aid[::-1][:20], prefix.kind[::-1][:20]
        source_ok, target_ok = seeds < self.high, aids < self.high
        result: list[np.ndarray] = []
        for graph in self.graphs:
            affinity = np.zeros((seeds.size, aids.size), dtype=np.float32)
            if source_ok.any() and target_ok.any():
                affinity[np.ix_(source_ok, target_ok)] = graph[seeds[source_ok]][
                    :, aids[target_ok]
                ].toarray()
            for _, kind in ACTIONS:
                for length in LENGTHS:
                    selected = affinity[:length]
                    if kind is not None:
                        selected = selected[kinds[:length] == kind]
                    result.extend((selected.sum(axis=0), selected.max(axis=0, initial=0)))
        return np.column_stack(result).astype(np.float32, copy=False)
