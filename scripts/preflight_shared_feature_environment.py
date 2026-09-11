"""Inspect an existing OTTO environment and audit frozen early query inputs; no fitting.

Run with a project interpreter. Dependencies are imported only in the real-data
stage. Recovered source files and the existing environment are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

OBJECTIVES = ("clicks", "carts", "orders")
COHORT_SHA256 = "622d26ff70b04b4c38d30a659ff1724bf1f386a0844647d4b3f4b3b7a6d6ddf1"
HISTORY_END = 1660687200000
FIT_END = 1660946400000
SELECTION_END = 1661032800000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def save_new(path: Path, data: bytes) -> None:
    """Reuse identical files, preserving any conflicting prior evidence."""
    if path.is_symlink():
        raise ValueError(f"Destination is a symlink: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing evidence differs; preserved: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def inspect_environment(repo: Path) -> dict[str, Any]:
    tomllib = importlib.import_module("tomllib")
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    requirements = [*project["dependencies"], *project["optional-dependencies"]["ml"]]
    versions: dict[str, Any] = {}
    for requirement in requirements:
        name, separator, expected = requirement.partition("==")
        if separator != "==":
            raise ValueError(f"Unrecognized dependency constraint: {requirement}")
        actual: str | None
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
    imports: dict[str, Any] = {}
    for name in ("numpy", "polars", "pyarrow", "duckdb", "lightgbm"):
        try:
            module = importlib.import_module(name)
            imports[name] = {"status": "passed", "version": getattr(module, "__version__", None)}
        except Exception as error:
            imports[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    python_matches = (sys.version_info.major, sys.version_info.minor) == (3, 13)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "python_requirement": project["requires-python"],
        "python_minor_matches": python_matches,
        "dependency_versions": versions,
        "imports": imports,
        "locked_environment_matches": python_matches
        and all(v["matches"] for v in versions.values()),
        "reader_imports_pass": all(v["status"] == "passed" for v in imports.values()),
        "installs": 0,
    }


def audit_session(
    observed: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    frozen: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    """Independent prefix/target checks, allowing equal timestamps at later indices."""
    if role not in ("fit", "selection"):
        raise ValueError("Only fitting and selection roles are allowed")
    start, end = (HISTORY_END, FIT_END) if role == "fit" else (FIT_END, SELECTION_END)
    session = frozen["session"]
    rows = sorted(observed, key=lambda row: row["event_index"])
    if not rows or len(rows) != frozen["observed_events"]:
        raise ValueError("Observed event count mismatch")
    if [row["event_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Observed prefix is not complete and contiguous")
    combined = rows + labels + queries
    if any(row["session"] != session or row["split_role"] != role for row in combined):
        raise ValueError("Session identity or split role mismatch")
    if any(row["event_type"] not in (0, 1, 2) or row["aid"] < 0 for row in rows):
        raise ValueError("Invalid observed action or item")
    timestamps = [row["ts"] for row in rows]
    if timestamps != sorted(timestamps) or not start <= timestamps[0] <= timestamps[-1] < end:
        raise ValueError("Observed timestamps violate temporal role")
    for key, actual in (("first_ts", timestamps[0]), ("query_ts", timestamps[-1]),
                        ("observed_last_index", len(rows) - 1), ("period_end", end)):
        if frozen[key] != actual:
            raise ValueError(f"Frozen prefix ledger mismatch: {key}")
    if len(queries) != 3 or {row["objective"] for row in queries} != set(OBJECTIVES):
        raise ValueError("Missing or duplicate objective ledger")
    unique: set[tuple[str, int]] = set()
    truth: dict[str, list[int]] = {objective: [] for objective in OBJECTIVES}
    for row in labels:
        if row["objective"] not in truth or row["aid"] < 0:
            raise ValueError("Invalid target action or item")
        target_key = (row["objective"], row["aid"])
        if target_key in unique:
            raise ValueError("Duplicate distinct target")
        unique.add(target_key)
        if not frozen["query_ts"] <= row["label_ts"] < end:
            raise ValueError("Target violates timestamp cutoff")
        if row["label_event_index"] <= frozen["observed_last_index"]:
            raise ValueError("Target is not after the observed prefix")
        truth[row["objective"]].append(row["aid"])
    if len(truth["clicks"]) > 1:
        raise ValueError("Next-click target must contain at most one item")
    denominators = []
    for objective in OBJECTIVES:
        query = next(row for row in queries if row["objective"] == objective)
        if any(query[key] != value for key, value in frozen.items()):
            raise ValueError("Query metadata differs from frozen ledger")
        count = len(truth[objective])
        if query["true_items"] != count or query["recall_denominator"] != min(20, count):
            raise ValueError("Pooled target denominator mismatch")
        denominators.append(min(20, count))
    return {"session": session, "observed_events": len(rows),
            "truth": {key: sorted(value) for key, value in truth.items()},
            "denominators": denominators}


def group_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["session"], []).append(row)
    return grouped


def audit_early(repo: Path, evidence: Path, output: Path) -> dict[str, Any]:
    """Use the canonical production reader and independently compare 512 full queries."""
    started = time.monotonic()
    sys.path.insert(0, str(repo / "src"))
    dataset = importlib.import_module("otto_recsys.research.dataset")
    np = importlib.import_module("numpy")
    pq = importlib.import_module("pyarrow.parquet")
    manifest_path = evidence / "preflight/early/corpus/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    if (protocol["history_end"], protocol["fit_end"], protocol["selection_end"]) != (
        HISTORY_END, FIT_END, SELECTION_END
    ) or manifest["status"] != "passed":
        raise ValueError("Early corpus temporal contract mismatch")
    sources = {
        "observed.parquet": evidence / "smoke_inputs/early/corpus/observed.parquet",
        "labels.parquet": evidence / "smoke_inputs/early/corpus/labels.parquet",
        "queries.parquet": evidence / "preflight/early/corpus/queries.parquet",
    }
    corpus = output / "query_inputs"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        if digest(path) != manifest["files"][name]:
            raise ValueError(f"Recovered input checksum mismatch: {name}")
        target = corpus / name
        if target.is_symlink():
            raise ValueError("Composed query input is a symlink")
        if target.exists():
            if digest(target) != manifest["files"][name]:
                raise ValueError("Composed query input differs; preserved")
        else:
            os.link(path, target)
    save_new(corpus / "manifest.json", manifest_path.read_bytes())
    cohort_path = evidence / "smoke_inputs/shared_feature_confirmation_cohorts.json"
    if digest(cohort_path) != COHORT_SHA256:
        raise ValueError("Frozen cohort artifact checksum mismatch")
    cohorts = json.loads(cohort_path.read_text())
    results: dict[str, Any] = {}
    seen: set[int] = set()
    artifacts: dict[str, Any] = {}
    schemas: dict[str, Any] = {}
    for role in ("fit", "selection"):
        role_started = time.monotonic()
        reader = dataset.Queries(corpus, role)
        ordered = reader.session[reader.indices(cohorts["cohort_seed"])]
        frozen_ids = cohorts["windows"]["early"][role]["ranked_session_ids"]
        if ordered[:len(frozen_ids)].tolist() != frozen_ids:
            raise ValueError("Canonical cohort hash/order differs from frozen selection")
        ids = sorted(frozen_ids[:256])
        if len(set(ids)) != 256 or seen.intersection(ids):
            raise ValueError("Duplicate or overlapping smoke cohort")
        seen.update(ids)
        ledger_path = evidence / f"smoke_inputs/cohorts/early_{role}_smoke_ledger.json"
        frozen_rows = json.loads(ledger_path.read_text())
        if sorted(row["session"] for row in frozen_rows) != ids:
            raise ValueError("Frozen smoke ledger membership differs")
        grouped: dict[str, Any] = {}
        tables: dict[str, Any] = {}
        for name, path in sources.items():
            table = pq.read_table(path, filters=[("split_role", "=", role), ("session", "in", ids)])
            schemas[name] = str(table.schema)
            tables[name] = table
            grouped[name] = group_rows(table.to_pylist())
        totals = np.zeros(3, dtype=np.int64)
        event_count = 0
        for frozen in frozen_rows:
            session = frozen["session"]
            observed_rows = grouped["observed.parquet"].get(session, [])
            summary = audit_session(observed_rows, grouped["labels.parquet"].get(session, []),
                                    grouped["queries.parquet"].get(session, []), frozen, role)
            index = int(np.searchsorted(reader.session, session))
            prefix = reader.prefix(index)
            prefix.validate()
            ordered_rows = sorted(observed_rows, key=lambda row: row["event_index"])
            vectors = ((prefix.aid, "aid"), (prefix.ts, "ts"), (prefix.kind, "event_type"))
            for actual, key in vectors:
                if actual.tolist() != [row[key] for row in ordered_rows]:
                    raise ValueError("Production reader and independent prefix differ")
            if reader.denominators[index].tolist() != summary["denominators"]:
                raise ValueError("Production reader denominator differs")
            for objective in OBJECTIVES:
                actual_truth = sorted(reader.labels.get((session, objective), []))
                if actual_truth != summary["truth"][objective]:
                    raise ValueError("Production reader target set differs")
            totals += np.array(summary["denominators"])
            event_count += summary["observed_events"]
        for name, table in tables.items():
            destination = output / f"early_{role}_{name}"
            # First publication and replay must produce byte-identical complete partitions.
            arrow = importlib.import_module("pyarrow")
            sink = arrow.BufferOutputStream()
            pq.write_table(table, sink, compression="zstd")
            data = sink.getvalue().to_pybytes()
            save_new(destination, data)
            artifacts[destination.name] = {"sha256": digest(destination), "bytes": len(data),
                                           "rows": table.num_rows}
        results[role] = {"sessions": 256, "pool_sessions": int(reader.session.size),
                         "observed_events": event_count, "pooled_denominators": totals.tolist(),
                         "elapsed_seconds": round(time.monotonic() - role_started, 3)}
        print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "stage": f"early_{role}_audit_passed", **results[role]}), flush=True)
    return {
        "status": "EARLY_PREFIX_TARGET_AUDIT_PASSED",
        "roles": results, "artifacts": artifacts, "input_schemas": schemas,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "model_fits": 0, "new_candidate_sets": 0, "feature_matrices_built": 0,
        "scope": "Canonical reader parity and frozen 256 complete queries per role; early only",
        "limitations": [
            "Not full raw-event replay; target provenance checked from saved timestamped labels.",
            "Historical wide graphs and 102/134 feature reconstruction remain unresolved.",
            "No official model metric produced and no feature selection or promotion.",
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = inspect_environment(args.repo)
    (args.output / "environment.json").write_bytes(encode(environment))
    if not args.audit:
        print(json.dumps(environment), flush=True)
        return 0
    if not environment["locked_environment_matches"] or not environment["reader_imports_pass"]:
        print("ENVIRONMENT_REVIEW_REQUIRED: no installation, fitting or source modification")
        return 2
    if args.evidence is None:
        raise ValueError("--evidence is required for --audit")
    first = audit_early(args.repo, args.evidence, args.output)
    second = audit_early(args.repo, args.evidence, args.output)
    if first["artifacts"] != second["artifacts"]:
        raise ValueError("Audit replay changed frozen partitions")
    first["byte_identical_replay"] = True
    first["replay_seconds"] = second["elapsed_seconds"]
    (args.output / "early_prefix_target_audit.json").write_bytes(encode(first))
    print(first["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
