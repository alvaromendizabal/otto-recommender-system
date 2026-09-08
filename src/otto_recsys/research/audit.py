"""Independently reconstruct labels, score sums and native-model predictions.

The auditor deliberately does not call the training/evaluation metric helpers.
It verifies the published experiment, never changes model selection, and states
the remaining raw JSONL conversion provenance limit explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.evaluation import verify_seal
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.protocol import atomic_json, sql_path
from otto_recsys.runtime import Heartbeat

WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def audit_source(root: Path, source: Path, *, threads: int, memory_gib: int) -> dict[str, Any]:
    """Rebuild all three roles' prefixes and targets from original Parquet events."""
    contract = read(root / "corpus/contract.json")
    source_hashes = contract["source"]["parts"]
    require(
        set(source_hashes) == {p.name for p in source.glob("part-*.parquet")},
        "source partition inventory changed",
    )
    for name, expected in source_hashes.items():
        require(sha256_file(source / name) == expected, "source partition checksum mismatch")
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(threads)}")
    connection.execute(f"SET memory_limit='{int(memory_gib)}GB'")
    try:
        for name, path in (
            ("events", source / "part-*.parquet"),
            ("ledger", root / "corpus/queries.parquet"),
            ("stored_labels", root / "corpus/labels.parquet"),
            ("stored_prefixes", root / "corpus/observed.parquet"),
        ):
            connection.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})"
            )
        connection.execute("""CREATE TABLE boundaries AS SELECT DISTINCT
            session, split_role, first_ts, query_ts, observed_last_index, period_end FROM ledger""")
        role_overlap = connection.execute("""SELECT count(*) FROM (
            SELECT session FROM boundaries GROUP BY session HAVING count(*)<>1)""").fetchone()
        require(role_overlap is not None and role_overlap[0] == 0, "session role overlap")
        connection.execute("""CREATE TABLE selected_events AS
            SELECT e.*, b.split_role, b.observed_last_index, b.period_end, b.query_ts, b.first_ts
            FROM events e INNER JOIN boundaries b USING(session)
            WHERE e.ts < b.period_end""")
        invalid = connection.execute("""SELECT count(*) FROM (
            SELECT session FROM selected_events GROUP BY session
            HAVING min(ts)<>min(first_ts) OR
            max(ts) FILTER (WHERE event_index<=observed_last_index)<>min(query_ts))""").fetchone()
        require(invalid is not None and invalid[0] == 0, "source query boundaries differ")
        connection.execute("""CREATE TABLE rebuilt_prefixes AS
            SELECT session,aid,ts,event_type,event_index,split_role FROM selected_events
            WHERE event_index<=observed_last_index""")
        connection.execute("""CREATE TABLE future AS SELECT * FROM selected_events
            WHERE event_index>observed_last_index""")
        connection.execute("""CREATE TABLE rebuilt_labels AS
            SELECT session,split_role,'clicks' AS objective,aid,ts AS label_ts,
                event_index AS label_event_index FROM future WHERE event_type=0
            QUALIFY row_number() OVER (PARTITION BY session ORDER BY event_index)=1
            UNION ALL
            SELECT session,split_role,CASE event_type WHEN 1 THEN 'carts' ELSE 'orders' END,
                aid,min(ts),min(event_index) FROM future WHERE event_type IN (1,2)
            GROUP BY session,split_role,event_type,aid""")
        differences = {}
        for kind in ("prefixes", "labels"):
            for left, right in (("rebuilt", "stored"), ("stored", "rebuilt")):
                row = connection.execute(
                    f"SELECT count(*) FROM (SELECT * FROM {left}_{kind} "
                    f"EXCEPT ALL SELECT * FROM {right}_{kind})"
                ).fetchone()
                count = int(row[0]) if row else -1
                differences[f"{kind}_{left}_minus_{right}"] = count
                require(count == 0, f"original events and stored {kind} differ")
        roles = connection.execute("""SELECT split_role,count(*) FROM boundaries
            GROUP BY split_role ORDER BY split_role""").fetchall()
        label_count = connection.execute("SELECT count(*) FROM rebuilt_labels").fetchone()
        return {
            "source_partitions": len(source_hashes),
            "roles": dict(roles),
            "reconstructed_labels": int(label_count[0]) if label_count else 0,
            "differences": differences,
            "source_identity_sha256": canonical_json_sha256(source_hashes),
            "scope": "exact original Parquet bytes; original raw JSONL conversion not replayed",
        }
    finally:
        connection.close()


def audit_statistics(root: Path, report: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Check the complete query ledger before independently pooling metric counts."""
    for name, digest in read(root / "corpus/manifest.json")["files"].items():
        require(sha256_file(root / "corpus" / name) == digest, "corpus checksum mismatch")
    parts = report["files"]
    require(bool(parts), "evaluation has no committed parts")
    for name, digest in parts.items():
        path = root / "evaluation" / name
        require(
            path.name == name and sha256_file(path) == digest, "evaluation part checksum mismatch"
        )
    frame = pl.concat([pl.read_parquet(root / "evaluation" / p) for p in sorted(parts)]).sort(
        "session"
    )
    queries = pl.read_parquet(root / "corpus/queries.parquet")
    labels = pl.read_parquet(root / "corpus/labels.parquet")
    require(
        queries.height == queries.unique(["session", "objective"]).height, "duplicate query ledger"
    )
    require(
        labels.height == labels.unique(["session", "objective", "aid"]).height,
        "duplicate truth item",
    )
    joined = labels.join(queries, on=["session", "split_role", "objective"], how="left")
    require(
        joined.filter(
            pl.col("query_ts").is_null()
            | (pl.col("label_ts") < pl.col("query_ts"))
            | (pl.col("label_ts") >= pl.col("period_end"))
            | (pl.col("label_event_index") <= pl.col("observed_last_index"))
        ).height
        == 0,
        "labels violate their future boundary",
    )
    counts = labels.group_by("session", "objective").agg(pl.col("aid").n_unique().alias("count"))
    ledger = queries.join(counts, on=["session", "objective"], how="left").with_columns(
        pl.col("count").fill_null(0)
    )
    require(
        ledger.filter(
            (pl.col("true_items") != pl.col("count"))
            | (pl.col("recall_denominator") != pl.col("count").clip(upper_bound=20))
        ).height
        == 0,
        "complete query denominators differ from unique future labels",
    )
    evaluation = ledger.filter(pl.col("split_role") == "evaluation")
    expected = evaluation["session"].unique().sort().to_numpy()
    require(
        np.array_equal(frame["session"].to_numpy(), expected),
        "evaluation dropped or duplicated queries",
    )
    require(frame.height == report["sessions"], "reported session count differs")
    pooled: dict[str, Any] = {
        name: {"objectives": {}, "weighted_recall_at_20": 0.0}
        for name in ("selected", "core", "fusion")
    }
    frontier: dict[str, Any] = {
        str(k): {"objectives": {}, "weighted_recall_at_20": 0.0} for k in (100, 200, 400)
    }
    for objective, weight in WEIGHTS.items():
        rows = evaluation.filter(pl.col("objective") == objective).sort("session")
        denominators = rows["recall_denominator"].to_numpy()
        require(
            np.array_equal(denominators, frame[f"denominator_{objective}"].to_numpy()),
            "per-query denominator differs",
        )
        denominator = int(denominators.sum())
        for name in pooled:
            hits = frame[f"{name}_{objective}_hits"].to_numpy()
            require(
                bool(((hits >= 0) & (hits <= denominators)).all()), "invalid per-query hit count"
            )
            total = int(hits.sum())
            score = total / denominator if denominator else 0.0
            result = {"hits": total, "denominator": denominator, "recall_at_20": score}
            original = report["scores"][name]["objectives"][objective]
            require(
                all(
                    math.isclose(float(v), float(original[k]), abs_tol=1e-12, rel_tol=0)
                    for k, v in result.items()
                ),
                "published metric differs from pooled counts",
            )
            pooled[name]["objectives"][objective] = result
            pooled[name]["weighted_recall_at_20"] += weight * score
        previous = np.zeros(frame.height, dtype=np.int64)
        for budget in (100, 200, 400):
            hits = frame[f"pool_{budget}_{objective}"].to_numpy()
            require(
                bool(((previous <= hits) & (hits <= denominators)).all()),
                "invalid nested candidate ceiling",
            )
            previous = hits
            result = {
                "hits": int(hits.sum()),
                "denominator": denominator,
                "recall_at_20": int(hits.sum()) / denominator if denominator else 0.0,
            }
            original = report["candidate_frontier"][str(budget)]["objectives"][objective]
            require(result == original, "candidate ceiling differs from pooled counts")
            frontier[str(budget)]["objectives"][objective] = result
            frontier[str(budget)]["weighted_recall_at_20"] += weight * result["recall_at_20"]
        for name in pooled:
            require(
                bool((frame[f"{name}_{objective}_hits"].to_numpy() <= previous).all()),
                "model hits exceed the candidate ceiling",
            )
    for name, values in pooled.items():
        require(
            math.isclose(
                values["weighted_recall_at_20"],
                report["scores"][name]["weighted_recall_at_20"],
                abs_tol=1e-12,
                rel_tol=0,
            ),
            "published weighted metric differs",
        )
    for name, interval in report["paired_intervals"].items():
        gain = pooled["selected"]["weighted_recall_at_20"] - pooled[name]["weighted_recall_at_20"]
        require(
            math.isclose(gain, interval["absolute_gain"], abs_tol=1e-12, rel_tol=0),
            "paired gain differs",
        )
    return frame, {
        "sessions": frame.height,
        "parts": len(parts),
        "scores": pooled,
        "candidate_frontier": frontier,
        "label_boundary_violations": 0,
    }


def audit_ablations(root: Path, seal: dict[str, Any]) -> dict[str, Any]:
    """Recompute all selection scores and check the pre-evaluation model choice."""
    report = read(root / "models/ablation_report.json")
    ledger = pl.read_parquet(root / "corpus/queries.parquet").filter(
        pl.col("split_role") == "selection"
    )
    expected = ledger["session"].unique().sort().to_numpy()
    scores: dict[str, dict[str, float]] = {}
    native_hashes = {}
    statistic_hashes = {}
    for variant, summary in report["variants"].items():
        path = root / "models" / variant / "selection_statistics.parquet"
        statistic_hashes[variant] = sha256_file(path)
        statistics = pl.read_parquet(path).sort("session")
        require(
            np.array_equal(statistics["session"].to_numpy(), expected),
            "ablation selection query ledger differs",
        )
        require(
            read(path.with_name("selection_metrics.json")) == summary,
            "ablation summary differs from its source report",
        )
        scores[variant] = {}
        for objective in OBJECTIVES:
            rows = ledger.filter(pl.col("objective") == objective).sort("session")
            denominators = rows["recall_denominator"].to_numpy()
            hits = statistics[f"hits_{objective}"].to_numpy()
            require(
                bool(((hits >= 0) & (hits <= denominators)).all()), "invalid selection hit count"
            )
            count = int(hits.sum())
            denominator = int(denominators.sum())
            score = count / denominator if denominator else 0.0
            claim = summary["objectives"][objective]
            require(
                count == claim["hits"]
                and denominator == claim["denominator"]
                and math.isclose(score, claim["recall_at_20"], abs_tol=1e-12, rel_tol=0),
                "ablation metric differs from pooled counts",
            )
            model = path.parent / objective / "model.txt"
            digest = sha256_file(model)
            require(
                digest == summary["models"][objective]["model_sha256"],
                "ablation model checksum differs",
            )
            native_hashes[str(model.relative_to(root))] = digest
            scores[variant][objective] = score
        total = sum(WEIGHTS[o] * scores[variant][o] for o in OBJECTIVES)
        require(
            math.isclose(total, summary["weighted_recall_at_20"], abs_tol=1e-12, rel_tol=0),
            "ablation weighted metric differs",
        )
    for objective in OBJECTIVES:
        best_score = max(scores[v][objective] for v in scores)
        winners = [v for v in scores if scores[v][objective] == best_score]
        winner = sorted(winners, key=lambda v: (report["variants"][v]["features"], v))[0]
        require(
            winner == seal["models"][objective]["variant"] == report["chosen"][objective],
            "sealed choice differs from the predeclared selection rule",
        )
    return {
        "variants": len(scores),
        "native_models": len(native_hashes),
        "selection_sessions": len(expected),
        "selection_statistic_sha256": statistic_hashes,
        "native_model_sha256": native_hashes,
        "chosen": report["chosen"],
    }


def replay_models(root: Path, frame: pl.DataFrame, *, sessions: int, seed: int) -> dict[str, Any]:
    """Rebuild features and use Python sets/sorting to audit sampled predictions."""
    require(sessions > 0, "model replay requires positive sessions")
    seal = verify_seal(root)
    queries = Queries(root / "corpus", "evaluation")
    engine = FeatureEngine(root / "retrieval")
    models = {
        m["sha256"]: lgb.Booster(model_file=str(root / m["path"]))
        for group in ("models", "core_models")
        for m in seal[group].values()
    }
    names = tuple(
        dict.fromkeys(
            n
            for group in ("models", "core_models")
            for m in seal[group].values()
            for n in m["features"]
        )
    )
    order = sorted(
        range(queries.session.size),
        key=lambda i: hashlib.sha256(f"audit:{seed}:{int(queries.session[i])}".encode()).digest(),
    )[:sessions]
    records = {
        row["session"]: row
        for row in frame.filter(
            pl.col("session").is_in([int(queries.session[i]) for i in order])
        ).iter_rows(named=True)
    }
    comparisons = 0
    for i in order:
        prefix = queries.prefix(i)
        pool = engine.candidates(prefix, 400)
        x = engine.transform(prefix, pool, names)
        row = records[prefix.session]
        fusion_weights = (
            (1.0, 0.1, 0.1, 1.0, 0.01),
            (0.3, 1.0, 0.3, 1.0, 0.01),
            (0.2, 0.5, 1.0, 1.0, 0.01),
        )
        for j, objective in enumerate(OBJECTIVES):
            truth = set(map(int, queries.labels.get((prefix.session, objective), [])))
            for label, group in (("selected", "models"), ("core", "core_models"), ("fusion", None)):
                if group is None:
                    scores = np.array(
                        [
                            sum(
                                w / (20 + float(r)) if r > 0 else 0.0
                                for w, r in zip(fusion_weights[j], ranks, strict=True)
                            )
                            for ranks in pool.source_ranks
                        ]
                    )
                else:
                    model = seal[group][objective]
                    require(
                        models[model["sha256"]].feature_name() == model["features"],
                        "native model feature order differs",
                    )
                    scores = np.asarray(
                        models[model["sha256"]].predict(
                            x[:, [names.index(n) for n in model["features"]]], num_threads=1
                        )
                    )
                top = sorted(
                    zip(map(int, pool.aid), map(float, scores), strict=True),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:20]
                hits = len(truth.intersection(aid for aid, _ in top))
                require(hits == row[f"{label}_{objective}_hits"], "native model replay differs")
                comparisons += 1
            for budget in (100, 200, 400):
                hits = min(20, len(truth.intersection(map(int, pool.aid[:budget]))))
                require(
                    hits == row[f"pool_{budget}_{objective}"], "replayed candidate ceiling differs"
                )
                comparisons += 1
    return {
        "sessions": len(order),
        "comparisons": comparisons,
        "mismatches": 0,
        "seed": seed,
        "session_ids_sha256": canonical_json_sha256([int(queries.session[i]) for i in order]),
        "scope": "independent metric/set arithmetic; shared frozen feature implementation",
    }


def audit_study(
    root: Path,
    source: Path,
    *,
    logger: logging.Logger,
    sessions: int = 256,
    seed: int = 20260908,
    threads: int = 4,
    memory_gib: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    report = read(root / "evaluation/report.json")
    seal = verify_seal(root)
    require(
        report["status"] == "passed" and report["seal_id"] == seal["seal_id"],
        "incomplete evaluation",
    )
    with Heartbeat(logger, stage="independent_research_audit", interval_seconds=15):
        frame, metrics = audit_statistics(root, report)
        ablations = audit_ablations(root, seal)
        source_result = audit_source(root, source, threads=threads, memory_gib=memory_gib)
        replay = replay_models(root, frame, sessions=sessions, seed=seed)
    result = {
        "status": "passed",
        "seal_id": seal["seal_id"],
        "evaluation_id": report["input_id"],
        "evaluation_report_sha256": sha256_file(root / "evaluation/report.json"),
        "audit_code_sha256": sha256_file(Path(__file__)),
        "source": source_result,
        "statistics": metrics,
        "ablations": ablations,
        "native_replay": replay,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["audit_id"] = canonical_json_sha256(result)
    (root / "audit").mkdir(exist_ok=True)
    atomic_json(root / "audit/report.json", result)
    return result
