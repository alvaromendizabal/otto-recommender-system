"""Audit earlier-window lineage, original events, native models and uncertainty."""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.audit import (
    audit_ablations,
    audit_source,
    audit_statistics,
    read,
    replay_models,
    require,
)
from otto_recsys.research.dataset import OBJECTIVES
from otto_recsys.research.evaluation import verify_seal
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.metrics import paired_bootstrap
from otto_recsys.research.protocol import TemporalProtocol, atomic_json
from otto_recsys.research.robustness import fingerprint, load_plan
from otto_recsys.research.robustness_report import PAYLOADS
from otto_recsys.research.study import variants
from otto_recsys.research.temporal_robustness import (
    INPUT_REPORTS,
    original_source,
    protocol_values,
    verify_window_launch,
)
from otto_recsys.runtime import Heartbeat


def checked_manifest(root: Path, name: str, contract: dict[str, Any]) -> dict[str, Any]:
    directory = root / name
    require(read(directory / "contract.json") == contract, f"{name} contract differs")
    report = read(directory / "manifest.json")
    require(
        report["status"] == "passed" and report["input_id"] == fingerprint(contract),
        f"{name} manifest identity differs",
    )
    require(bool(report["files"]), f"{name} has no verified data")
    for relative, digest in report["files"].items():
        path = directory / relative
        require(path.resolve().is_relative_to(directory.resolve()), "invalid input path")
        require(sha256_file(path) == digest, f"{name} data checksum differs")
    return report


def verify_window_inputs(
    repository: Path, root: Path, launch: dict[str, Any]
) -> dict[str, Any]:
    """Reject future fitted inputs and verify every prepared data part before reuse."""
    cell = verify_window_launch(repository, launch)
    plan, _ = load_plan(repository)
    code = plan["frozen_code"]
    protocol = TemporalProtocol(**protocol_values(launch)).contract()
    corpus = checked_manifest(root, "corpus", {
        "schema_version": 1, "protocol": protocol, "source": original_source(repository),
        "duckdb": duckdb.__version__, "code_sha256": code["protocol.py"],
    })
    require(corpus["protocol"] == protocol, "corpus window differs")
    for role in ("fit", "selection"):
        require(
            corpus["roles"][role]["sessions"] == plan[f"{role}_sessions"],
            "planned temporal cohort budget differs",
        )
    require(set(corpus["files"]) == {
        "history.parquet", "observed.parquet", "queries.parquet", "labels.parquet"
    }, "temporal artifact inventory differs")
    retrieval = checked_manifest(root, "retrieval", {
        "history_sha256": corpus["files"]["history.parquet"],
        "history_end": protocol["history_end"], "config": launch["retrieval"],
        "code_sha256": code["retrievers.py"], "duckdb": duckdb.__version__,
        "query_labels_used": False,
    })
    require(
        retrieval["history_end"] == protocol["history_end"]
        and retrieval["observed_history_max_ts"] < protocol["history_end"]
        and retrieval["query_labels_used"] is False
        and retrieval["completed_parts"] == launch["retrieval"]["source_partitions"],
        "historical retrieval availability or completeness differs",
    )
    expected_parts = {
        f"parts/part-{i:03d}.parquet"
        for i in range(launch["retrieval"]["source_partitions"])
    }
    require(set(retrieval["files"]) == expected_parts | {
        "history_statistics.parquet", "history_tail.parquet"
    }, "historical graph inventory differs")
    names = [f.name for f in feature_catalog()]
    seed = plan["cohort_seed"]
    cache = checked_manifest(root, "screening_cache", {
        "corpus_id": corpus["input_id"], "retrieval_id": retrieval["input_id"], "role": "fit",
        "features": names, "candidate_budget": 400, "negative_budget": 60, "seed": seed,
        "max_rows": launch["features"]["screening_rows"], "sessions_per_part": 256,
        "code": {n: code[n] for n in ("dataset.py", "features.py")},
    })
    require(
        cache["role"] == "fit" and cache["rows"] >= launch["features"]["screening_rows"]
        and cache["candidate_features"] == len(names), "training-only screen budget differs",
    )
    screen_contract = {
        "cache_id": cache["input_id"], "files": cache["files"],
        "max_retained": launch["features"]["max_retained"], "seed": seed, "folds": 3,
        "pilot_rounds": 100, "correlation_threshold": 0.995,
        "code_sha256": code["screening.py"], "lightgbm": lgb.__version__,
    }
    require(
        read(root / "screening/contract.json") == screen_contract, "screen procedure differs"
    )
    screen = read(root / "screening/selection.json")
    require(
        screen["status"] == "passed" and screen["input_id"] == fingerprint(screen_contract)
        and screen["fitting_rows"] == cache["rows"]
        and screen["fitting_sessions"] == cache["sessions"]
        and screen["candidate_features"] == len(names), "screen input identity differs",
    )
    retained = screen["retained"]
    require(
        len(set(retained)) == len(retained) == screen["retained_count"]
        and set(retained) <= set(names) and 0 < len(retained) <= 128,
        "screen retained features differ",
    )
    require(
        set(variants(tuple(retained))) == set(plan["variants"]),
        "screen cannot support all frozen feature-family ablations",
    )
    pilots = {}
    require(len(screen["pilots"]) == 9, "screen pilot count differs")
    for pilot in screen["pilots"]:
        name = f"pilot-{pilot['fold']}-{pilot['objective']}"
        require(
            read(root / "screening" / f"{name}.json") == pilot
            and pilot["input_id"] == screen["input_id"]
            and pilot["fit_rows"] + pilot["valid_rows"] == cache["rows"],
            "grouped fitting pilot lineage differs",
        )
        digest = sha256_file(root / "screening" / f"{name}.txt")
        require(digest == pilot["sha256"], "screen pilot model checksum differs")
        pilots[name] = digest
    require(
        set(pilots) == {f"pilot-{f}-{o}" for f in range(3) for o in OBJECTIVES},
        "grouped fitting pilot inventory differs",
    )
    caches = {}
    for role in ("fit", "selection"):
        caches[role] = checked_manifest(root, f"{role}_cache", {
            "corpus_id": corpus["input_id"], "retrieval_id": retrieval["input_id"],
            "role": role, "features": retained, "candidate_budget": 400,
            "negative_budget": 60 if role == "fit" else None, "seed": seed,
            "sessions_per_part": 256,
            "code": {n: code[n] for n in ("materialize.py", "features.py", "dataset.py")},
        })
        require(
            caches[role]["sessions"] == corpus["roles"][role]["sessions"]
            and caches[role]["role"] == role and caches[role]["features"] == retained,
            "model cache omits queries or changes features",
        )
    prepared = read(root / "preparation.json")
    require(
        prepared["status"] == "passed" and prepared["protocol_id"] == cell["protocol_id"]
        and prepared["window"] == cell["window"] and prepared["cohort_seed"] == seed
        and prepared["input_id"] == fingerprint({
            k: v for k, v in prepared.items() if k != "input_id"
        }),
        "window preparation identity differs",
    )
    expected_files = {
        p: sha256_file(root / p) for p in INPUT_REPORTS if p != "preparation.json"
    }
    require(prepared["files"] == expected_files, "prepared input receipts differ")
    return {
        "preparation_id": prepared["input_id"], "corpus_id": corpus["input_id"],
        "retrieval_id": retrieval["input_id"], "screening_id": screen["input_id"],
        "screening_sessions": screen["fitting_sessions"],
        "screening_rows": screen["fitting_rows"], "retained_features": len(retained),
        "pilot_sha256": pilots, "files": {p: sha256_file(root / p) for p in INPUT_REPORTS},
        "scope": "Original events; window-specific history; fitting-only grouped screening.",
    }


def verify_window(
    repository: Path, root: Path, source: Path, launch: dict[str, Any], *, logger: logging.Logger
) -> dict[str, Any]:
    start = time.perf_counter()
    plan, _ = load_plan(repository)
    cell = verify_window_launch(repository, launch)
    with Heartbeat(logger, stage="window_input_integrity", interval_seconds=15):
        preparation = verify_window_inputs(repository, root, launch)
    status, report = read(root / "job_status.json"), read(root / "evaluation/report.json")
    require(
        status["status"] == report["status"] == "passed" and status["stage"] == "complete",
        "window study is incomplete",
    )
    for key in ("source_commit", "source_sha256", "training", "robustness", "evaluation_seed"):
        require(status[key] == launch[key], f"completed window differs from launch: {key}")
    require(
        status["lock_sha256"] == plan["frozen_files"]["uv.lock"]
        and status["preparation_id"] == preparation["preparation_id"],
        "completed window environment or preparation differs",
    )
    seal = verify_seal(root)
    contract = read(root / "models/study_contract.json")
    retained = read(root / "screening/selection.json")["retained"]
    expected = {
        "fit_cache_id": read(root / "fit_cache/manifest.json")["input_id"],
        "selection_cache_id": read(root / "selection_cache/manifest.json")["input_id"],
        "selected_features_sha256": sha256_file(root / "screening/selection.json"),
        "corpus_id": preparation["corpus_id"], "config": launch["training"],
        "code": {n: plan["frozen_code"][n] for n in ("study.py", "training.py", "metrics.py")},
        "candidate_budget": 400,
        "feature_variants": {k: list(v) for k, v in variants(tuple(retained)).items()},
        "evaluation_access": "no evaluation labels accepted by this stage",
    }
    require(contract == expected, "frozen window study contract differs")
    require(
        fingerprint(contract) == status["study_id"] == seal["study_id"]
        == read(root / "models/ablation_report.json")["study_id"], "study lineage differs",
    )
    evaluation_contract = {
        "seal_id": seal["seal_id"], "seed": plan["bootstrap_seed"],
        "bootstrap_replicates": plan["bootstrap_replicates"],
        "code": {
            n: plan["frozen_code"][n] for n in ("evaluation.py", "materialize.py", "metrics.py")
        },
        "cohort": "all eligible reserved temporal sessions", "negative_sampling": False,
    }
    require(
        read(root / "evaluation/contract.json") == evaluation_contract
        and fingerprint(evaluation_contract) == report["input_id"] == status["evaluation_id"]
        and report["seal_id"] == seal["seal_id"], "evaluation procedure or lineage differs",
    )
    with Heartbeat(logger, stage="window_original_event_audit", interval_seconds=15):
        original_events = audit_source(
            root, source, threads=launch["training"]["threads"],
            memory_gib=launch["resources"]["memory_gib"],
        )
    with Heartbeat(logger, stage="window_metric_integrity", interval_seconds=15):
        frame, statistics = audit_statistics(root, report)
        selection = audit_ablations(root, seal)
    require(
        selection["native_models"] == 24 and selection["variants"] == 8,
        "planned temporal ablations are missing",
    )
    require(
        statistics["sessions"] == original_events["roles"]["evaluation"],
        "evaluation omits original queries",
    )
    with Heartbeat(logger, stage="window_native_replay", interval_seconds=15):
        native = replay_models(root, frame, sessions=256, seed=plan["cohort_seed"])
    denominator = frame.select([f"denominator_{o}" for o in OBJECTIVES]).to_numpy()
    selected = frame.select([f"selected_{o}_hits" for o in OBJECTIVES]).to_numpy()
    intervals = {}
    with Heartbeat(logger, stage="window_bootstrap_replay", interval_seconds=15):
        for name in ("core", "fusion"):
            actual = paired_bootstrap(
                selected, frame.select([f"{name}_{o}_hits" for o in OBJECTIVES]).to_numpy(),
                denominator, seed=plan["bootstrap_seed"],
                replicates=plan["bootstrap_replicates"],
            )
            expected_interval = report["paired_intervals"][name]
            for key in ("seed", "requested_replicates", "valid_replicates", "method", "scope"):
                require(actual[key] == expected_interval[key], "bootstrap procedure differs")
            for key in ("score", "absolute_gain", "score_interval", "gain_interval"):
                require(
                    bool(np.allclose(actual[key], expected_interval[key], rtol=0, atol=1e-12)),
                    "bootstrap replay differs",
                )
            intervals[name] = actual
    inputs = {
        "protocol_id": cell["protocol_id"], "launch": fingerprint(launch),
        "reference_audit_id": read(repository / "reports/research/audit.json")["audit_id"],
        "code": {
            p.name: sha256_file(p)
            for p in (Path(__file__), Path(__file__).with_name("audit.py"))
        },
        "files": {p: sha256_file(root / p) for p in sorted(PAYLOADS)},
        "window_inputs": preparation["files"],
        "model_sha256": selection["native_model_sha256"],
        "selection_statistic_sha256": selection["selection_statistic_sha256"],
    }
    result = {
        "status": "passed", "input_id": fingerprint(inputs), "inputs": inputs, **cell,
        "evaluation_id": report["input_id"], "seal_id": seal["seal_id"],
        "source_commit": launch["source_commit"], "source_sha256": launch["source_sha256"],
        "python": platform.python_version(), "numpy": np.__version__, "lightgbm": lgb.__version__,
        "preparation": preparation, "source": original_events,
        "statistics": statistics, "ablations": selection, "native_replay": native,
        "paired_intervals": intervals, "prediction_comparison": None,
        "scope": (
            "Fresh original-event reconstruction for this window; "
            "reference fitted inputs not reused."
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }
    result["audit_id"] = fingerprint(result)
    destination = root / "robustness_audit/report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    return result
