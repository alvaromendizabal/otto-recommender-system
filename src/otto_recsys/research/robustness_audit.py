"""Verify reference-seed replications without changing the frozen experiment."""

from __future__ import annotations

import copy
import hashlib
import logging
import platform
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.audit import (
    audit_ablations,
    audit_statistics,
    read,
    replay_models,
    require,
)
from otto_recsys.research.dataset import OBJECTIVES, Queries
from otto_recsys.research.evaluation import verify_seal
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.metrics import paired_bootstrap
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.robustness import fingerprint, load_plan, verify_seed_launch
from otto_recsys.runtime import Heartbeat


def verify_reference_contract(
    contract: dict[str, Any], reference_training: dict[str, Any], reference_study_id: str
) -> None:
    """Removing the intended seed change must reproduce the original study identity."""
    original = copy.deepcopy(contract)
    original["config"]["seed"] = reference_training["seed"]
    require(original["config"] == reference_training, "reference training settings changed")
    require(fingerprint(original) == reference_study_id, "reference data or method changed")


def top_items(aids: np.ndarray, scores: np.ndarray) -> list[int]:
    require(
        aids.ndim == scores.ndim == 1
        and aids.size == scores.size
        and np.unique(aids).size == aids.size
        and bool(np.isfinite(scores).all()),
        "prediction comparison requires unique items and aligned finite scores",
    )
    return list(map(int, aids[np.lexsort((aids, -scores))[:20]]))


def compare_predictions(
    root: Path, reference_models: Path, reference_seal: dict[str, Any], *, sessions: int, seed: int
) -> dict[str, Any]:
    """Compare real ranked item IDs, since model headers can change without predictions."""
    require(sessions > 0, "prediction comparison requires a positive session count")
    seal = verify_seal(root)
    queries = Queries(root / "corpus", "evaluation")
    order = sorted(
        range(queries.session.size),
        key=lambda i: hashlib.sha256(f"audit:{seed}:{int(queries.session[i])}".encode()).digest(),
    )[:sessions]
    entries = {"reference": reference_seal["models"], "replication": seal["models"]}
    models = {}
    for label, group in entries.items():
        for objective, entry in group.items():
            path = (
                reference_models / f"{objective}.txt"
                if label == "reference"
                else root / entry["path"]
            )
            require(sha256_file(path) == entry["sha256"], "comparison model checksum differs")
            model = lgb.Booster(model_file=str(path))
            require(model.feature_name() == entry["features"], "comparison feature order differs")
            models[label, objective] = model
    names = tuple(
        dict.fromkeys(
            n for group in entries.values() for m in group.values() for n in m["features"]
        )
    )
    engine = FeatureEngine(root / "retrieval")
    rankings: dict[str, dict[str, list[Any]]] = {
        label: {objective: [] for objective in OBJECTIVES} for label in entries
    }
    changes = {o: {"ordered_top20_changed": 0, "top20_set_changed": 0} for o in OBJECTIVES}
    for i in order:
        prefix = queries.prefix(i)
        pool = engine.candidates(prefix, 400)
        features = engine.transform(prefix, pool, names)
        for objective in OBJECTIVES:
            top = {}
            for label, group in entries.items():
                columns = [names.index(n) for n in group[objective]["features"]]
                scores = models[label, objective].predict(features[:, columns], num_threads=1)
                top[label] = top_items(pool.aid, np.asarray(scores))
                rankings[label][objective].append([prefix.session, top[label]])
            changes[objective]["ordered_top20_changed"] += top["reference"] != top["replication"]
            changes[objective]["top20_set_changed"] += set(top["reference"]) != set(
                top["replication"]
            )
    return {
        "sessions": len(order),
        "seed": seed,
        "session_ids_sha256": fingerprint([int(queries.session[i]) for i in order]),
        "objectives": {
            o: {
                **changes[o],
                **{f"{label}_ranking_sha256": fingerprint(rankings[label][o]) for label in entries},
            }
            for o in OBJECTIVES
        },
        "scope": (
            "Deterministic sampled reserved sessions; equal sampled rankings do not prove global "
            "identity. No labels select the probe sessions."
        ),
    }


def verify_replication(
    repository: Path, root: Path, launch: dict[str, Any], *, logger: logging.Logger
) -> dict[str, Any]:
    """Check every metric part, selection decision and native model before reporting."""
    start = time.perf_counter()
    plan, reference_launch = load_plan(repository)
    replication = verify_seed_launch(repository, launch)
    reference = repository / "reports/research"
    manifest = read(reference / "manifest.json")
    for name in (
        "audit.json",
        "ablations.json",
        "evaluation_seal.json",
        "temporal_manifest.json",
        "screening.json",
    ):
        require(
            sha256_file(reference / name) == manifest["files"][name],
            "reference evidence checksum differs",
        )
    original_audit = read(reference / "audit.json")
    reference_seal = read(reference / "evaluation_seal.json")
    require(original_audit["status"] == "passed", "original source audit did not pass")
    for objective, model in reference_seal["models"].items():
        require(
            sha256_file(reference / "inference_replay" / f"{objective}.txt") == model["sha256"],
            "reference native model checksum differs",
        )
    require(
        sha256_file(root / "corpus/manifest.json") == manifest["files"]["temporal_manifest.json"],
        "reference corpus changed",
    )
    status, report = read(root / "job_status.json"), read(root / "evaluation/report.json")
    require(
        status["status"] == "passed"
        and status["stage"] == "complete"
        and report["status"] == "passed",
        "replication is incomplete",
    )
    for key in ("source_commit", "source_sha256", "training", "robustness", "evaluation_seed"):
        require(status[key] == launch[key], f"completed job differs from launch: {key}")
    require(
        status["lock_sha256"] == plan["frozen_files"]["uv.lock"],
        "completed job used a different lock",
    )
    seal = verify_seal(root)
    for key in ("corpus_id", "retrieval_id", "candidate_budget", "feature_code_sha256"):
        require(seal[key] == reference_seal[key], f"reference evaluation inputs changed: {key}")
    for name, digest in read(root / "retrieval/manifest.json")["files"].items():
        path = root / "retrieval" / name
        require(
            path.resolve().is_relative_to((root / "retrieval").resolve()), "invalid retrieval path"
        )
        require(sha256_file(path) == digest, "historical retrieval checksum differs")
    contract = read(root / "models/study_contract.json")
    require(contract["config"] == launch["training"], "model training seed or settings differ")
    require(
        set(contract["feature_variants"]) == set(plan["variants"]), "planned ablations are missing"
    )
    verify_reference_contract(
        contract, reference_launch["training"], read(reference / "ablations.json")["study_id"]
    )
    ablations = read(root / "models/ablation_report.json")
    require(
        fingerprint(contract) == ablations["study_id"] == seal["study_id"] == status["study_id"],
        "ablation lineage differs",
    )
    evaluation_contract = read(root / "evaluation/contract.json")
    require(
        evaluation_contract["seed"] == plan["bootstrap_seed"]
        and evaluation_contract["bootstrap_replicates"] == plan["bootstrap_replicates"]
        and evaluation_contract["negative_sampling"] is False
        and evaluation_contract["seal_id"] == seal["seal_id"]
        and evaluation_contract["code"]
        == {n: plan["frozen_code"][n] for n in ("evaluation.py", "materialize.py", "metrics.py")},
        "evaluation procedure differs from the frozen protocol",
    )
    require(
        fingerprint(evaluation_contract) == report["input_id"] == status["evaluation_id"],
        "evaluation identity differs",
    )
    require(report["seal_id"] == seal["seal_id"], "evaluation used a different selection seal")
    inputs = {
        "protocol_id": replication["protocol_id"],
        "launch": fingerprint(launch),
        "reference_audit_id": original_audit["audit_id"],
        "code": {
            p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("audit.py"))
        },
        "files": {
            name: sha256_file(root / name)
            for name in (
                "job_status.json",
                "evaluation/report.json",
                "evaluation/contract.json",
                "evaluation_seal.json",
                "models/study_contract.json",
                "models/ablation_report.json",
            )
        },
    }
    with Heartbeat(logger, stage="replication_integrity", interval_seconds=15):
        frame, statistics = audit_statistics(root, report)
        selection = audit_ablations(root, seal)
    require(
        selection["native_models"] == 24 and selection["variants"] == 8,
        "replication omitted planned models",
    )
    require(
        statistics["sessions"]
        == read(reference / "temporal_manifest.json")["roles"]["evaluation"]["sessions"],
        "reserved cohort size differs",
    )
    inputs["model_sha256"] = selection["native_model_sha256"]
    inputs["selection_statistic_sha256"] = selection["selection_statistic_sha256"]
    identity = fingerprint(inputs)
    destination = root / "robustness_audit/report.json"
    if destination.exists():
        previous = read(destination)
        require(
            previous["input_id"] == identity and previous["status"] == "passed",
            "verification receipt belongs to different evidence",
        )
        require(
            previous["audit_id"]
            == fingerprint({k: v for k, v in previous.items() if k != "audit_id"}),
            "verification receipt was changed",
        )
        logger.info("replication_verification_reused", extra={"audit_id": previous["audit_id"]})
        return previous
    with Heartbeat(logger, stage="replication_native_replay", interval_seconds=15):
        native = replay_models(root, frame, sessions=256, seed=plan["cohort_seed"])
        predictions = compare_predictions(
            root,
            reference / "inference_replay",
            reference_seal,
            sessions=256,
            seed=plan["cohort_seed"],
        )
    denominator = frame.select([f"denominator_{o}" for o in OBJECTIVES]).to_numpy()
    selected = frame.select([f"selected_{o}_hits" for o in OBJECTIVES]).to_numpy()
    intervals = {}
    with Heartbeat(logger, stage="replication_bootstrap_replay", interval_seconds=15):
        for name in ("core", "fusion"):
            hits = frame.select([f"{name}_{o}_hits" for o in OBJECTIVES]).to_numpy()
            actual = paired_bootstrap(
                selected,
                hits,
                denominator,
                seed=plan["bootstrap_seed"],
                replicates=plan["bootstrap_replicates"],
            )
            expected = report["paired_intervals"][name]
            for key in ("seed", "requested_replicates", "valid_replicates", "method", "scope"):
                require(actual[key] == expected[key], "bootstrap procedure or sample count differs")
            for key in ("score", "absolute_gain", "score_interval", "gain_interval"):
                require(
                    bool(np.allclose(actual[key], expected[key], rtol=0, atol=1e-12)),
                    "bootstrap interval differs from replay",
                )
            intervals[name] = actual
    result = {
        "status": "passed",
        "input_id": identity,
        "inputs": inputs,
        **replication,
        "evaluation_id": report["input_id"],
        "seal_id": seal["seal_id"],
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "statistics": statistics,
        "ablations": selection,
        "native_replay": native,
        "prediction_comparison": predictions,
        "paired_intervals": intervals,
        "reused_source_audit": {
            "audit_id": original_audit["audit_id"],
            "corpus_manifest_sha256": manifest["files"]["temporal_manifest.json"],
            "scope": (
                "Original event reconstruction reused for the byte-identical reference corpus. "
                "New metric counts, native models, selection decisions and sampled predictions "
                "were verified separately."
            ),
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    result["audit_id"] = fingerprint(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    return result
