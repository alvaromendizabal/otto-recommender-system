"""Run the public reference from raw JSONL, without private workspace dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def artifact_inventory(root: Path) -> dict[str, str]:
    """Seal every completed artifact; logs remain explicitly outside the receipt."""
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("invalid artifact path: symbolic links are not artifacts")
        name = path.relative_to(root)
        if path.is_file() and name.as_posix() != "reproduction.json" and "logs" not in name.parts:
            inventory[name.as_posix()] = digest(path)
    return inventory


def verify(root: Path) -> dict[str, Any]:
    receipt_path = root / "reproduction.json"
    if receipt_path.is_symlink():
        raise ValueError("invalid artifact path: receipt is a symbolic link")
    receipt = json.loads(receipt_path.read_text())
    files = receipt.get("files")
    if receipt.get("status") != "passed" or not isinstance(files, dict) or not files:
        raise ValueError("a complete reproduction receipt is required")
    for name, expected in files.items():
        if not isinstance(name, str):
            raise ValueError("invalid artifact path")
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or relative.as_posix() != name
            or name == "reproduction.json"
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
            or any(
                (root / Path(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts) + 1)
            )
        ):
            raise ValueError("invalid artifact path")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("invalid artifact checksum")
    actual = artifact_inventory(root)
    if set(files) != set(actual):
        raise ValueError("unexpected or missing reproduction artifacts")
    for name, expected in files.items():
        if actual[name] != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")
    return receipt


def source_identity() -> dict[str, str]:
    paths = [ROOT / "reproduce.py", Path(__file__), ROOT / "uv.lock"]
    paths += sorted((ROOT / "src/otto_recsys").rglob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def fixture(path: Path) -> None:
    """Invented sessions, four disjoint chronological roles; never user records."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    base, day = 1_661_000_000_000, 86_400_000
    with temp.open("w") as stream:
        for period, count in enumerate((256, 128, 64, 64)):
            for i in range(count):
                length = 6 + i % 5
                events = []
                for j in range(length):
                    kind = (
                        "clicks" if j < length - 2 else ("carts" if j == length - 2 else "orders")
                    )
                    aid = 1 + (i * 7 + j // 2) % 128
                    if period == 3 and kind == "orders" and i % 11 == 0:
                        aid = 999
                    events.append(
                        {
                            "aid": aid,
                            "ts": base + period * day + i * 60_000 + j * 1000,
                            "type": kind,
                        }
                    )
                stream.write(json.dumps({"session": period * 10_000 + i, "events": events}) + "\n")
    temp.replace(path)


def export_predictions(root: Path, logger: logging.Logger, threads: int) -> dict[str, Any]:
    """Score observed prefixes with sealed models; no target columns enter the scorer."""
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from otto_recsys.research.evaluation import verify_seal
    from otto_recsys.research.features import FeatureEngine, Prefix
    from otto_recsys.research.inference import validate_submission

    seal = verify_seal(root)
    engine = FeatureEngine(root / "retrieval")
    observed = (
        pl.read_parquet(root / "corpus/observed.parquet")
        .filter(pl.col("split_role") == "evaluation")
        .sort("session", "event_index")
    )
    models = {
        task: lgb.Booster(model_file=str(root / record["path"]))
        for task, record in seal["models"].items()
    }
    destination = root / "recommendations.csv"
    with destination.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["session_type", "labels"])
        for (session,), frame in observed.group_by("session", maintain_order=True):
            prefix = Prefix(
                int(session),
                frame["aid"].to_numpy(),
                frame["ts"].to_numpy(),
                frame["event_type"].to_numpy(),
            )
            candidates = engine.candidates(prefix, 400)
            if candidates.aid.size < 20:
                raise ValueError("at least 20 distinct candidates are required for this export")
            for task, model in models.items():
                names = tuple(seal["models"][task]["features"])
                x = engine.transform(prefix, candidates, names)
                score = np.asarray(model.predict(x, num_threads=threads))
                if not np.isfinite(score).all():
                    raise ValueError("non-finite model scores")
                order = np.lexsort((candidates.aid, -score))[:20]
                writer.writerow([f"{session}_{task}", " ".join(map(str, candidates.aid[order]))])
    destination.with_suffix(".tmp").replace(destination)
    ids = np.sort(observed["session"].unique().to_numpy())
    report = validate_submission(destination, ids)
    logger.info("public_reference_export_complete")
    return {
        "sessions": int(ids.size),
        "rows": int(ids.size * 3),
        "sha256": digest(destination),
        "validation": report,
        "scope": "Evaluation-prefix recommendations; NOT a competition submission",
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    from run_research import load_protocol

    from otto_recsys.data.convert import convert_jsonl_to_parquet
    from otto_recsys.data.manifest import build_manifest, write_manifest
    from otto_recsys.data.raw_validation import validate_jsonl
    from otto_recsys.research.dataset import build_cache
    from otto_recsys.research.evaluation import run_evaluation
    from otto_recsys.research.features import feature_catalog
    from otto_recsys.research.materialize import model_cache
    from otto_recsys.research.protocol import TemporalProtocol, build_temporal_corpus
    from otto_recsys.research.retrievers import build_retrievers
    from otto_recsys.research.screening import core_features, screen_cache
    from otto_recsys.research.study import run_ablations

    root = args.output.resolve()
    if root == ROOT or ROOT.is_relative_to(root):
        raise ValueError("output must not be the repository or one of its parents")
    if root.exists() and any(root.iterdir()) and not (root / "reproduction_contract.json").exists():
        raise ValueError(
            "output is not empty and has no reproduction contract; use a new directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    raw = args.train_jsonl.resolve() if args.mode == "research" else root / "raw/train.jsonl"
    if args.mode == "train":
        fixture(raw)
    if not raw.is_file():
        raise FileNotFoundError("Provide an existing uncompressed training JSONL file")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "polars", "pyarrow", "duckdb", "lightgbm", "scipy")
    }
    contract = {
        "mode": args.mode,
        "source": source_identity(),
        "raw_sha256": digest(raw),
        "config_sha256": digest(args.config),
        "versions": versions,
        "threads": args.threads,
        "workers": args.workers,
        "memory_gib": args.memory_gib,
    }
    contract_path = root / "reproduction_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError(
            "output belongs to a different source/data/runtime contract; use a new directory"
        )
    write_json(contract_path, contract)
    if (root / "reproduction.json").exists():
        result = verify(root)
        print("REUSED: all committed outputs verified; no fitting or feature rebuild")
        return result
    logger = logging.getLogger("public_reference")
    validate_jsonl(raw, logger=logger)
    raw_manifest = root / "raw_manifest.json"
    if not raw_manifest.exists():
        write_manifest(
            build_manifest(
                [raw],
                source="local-reader-provided"
                if args.mode == "research"
                else "synthetic-public-fixture",
                logger=logger,
            ),
            raw_manifest,
        )
    converted = convert_jsonl_to_parquet(
        raw,
        root / "processed",
        raw_manifest,
        logger=logger,
        events_per_part=1_000_000,
        heartbeat_seconds=15,
    )
    if converted.status != "complete":
        raise ValueError("raw conversion did not complete")
    if args.mode == "research":
        protocol, config = load_protocol(args.config)
    else:
        day, base = 86_400_000, 1_661_000_000_000
        protocol = TemporalProtocol(
            history_end=base + day,
            fit_end=base + 2 * day,
            selection_end=base + 3 * day,
            evaluation_end=base + 4 * day,
            seed=20260908,
            fit_sessions=128,
            selection_sessions=64,
        )
        config = {
            "retrieval": {
                "history_tail": 30,
                "position_window": 3,
                "time_window_hours": 24,
                "source_partitions": 2,
                "neighbors_per_score": 40,
            },
            "training": {"rounds": 12, "patience": 5, "negative_budget": 30},
            "features": {"max_retained": 128, "screening_rows": 500_000},
        }
    print("STAGE 1/6: raw conversion and temporal corpus", flush=True)
    build_temporal_corpus(
        root / "processed",
        root / "corpus",
        protocol,
        logger=logger,
        threads=args.threads,
        memory_gib=args.memory_gib,
    )
    print("STAGE 2/6: historical co-visitation retrieval", flush=True)
    build_retrievers(
        root / "corpus",
        root / "retrieval",
        config["retrieval"],
        logger=logger,
        threads=args.threads,
        memory_gib=args.memory_gib,
    )
    print("STAGE 3/6: fitting-only feature preparation", flush=True)
    if args.mode == "research":
        build_cache(
            root / "corpus",
            root / "retrieval",
            root / "screening_cache",
            role="fit",
            names=tuple(f.name for f in feature_catalog()),
            candidate_budget=400,
            negative_budget=config["training"]["negative_budget"],
            seed=protocol.seed,
            max_rows=config["features"]["screening_rows"],
            logger=logger,
        )
        selected = screen_cache(
            root / "screening_cache",
            root / "screening",
            max_retained=config["features"]["max_retained"],
            seed=protocol.seed,
            logger=logger,
            threads=args.threads,
        )
        names = tuple(selected["retained"])
    else:
        names = (
            *core_features(),
            "hist_all_all",
            "hist_orders_h24",
            "intent_repeat_carts",
            "graph_time_all_n3_uniform_sum",
            "graph_order_all_n3_uniform_max",
        )
        write_json(
            root / "screening/selection.json",
            {"retained": names, "selection": "Small fixture profile; not the historical model"},
        )
    for role in ("fit", "selection"):
        model_cache(
            root / "corpus",
            root / "retrieval",
            root / f"{role}_cache",
            role=role,
            names=names,
            budget=400,
            negatives=config["training"]["negative_budget"],
            seed=protocol.seed,
            workers=args.workers,
            logger=logger,
        )
    print("STAGE 4/6: native task rankers, ablations and selection seal", flush=True)
    study = run_ablations(
        root, {**config["training"], "threads": args.threads, "seed": protocol.seed}, logger=logger
    )
    print("STAGE 5/6: complete temporal evaluation after model sealing", flush=True)
    evaluation = run_evaluation(
        root,
        seed=protocol.seed,
        workers=args.workers,
        threads=args.threads,
        logger=logger,
        bootstrap_replicates=100 if args.mode == "train" else 1000,
    )
    print("STAGE 6/6: target-free scoring, CSV validation and artifact sealing", flush=True)
    exported = export_predictions(root, logger, args.threads)
    result = {
        "schema_version": 1,
        "status": "passed",
        "mode": args.mode,
        "synthetic_data": args.mode == "train",
        "new_competition_score": False,
        "full_neural_release_reproduced": False,
        "feature_count": len(names),
        "training_variants": len(study["variants"]),
        "task_models": len(study["variants"]) * 3,
        "evaluation_sessions": evaluation["sessions"],
        "export": exported,
        "scores": evaluation["scores"],
        "versions": versions,
        "python": platform.python_version(),
        "files": {},
    }
    write_json(root / "summary.json", {k: v for k, v in result.items() if k != "files"})
    result["files"] = artifact_inventory(root)
    write_json(root / "reproduction.json", result)
    return verify(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review", "train", "research", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-gib", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.mode == "review":
            import subprocess

            for script, extra in (
                ("review_portfolio.py", []),
                ("run_public_demo.py", ["--output", str(args.output / "demo")]),
                ("run_public_demo.py", ["--output", str(args.output / "demo"), "--check"]),
            ):
                subprocess.run(
                    [sys.executable, "-S", str(ROOT / "scripts" / script), *extra], check=True
                )
            print("OTTO_REPRODUCTION_PASSED: evidence audit and synthetic demo; no model training")
        else:
            result = verify(args.output) if args.mode == "verify" else train(args)
            print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
            print("OTTO_REPRODUCTION_PASSED: public reference artifacts verified")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Use the locked ML environment. Completed outputs are preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
