"""One canonical, durable entry point for baseline candidate preparation and ranking."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from otto_recsys.cloud.ranking_stage import S3CandidateCheckpoints, S3ModelCheckpoints
from otto_recsys.logging_utils import configure_logging
from otto_recsys.ranking.candidates import CandidateConfig, build_candidates
from otto_recsys.ranking.feature_cache import write_json
from otto_recsys.ranking.lambdarank import RankerConfig
from otto_recsys.ranking.pipeline import run_ranking
from otto_recsys.ranking.reporting import write_ranking_notebook
from otto_recsys.runtime import Heartbeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("candidates", "train", "all"), default="all")
    parser.add_argument(
        "--ranking-cache", type=Path, default=Path("data/interim/ranking_training_cache")
    )
    parser.add_argument(
        "--observed-features", type=Path, default=Path("data/interim/ranking_features")
    )
    parser.add_argument("--covisit-dir", type=Path, default=Path("data/interim/covisit"))
    parser.add_argument("--vectors", type=Path, default=Path("models/item2vec/item_vectors.kv"))
    parser.add_argument("--index", type=Path, default=Path("models/faiss/item.index"))
    parser.add_argument(
        "--candidate-dir", type=Path, default=Path("data/interim/ranking_candidates")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ranking"))
    parser.add_argument(
        "--checkpoint-uri", required=True, help="Existing durable S3 project prefix"
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--batch-sessions", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--max-training-memory-gib", type=float, default=20)
    parser.add_argument("--outer-folds", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--publish-report", action="store_true")
    parser.add_argument("--execute-notebooks", action="store_true")
    args = parser.parse_args()
    if args.execute_notebooks and (not args.publish_report or args.stage == "candidates"):
        parser.error("--execute-notebooks requires --publish-report and a ranking stage")
    started = time.perf_counter()
    logger = configure_logging("ranking", log_dir=args.output_dir / "logs")
    candidate_store = S3CandidateCheckpoints(args.checkpoint_uri.rstrip("/") + "/candidates",
                                            region=args.region, logger=logger)
    model_store = S3ModelCheckpoints(args.checkpoint_uri.rstrip("/") + "/models",
                                    region=args.region, logger=logger)
    succeeded = False
    try:
        logger.info("ranking_pipeline_start", extra={"stage": args.stage})
        if args.stage in {"candidates", "all"}:
            candidate_logger = configure_logging(
                "ranking_candidates", log_dir=args.candidate_dir / "logs"
            )
            candidate_store.logger = candidate_logger
            candidates = build_candidates(
                args.ranking_cache, args.observed_features, args.covisit_dir,
                args.vectors, args.index, args.candidate_dir,
                config=CandidateConfig(candidate_k=args.candidate_k,
                                       batch_sessions=args.batch_sessions, threads=args.threads,
                                       memory_limit=args.memory_limit),
                logger=candidate_logger, checkpoints=candidate_store,
            )
            print(json.dumps(candidates, indent=2), flush=True)
        if args.stage in {"train", "all"}:
            result = run_ranking(
                args.candidate_dir, args.output_dir, outer_folds=tuple(args.outer_folds),
                config=RankerConfig(rounds=args.rounds, threads=args.threads), logger=logger,
                max_memory_gib=args.max_training_memory_gib, checkpoints=model_store,
            )
            if args.publish_report:
                write_json(Path("reports/metrics/ranking_evaluation.json"), result)
                notebook = Path("notebooks/08_ranking_evaluation.ipynb")
                write_ranking_notebook(result, notebook)
                if args.execute_notebooks:
                    environment = Path("artifacts/analysis_environment")
                    interpreter = environment / "bin/python"
                    commands = []
                    if not interpreter.is_file():
                        commands.append(["uv", "venv", str(environment),
                                         "--python", "3.12.13", "--no-project"])
                    commands.extend([
                        ["uv", "pip", "install", "--python", str(interpreter),
                         "-r", "notebooks/requirements.txt"],
                        [str(interpreter), "scripts/execute_notebooks.py"],
                    ])
                    with Heartbeat(logger, stage="ranking_notebooks", interval_seconds=15):
                        for command in commands:
                            subprocess.run(command, check=True)
                    executed = Path("artifacts/notebooks") / notebook.name
                    shutil.copyfile(executed, notebook)
                    model_store.upload(notebook, "notebooks/" + notebook.name)
                    model_store.upload(executed.with_suffix(".json"),
                                       "notebooks/" + executed.with_suffix(".json").name)
            print(json.dumps(result, indent=2), flush=True)
        succeeded = True
    except BaseException:
        logger.exception("ranking_pipeline_failed", extra={
            "elapsed_seconds": round(time.perf_counter() - started, 3)
        })
        raise
    finally:
        logger.info("ranking_pipeline_complete", extra={
            "elapsed_seconds": round(time.perf_counter() - started, 3)
        })
        try:
            if candidate_store.uri is not None:
                candidate_store.publish_logs(args.candidate_dir)
            if model_store.uri is not None:
                model_store.publish(args.output_dir)
        except Exception:
            logger.exception("ranking_final_publication_failed")
            if succeeded:
                raise
    print("OTTO_RANKING_PIPELINE_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
