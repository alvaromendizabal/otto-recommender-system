from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from otto_two_tower.checkpoint import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
    save_state_dict_atomic,
    write_json_atomic,
)
from otto_two_tower.config import DataConfig, ModelConfig, TrainConfig, config_payload
from otto_two_tower.data import HardNegativeBatchStream, ItemVocabulary, PackedSessionStore
from otto_two_tower.logging_utils import configure_logging
from otto_two_tower.model import TwoTowerModel
from otto_two_tower.telemetry import TrainingHeartbeat
from otto_two_tower.trainer import cosine_warmup_lambda, run_epoch


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-cache", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--item-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--valid-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    overall_started = time.perf_counter()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("two_tower_training", output_dir / "logs")

    data_config = DataConfig(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        validation_fold=args.validation_fold,
        train_rows=args.train_rows,
        valid_rows=args.valid_rows,
        seed=args.seed,
    )
    model_config = ModelConfig()
    train_config = TrainConfig(
        epochs=args.epochs,
        checkpoint_steps=args.checkpoint_steps,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    data_config.validate()
    model_config.validate()
    train_config.validate()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; use --allow-cpu only for local smoke tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(data_config.seed)

    ranking_manifest = _load_json(args.ranking_cache / "manifest.json")
    negative_manifest = _load_json(args.hard_negatives / "manifest.json")
    item_manifest = _load_json(args.item_data / "manifest.json")
    if ranking_manifest.get("validation_manifest_id") != negative_manifest.get(
        "validation_manifest_id"
    ):
        raise RuntimeError("ranking cache and hard-negative validation manifests differ")
    input_payload = {
        "code_commit": args.code_commit,
        "ranking": ranking_manifest,
        "hard_negatives": negative_manifest,
        "items": item_manifest,
        "config": config_payload(data_config, model_config, train_config),
    }
    input_id = _canonical_sha256(input_payload)

    logger.info(
        "training_start",
        extra={
            "event": "training_start",
            "stage": "initialization",
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "input_id": input_id,
        },
    )

    vocabulary = ItemVocabulary.load(args.item_data)
    if vocabulary.vectors.shape[1] != model_config.embedding_dim:
        raise RuntimeError("item-vector dimension does not match model configuration")
    session_store = PackedSessionStore.from_parquet(
        args.ranking_cache,
        vocabulary,
        max_seq_len=data_config.max_seq_len,
        time_buckets=model_config.time_buckets,
    )
    stream = HardNegativeBatchStream(
        args.hard_negatives,
        vocabulary,
        batch_size=data_config.batch_size,
        validation_fold=data_config.validation_fold,
        seed=data_config.seed,
    )
    pretrained = torch.from_numpy(np.asarray(vocabulary.vectors, dtype=np.float32))
    model = TwoTowerModel(
        pretrained,
        padding_index=vocabulary.padding_index,
        config=model_config,
        max_seq_len=data_config.max_seq_len,
    ).to(device)

    sparse_parameters = [model.item_embedding.weight]
    dense_parameters = [
        parameter for name, parameter in model.named_parameters() if name != "item_embedding.weight"
    ]
    sparse_optimizer = torch.optim.SparseAdam(sparse_parameters, lr=train_config.sparse_lr)
    dense_optimizer = torch.optim.AdamW(
        dense_parameters,
        lr=train_config.dense_lr,
        weight_decay=train_config.weight_decay,
    )

    total_rows = int(negative_manifest["output_rows"])
    estimated_train_rows = math.ceil(total_rows * (data_config.folds - 1) / data_config.folds)
    effective_train_rows = (
        min(estimated_train_rows, data_config.train_rows)
        if data_config.train_rows is not None
        else estimated_train_rows
    )
    train_batches = max(math.ceil(effective_train_rows / data_config.batch_size), 1)
    total_steps = train_batches * train_config.epochs
    warmup_steps = int(total_steps * train_config.warmup_fraction)

    def scheduler_function(step: int) -> float:
        return cosine_warmup_lambda(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
    dense_scheduler = torch.optim.lr_scheduler.LambdaLR(dense_optimizer, scheduler_function)
    sparse_scheduler = torch.optim.lr_scheduler.LambdaLR(sparse_optimizer, scheduler_function)

    checkpoint_path = output_dir / "checkpoint.pt"
    run_contract = {
        "input_id": input_id,
        "code_commit": args.code_commit,
        "validation_manifest_id": ranking_manifest["validation_manifest_id"],
        "config": input_payload["config"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    write_json_atomic(run_contract, output_dir / "run_contract.json")
    state = TrainingState()
    if args.resume and not checkpoint_path.is_file():
        raise RuntimeError("--resume requested but checkpoint.pt is missing")
    if args.resume:
        state = load_checkpoint(
            checkpoint_path,
            model=model,
            dense_optimizer=dense_optimizer,
            sparse_optimizer=sparse_optimizer,
            dense_scheduler=dense_scheduler,
            sparse_scheduler=sparse_scheduler,
            expected_input_id=input_id,
            map_location=device,
        )
        logger.info(
            "checkpoint_resumed",
            extra={
                "event": "checkpoint_resumed",
                "stage": "initialization",
                "epoch": state.epoch,
                "step": state.global_step,
                "next_batch": state.next_batch,
            },
        )

    progress: dict[str, Any] = {
        "epoch": state.epoch,
        "step": state.global_step,
        "examples": 0,
        "loss": None,
        "mrr": None,
        "hit10": None,
    }

    def checkpoint_callback(current: TrainingState) -> None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            dense_optimizer=dense_optimizer,
            sparse_optimizer=sparse_optimizer,
            dense_scheduler=dense_scheduler,
            sparse_scheduler=sparse_scheduler,
            state=current,
            input_id=input_id,
            config=input_payload["config"],
        )

    history = state.history
    with TrainingHeartbeat(
        logger,
        stage="two_tower_training",
        interval_seconds=train_config.heartbeat_seconds,
        progress_provider=lambda: dict(progress),
    ):
        for epoch in range(state.epoch, train_config.epochs):
            state.epoch = epoch
            train_metrics = run_epoch(
                model=model,
                session_store=session_store,
                stream=stream,
                device=device,
                epoch=epoch,
                training=True,
                dense_optimizer=dense_optimizer,
                sparse_optimizer=sparse_optimizer,
                dense_scheduler=dense_scheduler,
                sparse_scheduler=sparse_scheduler,
                in_batch_weight=train_config.in_batch_weight,
                dense_grad_clip=train_config.dense_grad_clip,
                sparse_grad_clip=train_config.sparse_grad_clip,
                max_rows=data_config.train_rows,
                start_batch=state.next_batch,
                state=state,
                checkpoint_callback=checkpoint_callback,
                checkpoint_steps=train_config.checkpoint_steps,
                progress=progress,
                bf16=train_config.bf16,
            )
            state.next_batch = 0
            valid_metrics = run_epoch(
                model=model,
                session_store=session_store,
                stream=stream,
                device=device,
                epoch=epoch,
                training=False,
                dense_optimizer=None,
                sparse_optimizer=None,
                dense_scheduler=None,
                sparse_scheduler=None,
                in_batch_weight=train_config.in_batch_weight,
                dense_grad_clip=train_config.dense_grad_clip,
                sparse_grad_clip=train_config.sparse_grad_clip,
                max_rows=data_config.valid_rows,
                start_batch=0,
                state=state,
                checkpoint_callback=None,
                checkpoint_steps=train_config.checkpoint_steps,
                progress=progress,
                bf16=train_config.bf16,
            )
            record = {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "valid": asdict(valid_metrics),
            }
            history.append(record)
            state.history = history
            logger.info(
                "epoch_complete",
                extra={
                    "event": "epoch_complete",
                    "stage": "two_tower_training",
                    "epoch": epoch,
                    "loss": round(valid_metrics.loss, 6),
                    "mrr": round(valid_metrics.mrr, 6),
                    "hit10": round(valid_metrics.hit10, 6),
                    "elapsed_seconds": round(
                        train_metrics.elapsed_seconds + valid_metrics.elapsed_seconds,
                        3,
                    ),
                },
            )

            improved = valid_metrics.loss < state.best_valid_loss
            if improved:
                state.best_valid_loss = valid_metrics.loss
                state.epochs_without_improvement = 0
                save_state_dict_atomic(model.state_dict(), output_dir / "best_model.pt")
            else:
                state.epochs_without_improvement += 1

            state.epoch = epoch + 1
            checkpoint_callback(state)
            write_json_atomic(
                {"input_id": input_id, "history": history},
                output_dir / "metrics.json",
            )
            if state.epochs_without_improvement > train_config.early_stopping_patience:
                logger.info(
                    "early_stopping",
                    extra={
                        "event": "early_stopping",
                        "stage": "two_tower_training",
                        "epoch": epoch,
                    },
                )
                break

    total_elapsed = time.perf_counter() - overall_started
    training_manifest = {
        "input_id": input_id,
        "validation_manifest_id": ranking_manifest["validation_manifest_id"],
        "validation_fold": data_config.validation_fold,
        "code_commit": args.code_commit,
        "config": input_payload["config"],
        "best_valid_loss": state.best_valid_loss,
        "global_step": state.global_step,
        "elapsed_seconds": round(total_elapsed, 3),
        "history": history,
    }
    write_json_atomic(training_manifest, output_dir / "training_manifest.json")
    logger.info(
        "training_complete",
        extra={
            "event": "training_complete",
            "stage": "two_tower_training",
            "status": "passed",
            "step": state.global_step,
            "elapsed_seconds": round(total_elapsed, 3),
        },
    )
    print(json.dumps(training_manifest, indent=2, sort_keys=True))
    print("OTTO_TWO_TOWER_TRAINING_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
