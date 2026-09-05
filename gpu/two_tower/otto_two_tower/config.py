from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

OBJECTIVES: tuple[str, ...] = ("clicks", "carts", "orders")
OBJECTIVE_TO_ID: dict[str, int] = {name: index for index, name in enumerate(OBJECTIVES)}


@dataclass(frozen=True)
class DataConfig:
    max_seq_len: int = 50
    batch_size: int = 256
    validation_fold: int = 0
    folds: int = 5
    train_rows: int | None = None
    valid_rows: int | None = None
    seed: int = 20260905

    def validate(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if not 0 <= self.validation_fold < self.folds:
            raise ValueError("validation_fold is outside the fold range")
        for name, value in (("train_rows", self.train_rows), ("valid_rows", self.valid_rows)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.10
    time_buckets: int = 32
    objective_count: int = 3
    temperature: float = 0.07

    def validate(self) -> None:
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.time_buckets < 2:
            raise ValueError("time_buckets must be at least 2")
        if self.objective_count != len(OBJECTIVES):
            raise ValueError("objective_count must match the OTTO objectives")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    dense_lr: float = 2.0e-4
    sparse_lr: float = 8.0e-4
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.06
    dense_grad_clip: float = 1.0
    sparse_grad_clip: float = 5.0
    in_batch_weight: float = 0.35
    checkpoint_steps: int = 500
    heartbeat_seconds: float = 30.0
    early_stopping_patience: int = 2
    bf16: bool = True

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.dense_lr <= 0 or self.sparse_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.dense_grad_clip <= 0 or self.sparse_grad_clip <= 0:
            raise ValueError("gradient clipping thresholds must be positive")
        if self.in_batch_weight < 0:
            raise ValueError("in_batch_weight cannot be negative")
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")


def config_payload(
    data: DataConfig,
    model: ModelConfig,
    train: TrainConfig,
) -> dict[str, Any]:
    return {
        "data": asdict(data),
        "model": asdict(model),
        "train": asdict(train),
    }
