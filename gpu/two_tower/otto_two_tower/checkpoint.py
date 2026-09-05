from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TrainingState:
    epoch: int = 0
    next_batch: int = 0
    global_step: int = 0
    best_valid_loss: float = float("inf")
    epochs_without_improvement: int = 0


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    dense_optimizer: torch.optim.Optimizer,
    sparse_optimizer: torch.optim.Optimizer,
    dense_scheduler: torch.optim.lr_scheduler.LRScheduler,
    sparse_scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    input_id: str,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "dense_optimizer": dense_optimizer.state_dict(),
        "sparse_optimizer": sparse_optimizer.state_dict(),
        "dense_scheduler": dense_scheduler.state_dict(),
        "sparse_scheduler": sparse_scheduler.state_dict(),
        "state": asdict(state),
        "rng": capture_rng_state(),
        "input_id": input_id,
        "config": config,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    dense_optimizer: torch.optim.Optimizer,
    sparse_optimizer: torch.optim.Optimizer,
    dense_scheduler: torch.optim.lr_scheduler.LRScheduler,
    sparse_scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_input_id: str,
    map_location: torch.device,
) -> TrainingState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("input_id") != expected_input_id:
        raise RuntimeError("checkpoint input_id does not match current training inputs")
    model.load_state_dict(payload["model"])
    dense_optimizer.load_state_dict(payload["dense_optimizer"])
    sparse_optimizer.load_state_dict(payload["sparse_optimizer"])
    dense_scheduler.load_state_dict(payload["dense_scheduler"])
    sparse_scheduler.load_state_dict(payload["sparse_scheduler"])
    restore_rng_state(payload["rng"])
    return TrainingState(**payload["state"])
