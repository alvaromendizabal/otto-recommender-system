from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
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
    history: list[dict[str, Any]] = field(default_factory=list)


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


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_state_dict_atomic(state_dict: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, temporary)
    os.replace(temporary, path)


def progress_payload(*, state: TrainingState, input_id: str) -> dict[str, Any]:
    return {
        "input_id": input_id,
        "epoch": state.epoch,
        "next_batch": state.next_batch,
        "global_step": state.global_step,
        "best_valid_loss": state.best_valid_loss,
        "epochs_without_improvement": state.epochs_without_improvement,
    }


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
    write_json_atomic(
        progress_payload(state=state, input_id=input_id),
        path.parent / "progress.json",
    )


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
