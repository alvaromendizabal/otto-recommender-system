from __future__ import annotations

import json
import os
from collections.abc import Iterable
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


CHECKPOINT_FORMAT_VERSION = 2


def _rng_state_to_bytes(value: torch.Tensor) -> bytes:
    normalized = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return normalized.numpy().tobytes()


def _rng_state_to_cpu_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.asarray(value, dtype=np.uint8).copy()).contiguous()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy()).contiguous()
    if isinstance(value, Iterable) and not isinstance(value, (str, dict)):
        return torch.tensor(list(value), dtype=torch.uint8, device="cpu").contiguous()
    raise TypeError(f"unsupported RNG state type: {type(value)!r}")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch": _rng_state_to_bytes(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        state["cuda"] = [
            _rng_state_to_bytes(cuda_state)
            for cuda_state in torch.cuda.get_rng_state_all()
        ]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_rng_state_to_cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            _rng_state_to_cpu_byte_tensor(cuda_state)
            for cuda_state in state["cuda"]
        ]
        device_count = torch.cuda.device_count()
        if len(cuda_states) != device_count:
            raise RuntimeError(
                "checkpoint CUDA RNG state count does not match current CUDA device count: "
                f"checkpoint={len(cuda_states)} current={device_count}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


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
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
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
    checkpoint_format_version = int(payload.get("checkpoint_format_version", 1))
    if checkpoint_format_version not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise RuntimeError(
            f"unsupported checkpoint format version: {checkpoint_format_version}"
        )
    if payload.get("input_id") != expected_input_id:
        raise RuntimeError("checkpoint input_id does not match current training inputs")
    model.load_state_dict(payload["model"])
    dense_optimizer.load_state_dict(payload["dense_optimizer"])
    sparse_optimizer.load_state_dict(payload["sparse_optimizer"])
    dense_scheduler.load_state_dict(payload["dense_scheduler"])
    sparse_scheduler.load_state_dict(payload["sparse_scheduler"])
    restore_rng_state(payload["rng"])
    return TrainingState(**payload["state"])
