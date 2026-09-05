from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from .checkpoint import TrainingState
from .data import HardNegativeBatchStream, PackedSessionStore
from .loss import objective_conditioned_contrastive_loss
from .model import TwoTowerModel


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    explicit_loss: float
    in_batch_loss: float
    mrr: float
    hit1: float
    hit10: float
    rows: int
    batches: int
    elapsed_seconds: float


def cosine_warmup_lambda(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_sparse_gradient_(parameter: torch.nn.Parameter, max_norm: float) -> None:
    gradient = parameter.grad
    if gradient is None or not gradient.is_sparse:
        return
    gradient = gradient.coalesce()
    values = gradient._values()
    norm = values.norm()
    if torch.isfinite(norm) and norm > max_norm:
        values.mul_(max_norm / (norm + 1e-12))
    parameter.grad = gradient


def run_epoch(
    *,
    model: TwoTowerModel,
    session_store: PackedSessionStore,
    stream: HardNegativeBatchStream,
    device: torch.device,
    epoch: int,
    training: bool,
    dense_optimizer: torch.optim.Optimizer | None,
    sparse_optimizer: torch.optim.Optimizer | None,
    dense_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    sparse_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    in_batch_weight: float,
    dense_grad_clip: float,
    sparse_grad_clip: float,
    max_rows: int | None,
    start_batch: int,
    state: TrainingState,
    checkpoint_callback: Callable[[TrainingState], None] | None,
    checkpoint_steps: int,
    progress: dict[str, Any],
    bf16: bool,
) -> EpochMetrics:
    model.train(training)
    started = time.perf_counter()
    totals = {
        "loss": 0.0,
        "explicit": 0.0,
        "in_batch": 0.0,
        "mrr": 0.0,
        "hit1": 0.0,
        "hit10": 0.0,
        "rows": 0,
        "batches": 0,
    }

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in stream.iter_batches(
            epoch=epoch,
            training=training,
            device=device,
            max_rows=max_rows,
            start_batch=start_batch,
        ):
            if training:
                assert dense_optimizer is not None
                assert sparse_optimizer is not None
                dense_optimizer.zero_grad(set_to_none=True)
                sparse_optimizer.zero_grad(set_to_none=True)

            sequence = session_store.batch(batch.session_ids, device)
            autocast_enabled = device.type == "cuda" and bf16
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                query = model.encode_session(sequence, batch.objective_ids)
                positive = model.encode_candidates(batch.positive_indices, batch.objective_ids)
                negatives = model.encode_candidates(batch.negative_indices, batch.objective_ids)
                sessions_tensor = torch.from_numpy(batch.session_ids).to(
                    device=device,
                    non_blocking=True,
                )
                result = objective_conditioned_contrastive_loss(
                    query,
                    positive,
                    negatives,
                    session_ids=sessions_tensor,
                    objective_ids=batch.objective_ids,
                    positive_aids=batch.positive_aids,
                    scale=model.scale(),
                    in_batch_weight=in_batch_weight,
                )

            if training:
                if dense_optimizer is None or sparse_optimizer is None:
                    raise RuntimeError("training requires both dense and sparse optimizers")
                result.loss.backward()
                dense_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if name != "item_embedding.weight" and parameter.grad is not None
                ]
                clip_grad_norm_(dense_parameters, dense_grad_clip)
                clip_sparse_gradient_(model.item_embedding.weight, sparse_grad_clip)
                dense_optimizer.step()
                sparse_optimizer.step()
                if dense_scheduler is not None:
                    dense_scheduler.step()
                if sparse_scheduler is not None:
                    sparse_scheduler.step()
                state.global_step += 1
                state.next_batch = batch_index + 1

            rows = batch.size
            totals["loss"] += float(result.loss.detach()) * rows
            totals["explicit"] += float(result.explicit_loss.detach()) * rows
            totals["in_batch"] += float(result.in_batch_loss.detach()) * rows
            totals["mrr"] += float(result.mrr.detach()) * rows
            totals["hit1"] += float(result.hit1.detach()) * rows
            totals["hit10"] += float(result.hit10.detach()) * rows
            totals["rows"] += rows
            totals["batches"] += 1

            progress.update(
                {
                    "epoch": epoch,
                    "step": state.global_step,
                    "examples": totals["rows"],
                    "loss": round(totals["loss"] / totals["rows"], 5),
                    "mrr": round(totals["mrr"] / totals["rows"], 5),
                    "hit10": round(totals["hit10"] / totals["rows"], 5),
                }
            )
            if (
                training
                and checkpoint_callback is not None
                and state.global_step % checkpoint_steps == 0
            ):
                checkpoint_callback(state)

    elapsed = time.perf_counter() - started
    if totals["rows"] == 0:
        raise RuntimeError("epoch produced zero rows")
    rows = int(totals["rows"])
    return EpochMetrics(
        loss=totals["loss"] / rows,
        explicit_loss=totals["explicit"] / rows,
        in_batch_loss=totals["in_batch"] / rows,
        mrr=totals["mrr"] / rows,
        hit1=totals["hit1"] / rows,
        hit10=totals["hit10"] / rows,
        rows=rows,
        batches=int(totals["batches"]),
        elapsed_seconds=elapsed,
    )
