"""Independent adaptation of a leading OTTO supervised sequence-retrieval mechanism.

This public kernel contains the model and loss mathematics used by the private,
checkpointed AWS runner. It does not include competition data, weights, optimizer
checkpoints, AWS paths, or upstream source code.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SequenceEncoder(nn.Module):
    """Task-conditioned sequence encoder for click/cart/order retrieval."""

    def __init__(self, item_count: int, dim: int = 500, time_dim: int = 50, length: int = 10):
        super().__init__()
        self.pad = item_count - 1
        self.length = length
        self.item = nn.Embedding(item_count, dim, padding_idx=self.pad)
        self.hour = nn.Embedding(169, time_dim, padding_idx=168)
        self.task = nn.Embedding(7, dim, padding_idx=0)
        width = length * (dim + time_dim + 3)
        self.network = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.BatchNorm1d(width),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.BatchNorm1d(width // 2),
            nn.Linear(width // 2, dim),
        )
        nn.init.uniform_(self.item.weight, -1.0, 1.0)
        nn.init.uniform_(self.hour.weight, -1.0, 1.0)
        with torch.no_grad():
            self.item.weight[self.pad].zero_()
            self.hour.weight[168].zero_()

    def base(self, items: torch.Tensor, hours: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not (items.shape == hours.shape == actions.shape and items.shape[1] == self.length):
            raise ValueError("sequence inputs must share [batch, history_length]")
        mask = items.ne(self.pad).unsqueeze(-1)
        cyclic_hour = hours.remainder(168)
        hour_context = (
            self.hour(cyclic_hour)
            + self.hour((cyclic_hour + 1) % 168)
            + self.hour((cyclic_hour + 2) % 168)
        )
        angle = hours.float().remainder(24) * (2 * torch.pi / 24)
        event_context = torch.stack((angle.sin(), angle.cos(), actions.float()), dim=-1)
        encoded = torch.cat((self.item(items), event_context, hour_context), dim=-1) * mask
        return self.network(encoded.flatten(1))

    def query(self, items: torch.Tensor, hours: torch.Tensor, actions: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.base(items, hours, actions).float() + self.task(task).float(), dim=-1, eps=1e-8)


def contrastive_loss(model: SequenceEncoder, items: torch.Tensor, hours: torch.Tensor, actions: torch.Tensor,
                     targets: torch.Tensor, target_types: torch.Tensor, negatives: torch.Tensor, *,
                     temperature: float = 0.05, hard_negatives: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted-mean + hardest-positive contrastive objective with collision masking."""
    if temperature <= 0 or hard_negatives <= 0 or negatives.numel() == 0:
        raise ValueError("invalid contrastive configuration")
    valid = targets.ne(model.pad) & target_types.gt(0)
    if not bool(valid.any(1).all()) or not bool(valid[:, 0].all()):
        raise ValueError("every query must contain a valid positive target")
    base = model.base(items, hours, actions).float()
    query = F.normalize(base[:, None, :] + model.task(target_types).float(), dim=-1, eps=1e-8)
    positives = F.normalize(model.item(targets).float(), dim=-1, eps=1e-8)
    positive_similarity = (query * positives).sum(-1)
    weighted_positive = (positive_similarity * target_types * valid).sum(1) / (target_types * valid).sum(1)
    hardest_positive = positive_similarity.masked_fill(~valid, float("inf")).min(1).values
    positive = 0.5 * (weighted_positive + hardest_positive)
    negative_vectors = F.normalize(model.item(negatives).float(), dim=-1, eps=1e-8)
    negative_similarity = query[:, 0] @ negative_vectors.T
    for col in range(targets.shape[1]):
        collision = targets[:, col, None].eq(negatives[None, :]) & valid[:, col, None]
        negative_similarity = negative_similarity.masked_fill(collision, float("-inf"))
    negative_similarity = negative_similarity.float().topk(min(hard_negatives, len(negatives)), dim=1).values
    if not bool(torch.isfinite(negative_similarity).any(1).all()):
        raise ValueError("no eligible negative remains for at least one query")
    logits = torch.cat((positive[:, None], negative_similarity), dim=1) / temperature
    loss = (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()
    accuracy = (logits.argmax(1) == 0).float().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite contrastive loss")
    return loss, accuracy
