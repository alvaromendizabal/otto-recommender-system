from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
    explicit_loss: torch.Tensor
    in_batch_loss: torch.Tensor
    mrr: torch.Tensor
    hit1: torch.Tensor
    hit10: torch.Tensor


def _explicit_metrics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = logits[:, :1]
    rank = 1 + (logits[:, 1:] > positive).sum(dim=1)
    reciprocal = rank.to(torch.float32).reciprocal()
    return reciprocal.mean(), (rank <= 1).float().mean(), (rank <= 10).float().mean()


def _in_batch_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    *,
    session_ids: torch.Tensor,
    objective_ids: torch.Tensor,
    positive_aids: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    weighted_losses: list[torch.Tensor] = []
    weighted_sizes: list[int] = []
    for objective in torch.unique(objective_ids):
        selector = objective_ids == objective
        indices = torch.nonzero(selector, as_tuple=False).flatten()
        if indices.numel() <= 1:
            continue
        q = query.index_select(0, indices)
        p = positive.index_select(0, indices)
        sessions = session_ids.index_select(0, indices)
        aids = positive_aids.index_select(0, indices)
        logits = q @ p.transpose(0, 1) * scale
        size = int(indices.numel())
        eye = torch.eye(size, device=logits.device, dtype=torch.bool)
        same_session = sessions[:, None] == sessions[None, :]
        same_aid = aids[:, None] == aids[None, :]
        invalid_negative = (same_session | same_aid) & ~eye
        logits = logits.masked_fill(invalid_negative, -torch.inf)
        targets = torch.arange(size, device=logits.device)
        weighted_losses.append(F.cross_entropy(logits, targets) * size)
        weighted_sizes.append(size)
    if not weighted_losses:
        return query.new_zeros(())
    return torch.stack(weighted_losses).sum() / sum(weighted_sizes)


def objective_conditioned_contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    session_ids: torch.Tensor,
    objective_ids: torch.Tensor,
    positive_aids: torch.Tensor,
    scale: torch.Tensor,
    in_batch_weight: float,
) -> LossResult:
    positive_logits = torch.sum(query * positive, dim=-1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", query, negatives)
    explicit_logits = torch.cat((positive_logits, negative_logits), dim=1) * scale
    targets = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
    explicit_loss = F.cross_entropy(explicit_logits, targets)
    in_batch = _in_batch_loss(
        query,
        positive,
        session_ids=session_ids,
        objective_ids=objective_ids,
        positive_aids=positive_aids,
        scale=scale,
    )
    total = explicit_loss + in_batch_weight * in_batch
    mrr, hit1, hit10 = _explicit_metrics(explicit_logits)
    return LossResult(
        loss=total,
        explicit_loss=explicit_loss,
        in_batch_loss=in_batch,
        mrr=mrr,
        hit1=hit1,
        hit10=hit10,
    )
