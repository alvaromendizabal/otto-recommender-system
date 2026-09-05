from __future__ import annotations

import torch

from otto_two_tower.loss import objective_conditioned_contrastive_loss


def test_contrastive_loss_is_finite_and_masks_same_session() -> None:
    query = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    positive = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    negatives = torch.nn.functional.normalize(torch.randn(4, 5, 8), dim=-1)
    result = objective_conditioned_contrastive_loss(
        query,
        positive,
        negatives,
        session_ids=torch.tensor([10, 10, 20, 30]),
        objective_ids=torch.tensor([1, 1, 1, 2]),
        positive_aids=torch.tensor([100, 101, 100, 300]),
        scale=torch.tensor(10.0),
        in_batch_weight=0.35,
    )
    assert torch.isfinite(result.loss)
    assert 0.0 <= float(result.hit10) <= 1.0
    assert 0.0 <= float(result.mrr) <= 1.0
