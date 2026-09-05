from __future__ import annotations

import torch

from otto_two_tower.config import ModelConfig
from otto_two_tower.data import SequenceBatch
from otto_two_tower.model import TwoTowerModel


def test_two_tower_shapes_and_normalization() -> None:
    vectors = torch.randn(32, 8)
    config = ModelConfig(embedding_dim=8, hidden_dim=16, time_buckets=8)
    model = TwoTowerModel(vectors, padding_index=33, config=config, max_seq_len=4)
    sequence = SequenceBatch(
        item_indices=torch.tensor([[1, 2, 3, 4], [33, 5, 6, 7]]),
        event_types=torch.tensor([[0, 1, 0, 2], [0, 0, 1, 2]]),
        time_buckets=torch.tensor([[3, 2, 1, 0], [0, 3, 1, 0]]),
        mask=torch.tensor([[True, True, True, True], [False, True, True, True]]),
    )
    objectives = torch.tensor([0, 2])
    query = model.encode_session(sequence, objectives)
    candidates = model.encode_candidates(torch.tensor([[1, 2], [3, 4]]), objectives)
    assert query.shape == (2, 8)
    assert candidates.shape == (2, 2, 8)
    assert torch.allclose(query.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(candidates.norm(dim=-1), torch.ones(2, 2), atol=1e-5)
