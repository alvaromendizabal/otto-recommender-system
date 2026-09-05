from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .data import SequenceBatch


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        pretrained_item_vectors: torch.Tensor,
        *,
        padding_index: int,
        config: ModelConfig,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        config.validate()
        if pretrained_item_vectors.ndim != 2:
            raise ValueError("pretrained_item_vectors must be rank 2")
        if pretrained_item_vectors.shape[1] != config.embedding_dim:
            raise ValueError("pretrained item-vector dimension does not match embedding_dim")

        item_count = int(pretrained_item_vectors.shape[0])
        unknown_index = item_count
        expected_padding = item_count + 1
        if padding_index != expected_padding:
            raise ValueError("padding_index must immediately follow the unknown item index")

        extended = torch.zeros(
            (item_count + 2, config.embedding_dim),
            dtype=torch.float32,
        )
        extended[:item_count].copy_(pretrained_item_vectors.float())
        nn.init.normal_(extended[unknown_index : unknown_index + 1], std=0.02)

        self.item_embedding = nn.Embedding(
            item_count + 2,
            config.embedding_dim,
            padding_idx=padding_index,
            sparse=True,
            _weight=extended,
        )
        self.event_type_embedding = nn.Embedding(3, config.embedding_dim)
        self.time_embedding = nn.Embedding(config.time_buckets, config.embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, config.embedding_dim)
        self.objective_embedding = nn.Embedding(config.objective_count, config.embedding_dim)

        self.key_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.value_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.query_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        self.session_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.temperature), dtype=torch.float32)
        )
        self.max_seq_len = max_seq_len
        self.embedding_dim = config.embedding_dim

    def encode_session(
        self,
        sequence: SequenceBatch,
        objective_ids: torch.Tensor,
    ) -> torch.Tensor:
        item = self.item_embedding(sequence.item_indices)
        event = self.event_type_embedding(sequence.event_types)
        time = self.time_embedding(sequence.time_buckets)
        positions = torch.arange(
            self.max_seq_len,
            device=sequence.item_indices.device,
            dtype=torch.long,
        ).unsqueeze(0)
        position = self.position_embedding(positions)
        hidden = item + event + time + position

        objective = self.objective_embedding(objective_ids)
        query = self.query_projection(objective).unsqueeze(1)
        keys = self.key_projection(hidden)
        values = self.value_projection(hidden)
        attention_logits = (keys * query).sum(dim=-1) / math.sqrt(self.embedding_dim)
        attention_logits = attention_logits.masked_fill(~sequence.mask, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = torch.sum(values * attention.unsqueeze(-1), dim=1)

        denominator = sequence.mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        mean_pooled = torch.sum(hidden * sequence.mask.unsqueeze(-1), dim=1) / denominator
        last = hidden[:, -1, :]
        representation = torch.cat((pooled, mean_pooled, last, objective), dim=-1)
        return F.normalize(self.session_mlp(representation), dim=-1)

    def encode_candidates(
        self,
        item_indices: torch.Tensor,
        objective_ids: torch.Tensor,
    ) -> torch.Tensor:
        item = self.item_embedding(item_indices)
        objective = self.objective_embedding(objective_ids)
        while objective.ndim < item.ndim:
            objective = objective.unsqueeze(1)
        objective = objective.expand(*item.shape[:-1], objective.shape[-1])
        representation = torch.cat((item, objective), dim=-1)
        return F.normalize(self.candidate_mlp(representation), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)
