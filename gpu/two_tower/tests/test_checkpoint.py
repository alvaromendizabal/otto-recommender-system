from __future__ import annotations

import json
from pathlib import Path

import torch

from otto_two_tower.checkpoint import TrainingState, load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_persists_progress_and_history(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    sparse = torch.nn.Embedding(10, 4, sparse=True)
    dense_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sparse_optimizer = torch.optim.SparseAdam(sparse.parameters(), lr=1e-3)
    dense_scheduler = torch.optim.lr_scheduler.LambdaLR(dense_optimizer, lambda _: 1.0)
    sparse_scheduler = torch.optim.lr_scheduler.LambdaLR(sparse_optimizer, lambda _: 1.0)

    class Combined(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.sparse = sparse

    combined = Combined()
    state = TrainingState(
        epoch=2,
        next_batch=7,
        global_step=99,
        best_valid_loss=0.4,
        history=[{"epoch": 0, "valid": {"loss": 0.4}}],
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=combined,
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_scheduler=dense_scheduler,
        sparse_scheduler=sparse_scheduler,
        state=state,
        input_id="abc",
        config={"x": 1},
    )
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["global_step"] == 99
    assert progress["next_batch"] == 7

    loaded = load_checkpoint(
        path,
        model=combined,
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_scheduler=dense_scheduler,
        sparse_scheduler=sparse_scheduler,
        expected_input_id="abc",
        map_location=torch.device("cpu"),
    )
    assert loaded.epoch == 2
    assert loaded.next_batch == 7
    assert loaded.global_step == 99
    assert loaded.history == [{"epoch": 0, "valid": {"loss": 0.4}}]
