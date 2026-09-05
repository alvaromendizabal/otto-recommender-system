from __future__ import annotations

from pathlib import Path

import torch

from otto_two_tower.checkpoint import TrainingState, load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
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
    state = TrainingState(epoch=2, next_batch=7, global_step=99, best_valid_loss=0.4)
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


def test_checkpoint_rejects_input_mismatch(tmp_path: Path) -> None:
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
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=combined,
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_scheduler=dense_scheduler,
        sparse_scheduler=sparse_scheduler,
        state=TrainingState(global_step=10),
        input_id="expected",
        config={"x": 1},
    )

    try:
        load_checkpoint(
            path,
            model=combined,
            dense_optimizer=dense_optimizer,
            sparse_optimizer=sparse_optimizer,
            dense_scheduler=dense_scheduler,
            sparse_scheduler=sparse_scheduler,
            expected_input_id="different",
            map_location=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "input_id" in str(exc)
    else:
        raise AssertionError("checkpoint input mismatch was not rejected")
