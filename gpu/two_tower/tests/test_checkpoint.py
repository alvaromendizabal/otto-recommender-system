from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def test_rng_state_roundtrip_is_map_location_safe() -> None:
    from otto_two_tower.checkpoint import capture_rng_state, restore_rng_state

    torch.manual_seed(12345)
    captured = capture_rng_state()
    assert isinstance(captured["torch"], bytes)

    expected = torch.rand(8)
    torch.manual_seed(99999)
    restore_rng_state(captured)
    actual = torch.rand(8)

    assert torch.equal(actual, expected)


def test_restore_rng_state_normalizes_legacy_tensor_payload() -> None:
    from otto_two_tower.checkpoint import restore_rng_state

    torch.manual_seed(42)
    numpy_state = np.random.get_state()
    legacy_state = torch.get_rng_state().to(dtype=torch.int64)
    expected = torch.rand(8)

    torch.manual_seed(7)
    restore_rng_state({"numpy": numpy_state, "torch": legacy_state})
    actual = torch.rand(8)

    assert torch.equal(actual, expected)


def test_checkpoint_payload_uses_versioned_rng_bytes(tmp_path: Path) -> None:
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

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model=Combined(),
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_scheduler=dense_scheduler,
        sparse_scheduler=sparse_scheduler,
        state=TrainingState(global_step=5),
        input_id="rng-format-test",
        config={"test": True},
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_format_version"] == 2
    assert isinstance(payload["rng"]["torch"], bytes)
