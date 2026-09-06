from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch

from otto_two_tower.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from otto_two_tower.config import ModelConfig
from otto_two_tower.data import SequenceBatch
from otto_two_tower.loss import objective_conditioned_contrastive_loss
from otto_two_tower.model import TwoTowerModel


def _nvidia_smi() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def main() -> int:
    started = time.perf_counter()
    print(f"GPU_RUNTIME_VALIDATION_START unix_seconds={time.time():.3f}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the SageMaker GPU container")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required for the two-tower GPU contract")

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "nvidia_smi": _nvidia_smi(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    vectors = torch.randn(128, 16)
    config = ModelConfig(embedding_dim=16, hidden_dim=32, time_buckets=8)
    model = TwoTowerModel(vectors, padding_index=129, config=config, max_seq_len=4).to(device)
    sequence = SequenceBatch(
        item_indices=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device),
        event_types=torch.tensor([[0, 1, 0, 2], [0, 0, 1, 2]], device=device),
        time_buckets=torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]], device=device),
        mask=torch.ones((2, 4), dtype=torch.bool, device=device),
    )
    objectives = torch.tensor([0, 2], device=device)
    positives = torch.tensor([9, 10], device=device)
    negatives = torch.tensor([[11, 12, 13], [14, 15, 16]], device=device)
    positive_aids = torch.tensor([900, 1000], device=device)
    session_ids = torch.tensor([100, 200], device=device)

    dense_parameters = [
        parameter for name, parameter in model.named_parameters() if name != "item_embedding.weight"
    ]
    dense_optimizer = torch.optim.AdamW(dense_parameters, lr=1e-3)
    sparse_optimizer = torch.optim.SparseAdam([model.item_embedding.weight], lr=1e-3)
    dense_scheduler = torch.optim.lr_scheduler.LambdaLR(dense_optimizer, lambda _: 1.0)
    sparse_scheduler = torch.optim.lr_scheduler.LambdaLR(sparse_optimizer, lambda _: 1.0)

    dense_optimizer.zero_grad(set_to_none=True)
    sparse_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        query = model.encode_session(sequence, objectives)
        positive = model.encode_candidates(positives, objectives)
        negative = model.encode_candidates(negatives, objectives)
        result = objective_conditioned_contrastive_loss(
            query,
            positive,
            negative,
            session_ids=session_ids,
            objective_ids=objectives,
            positive_aids=positive_aids,
            scale=model.scale(),
            in_batch_weight=0.35,
        )
    if not torch.isfinite(result.loss):
        raise RuntimeError("GPU runtime validation produced a non-finite loss")
    result.loss.backward()
    dense_optimizer.step()
    sparse_optimizer.step()
    dense_scheduler.step()
    sparse_scheduler.step()

    checkpoint = Path("/tmp/otto-two-tower-runtime-checkpoint.pt")
    state = TrainingState(epoch=0, next_batch=2, global_step=1, best_valid_loss=1.0)
    save_checkpoint(
        checkpoint,
        model=model,
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_scheduler=dense_scheduler,
        sparse_scheduler=sparse_scheduler,
        state=state,
        input_id="runtime-validation",
        config={"runtime_validation": True},
    )
    loaded = load_checkpoint(
        checkpoint,
        model=model,
        dense_optimizer=dense_optimizer,
        sparse_optimizer=sparse_optimizer,
        dense_scheduler=dense_scheduler,
        sparse_scheduler=sparse_scheduler,
        expected_input_id="runtime-validation",
        map_location=device,
    )
    if loaded.global_step != 1 or loaded.next_batch != 2:
        raise RuntimeError("checkpoint round-trip did not preserve progress")
    print(
        "CHECKPOINT_RNG_ROUNDTRIP_PASSED "
        f"checkpoint_format_version={CHECKPOINT_FORMAT_VERSION} map_location={device.type}",
        flush=True,
    )

    elapsed = time.perf_counter() - started
    print(f"OTTO_GPU_RUNTIME_VALIDATION_PASSED total_seconds={elapsed:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
