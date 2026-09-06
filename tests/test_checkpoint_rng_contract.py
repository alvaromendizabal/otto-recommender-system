from __future__ import annotations

from pathlib import Path


def test_checkpoint_rng_state_is_map_location_safe_by_contract() -> None:
    payload = Path(
        "gpu/two_tower/otto_two_tower/checkpoint.py"
    ).read_text(encoding="utf-8")

    assert "CHECKPOINT_FORMAT_VERSION = 2" in payload
    assert '"torch": _rng_state_to_bytes(torch.get_rng_state())' in payload
    assert 'device="cpu", dtype=torch.uint8' in payload
    assert 'torch.set_rng_state(_rng_state_to_cpu_byte_tensor(state["torch"]))' in payload
    assert "torch.cuda.set_rng_state_all(cuda_states)" in payload


def test_gpu_runtime_validation_exercises_checkpoint_rng_roundtrip() -> None:
    payload = Path("gpu/two_tower/runtime_validation.py").read_text(encoding="utf-8")

    assert 'map_location=device' in payload
    assert 'CHECKPOINT_RNG_ROUNDTRIP_PASSED' in payload
