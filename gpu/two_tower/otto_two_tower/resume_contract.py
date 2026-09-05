from __future__ import annotations

from typing import Any


def resume_proof_payload(
    *,
    input_id: str,
    code_commit: str,
    resumed_from_step: int,
    final_step: int,
) -> dict[str, Any]:
    passed = resumed_from_step > 0 and final_step > resumed_from_step
    return {
        "status": "passed" if passed else "failed",
        "input_id": input_id,
        "code_commit": code_commit,
        "resumed_from_step": resumed_from_step,
        "final_step": final_step,
        "advanced_steps": final_step - resumed_from_step,
    }
