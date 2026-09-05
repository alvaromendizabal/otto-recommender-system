from __future__ import annotations

from otto_two_tower.resume_contract import resume_proof_payload


def test_resume_proof_requires_nonzero_restored_step_and_advancement() -> None:
    passed = resume_proof_payload(
        input_id="input",
        code_commit="abc",
        resumed_from_step=40,
        final_step=80,
    )
    assert passed["status"] == "passed"
    assert passed["advanced_steps"] == 40

    no_restore = resume_proof_payload(
        input_id="input",
        code_commit="abc",
        resumed_from_step=0,
        final_step=40,
    )
    assert no_restore["status"] == "failed"

    no_advance = resume_proof_payload(
        input_id="input",
        code_commit="abc",
        resumed_from_step=40,
        final_step=40,
    )
    assert no_advance["status"] == "failed"
