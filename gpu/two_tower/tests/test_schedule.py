from __future__ import annotations

from otto_two_tower.trainer import cosine_warmup_lambda


def test_cosine_warmup_schedule_bounds() -> None:
    values = [
        cosine_warmup_lambda(step, total_steps=100, warmup_steps=10)
        for step in range(100)
    ]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[0] < values[9]
    assert values[-1] < values[20]
