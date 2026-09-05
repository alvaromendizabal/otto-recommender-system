from __future__ import annotations

from typing import Any

import pytest

from otto_recsys.ranking.training_cache import (
    deterministic_cut_index,
    future_labels,
    session_is_selected,
    split_training_example,
)


def _events() -> list[dict[str, Any]]:
    return [
        {"aid": 10, "ts": 100, "type": "clicks"},
        {"aid": 20, "ts": 200, "type": "carts"},
        {"aid": 30, "ts": 300, "type": "clicks"},
        {"aid": 40, "ts": 400, "type": "orders"},
        {"aid": 50, "ts": 500, "type": "carts"},
        {"aid": 40, "ts": 600, "type": "orders"},
    ]


def test_session_selection_is_deterministic_and_partitioned() -> None:
    outcomes = [
        session_is_selected(
            12345,
            seed=7,
            denominator=8,
            remainder=remainder,
        )
        for remainder in range(8)
    ]
    assert sum(outcomes) == 1
    assert outcomes == [
        session_is_selected(
            12345,
            seed=7,
            denominator=8,
            remainder=remainder,
        )
        for remainder in range(8)
    ]


def test_cut_index_always_preserves_observed_prefix_and_future() -> None:
    for session in range(100):
        cut = deterministic_cut_index(
            session,
            10,
            seed=42,
            min_prefix_events=2,
        )
        assert 2 <= cut <= 9
    with pytest.raises(ValueError):
        deterministic_cut_index(1, 2, seed=42, min_prefix_events=2)


def test_future_labels_match_otto_objectives_and_deduplicate_buys() -> None:
    labels = future_labels(_events()[2:], session=1)
    assert labels["clicks"] == 30
    assert isinstance(labels["clicks"], int)
    assert labels["carts"] == [50]
    assert labels["orders"] == [40]


def test_split_training_example_never_places_hidden_event_in_prefix() -> None:
    record = {"session": 99, "events": _events()}
    session, prefix, future, cut = split_training_example(
        record,
        seed=17,
        min_prefix_events=2,
        max_prefix_events=3,
    )
    assert session == 99
    assert 2 <= cut < len(_events())
    assert prefix == _events()[:cut][-3:]
    assert future == _events()[cut:]
    assert set(id(event) for event in prefix).isdisjoint(
        set(id(event) for event in future)
    )
