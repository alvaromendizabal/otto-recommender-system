from otto_recsys.retrieval.baseline import (
    PopularityIndex,
    recommend,
    session_candidates,
)


def test_session_candidates_are_unique() -> None:
    events = [
        {"aid": 1, "ts": 1, "type": "clicks"},
        {"aid": 2, "ts": 2, "type": "carts"},
        {"aid": 1, "ts": 3, "type": "clicks"},
    ]

    ranked = session_candidates(
        events,
        "clicks",
        limit=20,
    )

    aids = [aid for aid, _ in ranked]

    assert len(aids) == len(set(aids))


def test_target_conditioning_changes_scores() -> None:
    events = [
        {"aid": 1, "ts": 1, "type": "clicks"},
        {"aid": 2, "ts": 2, "type": "carts"},
    ]

    clicks = session_candidates(
        events,
        "clicks",
        limit=20,
    )
    carts = session_candidates(
        events,
        "carts",
        limit=20,
    )

    assert clicks != carts


def test_recommend_fills_without_duplicates() -> None:
    events = [
        {"aid": 1, "ts": 1, "type": "clicks"},
        {"aid": 2, "ts": 2, "type": "carts"},
    ]

    popularity = PopularityIndex(
        clicks=(2, 3, 4, 5),
        carts=(2, 3, 4, 5),
        orders=(2, 3, 4, 5),
    )

    predictions = recommend(
        events,
        "clicks",
        popularity,
        k=4,
    )

    assert len(predictions) == 4
    assert len(predictions) == len(set(predictions))
