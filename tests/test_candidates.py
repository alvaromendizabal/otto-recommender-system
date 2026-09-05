from otto_recsys.retrieval.candidates import (
    merge_candidate_sources,
)


def test_merge_preserves_provenance() -> None:
    candidates = merge_candidate_sources(
        {
            "session": [
                (10, 0.9),
                (20, 0.8),
            ],
            "popularity": [
                (20, 4.0),
                (30, 3.0),
            ],
        },
        limit=3,
    )

    by_aid = {
        candidate.aid: candidate
        for candidate in candidates
    }

    assert by_aid[20].source_count == 2
    assert by_aid[20].source_ranks == {
        "session": 2,
        "popularity": 1,
    }


def test_merge_deduplicates_within_source() -> None:
    candidates = merge_candidate_sources(
        {
            "source": [
                (10, 1.0),
                (10, 0.5),
                (20, 0.4),
            ],
        },
        limit=10,
    )

    assert [
        candidate.aid
        for candidate in candidates
    ] == [10, 20]
