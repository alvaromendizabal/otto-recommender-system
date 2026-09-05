from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """Merged candidate with complete retriever provenance."""

    aid: int
    rrf_score: float
    source_count: int
    source_scores: dict[str, float]
    source_ranks: dict[str, int]


def merge_candidate_sources(
    sources: Mapping[
        str,
        Sequence[tuple[int, float]],
    ],
    *,
    limit: int,
    rrf_k: float = 60.0,
) -> list[Candidate]:
    """Merge heterogeneous candidate sources using reciprocal-rank fusion."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    source_scores: dict[
        int,
        dict[str, float],
    ] = {}
    source_ranks: dict[
        int,
        dict[str, int],
    ] = {}
    fused_scores: dict[int, float] = {}

    for source_name, ranked_items in sources.items():
        seen: set[int] = set()

        for rank, (aid, raw_score) in enumerate(
            ranked_items,
            start=1,
        ):
            if aid in seen:
                continue

            seen.add(aid)

            source_scores.setdefault(
                aid,
                {},
            )[source_name] = float(raw_score)

            source_ranks.setdefault(
                aid,
                {},
            )[source_name] = rank

            fused_scores[aid] = (
                fused_scores.get(aid, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    ranked_aids = sorted(
        fused_scores,
        key=lambda aid: (
            -fused_scores[aid],
            -len(source_ranks[aid]),
            aid,
        ),
    )

    return [
        Candidate(
            aid=aid,
            rrf_score=fused_scores[aid],
            source_count=len(source_ranks[aid]),
            source_scores=source_scores[aid],
            source_ranks=source_ranks[aid],
        )
        for aid in ranked_aids[:limit]
    ]
