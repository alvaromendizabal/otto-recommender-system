"""Session-disjoint inner selection within the existing exploratory outer folds."""

from __future__ import annotations

import hashlib
from typing import Literal

Role = Literal["fit", "inner", "outer"]


def inner_partition(session: int, *, seed: int, partitions: int = 5) -> int:
    """Independent stable assignment; does not inspect labels or input row order."""
    if session < 0 or not 2 <= partitions <= 255:
        raise ValueError("invalid session or inner partition count")
    digest = hashlib.blake2b(
        f"otto-ranking-inner:{seed}:{session}".encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % partitions


def split_role(fold: int, inner: int, *, outer_fold: int, folds: int = 5) -> Role:
    """Resolve one fit's role. Outer rows can never select its checkpoint."""
    if not 2 <= folds <= 255 or not 0 <= fold < folds or not 0 <= outer_fold < folds:
        raise ValueError("invalid outer fold")
    if not 0 <= inner < 5:
        raise ValueError("invalid inner partition")
    if fold == outer_fold:
        return "outer"
    return "inner" if inner == 0 else "fit"
