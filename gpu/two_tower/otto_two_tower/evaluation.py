"""Exact catalogue retrieval and verified, atomic evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verified_part(path: Path, input_id: str) -> dict[str, Any] | None:
    receipt = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not receipt.is_file():
        return None
    try:
        value = read_json(receipt)
        if value.get("input_id") != input_id:
            raise ValueError(f"part belongs to different evaluation: {path}")
        if value.get("sha256") == sha256_file(path):
            return value
    except (OSError, json.JSONDecodeError):
        return None
    return None


def commit_part(temporary: Path, path: Path, input_id: str, **metadata: Any) -> None:
    from .checkpoint import write_json_atomic

    temporary.replace(path)
    write_json_atomic(
        {**metadata, "input_id": input_id, "sha256": sha256_file(path)},
        path.with_suffix(path.suffix + ".json"),
    )


def lexicographic_topk(
    scores: torch.Tensor, item_ids: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank by descending score then ascending aid, including boundary ties.

    torch.topk alone does not guarantee stable tie selection. Its threshold
    identifies the cutoff; selecting the smallest tied aids resolves it exactly.
    """
    if scores.ndim != 2 or not 0 < k <= scores.shape[1]:
        raise ValueError("k must be positive and no larger than the score width")
    ids = item_ids.expand_as(scores)
    values, positions = scores.topk(k, dim=1)
    cutoff = values[:, -1:]
    chosen_ids = ids.gather(1, positions)
    sentinel = torch.iinfo(torch.int64).max
    tied_ids = torch.where(scores == cutoff, ids, sentinel)
    smallest_ties = tied_ids.topk(k, dim=1, largest=False).values
    strict = values > cutoff
    merged_scores = torch.cat(
        (values.masked_fill(~strict, -torch.inf), cutoff.expand_as(smallest_ties)), dim=1
    )
    merged_ids = torch.cat((chosen_ids.masked_fill(~strict, sentinel), smallest_ties), dim=1)
    merged_scores = merged_scores.masked_fill(merged_ids == sentinel, -torch.inf)
    order = torch.argsort(merged_ids, dim=1, stable=True)
    by_id = merged_scores.gather(1, order)
    order = order.gather(1, torch.argsort(by_id, dim=1, descending=True, stable=True)[:, :k])
    return merged_scores.gather(1, order), merged_ids.gather(1, order)


@torch.inference_mode()
def exact_search(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    item_ids: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
    validate_candidates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("embeddings must be matrices")
    if queries.shape[1] != candidates.shape[1] or len(item_ids) != len(candidates):
        raise ValueError("embedding dimensions or catalogue IDs do not match")
    if chunk_size <= 0 or not 0 < k <= len(candidates):
        raise ValueError("invalid search size")
    if not torch.isfinite(queries).all() or (
        validate_candidates and not torch.isfinite(candidates).all()
    ):
        raise ValueError("non-finite embeddings")
    scores = torch.empty((len(queries), 0), device=queries.device)
    ids = torch.empty((len(queries), 0), dtype=torch.int64, device=queries.device)
    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        block = queries @ candidates[start:end].T
        values, aids = lexicographic_topk(block, item_ids[start:end], min(k, end - start))
        scores, ids = lexicographic_topk(
            torch.cat((scores, values), dim=1),
            torch.cat((ids, aids), dim=1),
            min(k, scores.shape[1] + values.shape[1]),
        )
    return scores, ids
