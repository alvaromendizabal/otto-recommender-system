from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from otto_two_tower.ann_search import search
from otto_two_tower.catalogue import Catalogue, validate_files
from otto_two_tower.evaluation import sha256_file


def catalogue() -> Catalogue:
    ids = np.array([9, 2, 15, 5], dtype=np.int32)
    inverse = np.full(16, -1, dtype=np.int32)
    inverse[ids] = np.arange(4)
    return Catalogue(ids, inverse)


def test_unsorted_sparse_ids_roundtrip_and_rankings_preserve_vector_alignment() -> None:
    lookup = catalogue()
    assert np.array_equal(lookup.rows(np.array([[15, 9], [2, 5]])), [[2, 0], [1, 3]])
    vectors = np.array([[0.5, 1], [1, 0], [-1, 0], [1, 0]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    exact = faiss.IndexFlatIP(2)
    exact.add(vectors)
    labelled = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
    labelled.add_with_ids(vectors, lookup.item_ids)
    expected_ids = np.array([[2, 5, 9, 15], [9, 2, 5, 15]])
    for index, positional in ((exact, True), (labelled, False)):
        scores, found = search(index, queries, vectors, lookup, 4, positional=positional)
        assert np.array_equal(found, expected_ids)
        expected_scores = np.array([[1, 1, 0.5, -1], [1, 0, 0, 0]])
        np.testing.assert_array_equal(scores, expected_scores)
    assert lookup.item_ids.tolist() == [9, 2, 15, 5]


@pytest.mark.parametrize("aids", [[-1], [0], [16], [2, 0], [1.5]])
def test_unknown_ids_are_rejected(aids: list[float]) -> None:
    with pytest.raises(ValueError):
        catalogue().rows(np.array(aids))


@pytest.mark.parametrize(
    "found,positional",
    [([9, -1], False), ([9, 9], False), ([9, 0], False), ([0, 4], True)],
)
def test_invalid_search_neighbors_cannot_produce_predictions(
    found: list[int], positional: bool
) -> None:
    class Index:
        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((1, 2)), np.array([found])

    with pytest.raises(ValueError):
        search(Index(), np.ones((1, 2)), np.ones((4, 2)), catalogue(), 2, positional=positional)


@pytest.mark.parametrize("damage", ["duplicate", "misaligned", "extra", "negative", "float"])
def test_invalid_catalogue_is_rejected(damage: str) -> None:
    valid = catalogue()
    ids, inverse = valid.item_ids.copy(), valid.aid_to_index.copy()
    if damage == "duplicate":
        ids[0] = ids[1]
    elif damage == "misaligned":
        inverse[9] = 2
    elif damage == "extra":
        inverse[0] = 0
    elif damage == "negative":
        inverse[0] = -2
    else:
        ids = ids.astype(float)
    with pytest.raises(ValueError):
        Catalogue(ids, inverse)


def test_preflight_verifies_saved_hashes_and_row_count(tmp_path: Path) -> None:
    valid = catalogue()
    manifest = {"items": 4}
    for name, values in (("item_ids", valid.item_ids), ("aid_to_index", valid.aid_to_index)):
        path = tmp_path / (name + ".npy")
        np.save(path, values)
        manifest[name + "_sha256"] = sha256_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = validate_files(tmp_path)
    assert report["status"] == "passed" and report["ids_sorted"] is False
    (tmp_path / "item_ids.npy").write_bytes(b"interrupted transfer")
    with pytest.raises(ValueError, match="checksum"):
        validate_files(tmp_path)
