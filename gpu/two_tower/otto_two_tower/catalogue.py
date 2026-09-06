"""Validated item-ID lookup that preserves the trained embedding row order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class Catalogue:
    """Use the exported inverse map; item IDs need not be sorted or contiguous."""

    def __init__(self, item_ids: np.ndarray, aid_to_index: np.ndarray) -> None:
        for value in (item_ids, aid_to_index):
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.signedinteger):
                raise ValueError("catalogue IDs and inverse map must be signed integer vectors")
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.aid_to_index = aid_to_index
        if not len(item_ids) or np.any(item_ids < 0) or np.any(item_ids >= len(aid_to_index)):
            raise ValueError("catalogue IDs are empty or outside the inverse map")
        if not np.array_equal(aid_to_index[item_ids], np.arange(len(item_ids))):
            raise ValueError("catalogue IDs must be unique with an aligned inverse map")
        if (
            np.any(aid_to_index < -1)
            or np.any(aid_to_index >= len(item_ids))
            or np.count_nonzero(aid_to_index >= 0) != len(item_ids)
        ):
            raise ValueError("inverse map contains invalid or extra catalogue rows")

    def rows(self, aids: np.ndarray) -> np.ndarray:
        """Resolve arbitrary-shaped IDs without changing the catalogue or vectors."""
        if not np.issubdtype(aids.dtype, np.integer):
            raise ValueError("catalogue lookup requires integer IDs")
        if np.any(aids < 0) or np.any(aids >= len(self.aid_to_index)):
            raise ValueError("unknown catalogue IDs")
        positions = self.aid_to_index[aids]
        if np.any(positions < 0):
            raise ValueError("unknown catalogue IDs")
        return np.asarray(positions, dtype=np.int64)


def validate_files(root: Path) -> dict[str, object]:
    """Check the small production lookup arrays before launching paid compute."""
    manifest = json.loads((root / "manifest.json").read_text())
    arrays = {}
    for name in ("item_ids", "aid_to_index"):
        path = root / (name + ".npy")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest[name + "_sha256"]:
            raise ValueError(f"catalogue input checksum mismatch: {name}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    catalogue = Catalogue(**arrays)
    if manifest["items"] != len(catalogue.item_ids):
        raise ValueError("catalogue manifest row count mismatch")
    return {
        "status": "passed",
        "catalogue_items": len(catalogue.item_ids),
        "ids_sorted": bool(np.all(catalogue.item_ids[1:] > catalogue.item_ids[:-1])),
        "row_order": "preserved from trained vocabulary",
        "item_ids_sha256": manifest["item_ids_sha256"],
        "aid_to_index_sha256": manifest["aid_to_index_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--item-data", type=Path, required=True)
    print(json.dumps(validate_files(parser.parse_args().item_data), sort_keys=True))
