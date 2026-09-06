from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from otto_recsys.cloud import source_preflight
from otto_recsys.experiments.manifest import sha256_file


@pytest.mark.parametrize("damage", [None, "checksum", "inverse", "manifest"])
def test_packaged_preflight_uses_actual_inputs_without_cloud_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    ids = np.array([7, 2, 4], dtype=np.int32)
    inverse = np.full(8, -1, dtype=np.int32)
    inverse[ids] = np.arange(3)
    if damage == "inverse":
        inverse[7] = 1
    manifest = {"items": 3}
    for name, values in (("item_ids", ids), ("aid_to_index", inverse)):
        np.save(inputs / (name + ".npy"), values)
        manifest[name + "_sha256"] = sha256_file(inputs / (name + ".npy"))
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    if damage == "checksum":
        (inputs / "item_ids.npy").write_bytes(b"bad transfer")
    expected = {**manifest, "items": 4} if damage == "manifest" else manifest
    actual_run = source_preflight.run_command
    downloads = []

    def run(command: list[str], **kwargs: object):
        if command[0] == "aws":
            assert command[:3] == ["aws", "s3", "cp"]
            assert command[3].startswith("s3://bucket/items/")
            downloads.append(command[3])
            shutil.copyfile(inputs / command[3].rsplit("/", 1)[-1], command[4])
            return None
        return actual_run(command, **kwargs)

    monkeypatch.setattr(source_preflight, "run_command", run)
    definition = {
        "Steps": [
            {
                "Arguments": {
                    "InputDataConfig": [
                        {
                            "ChannelName": "items",
                            "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/items/"}},
                        }
                    ]
                }
            }
        ]
    }

    def validate() -> dict:
        return source_preflight.validate_ann_catalogue(
            Path("gpu/two_tower"), definition, expected, tmp_path / "downloaded"
        )

    if damage:
        with pytest.raises((ValueError, RuntimeError)):
            validate()
    else:
        report = validate()
        assert report["catalogue_items"] == 3 and report["ids_sorted"] is False
        assert report["elapsed_seconds"] > 0 and report["verified_at_utc"].endswith("+00:00")
    assert len(downloads) == 3
