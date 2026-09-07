"""The public analytical notebook must be executable and honest about evidence."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path


def test_ann_notebook_has_executed_code_and_no_error_outputs() -> None:
    notebook = json.loads(Path("notebooks/06_ann_benchmark.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) >= 6
    for cell in cells:
        ast.parse("".join(cell["source"]))
        assert cell["execution_count"] is not None
        assert all(output["output_type"] != "error" for output in cell["outputs"])
    assert "notebook_complete" in str(cells[-1]["outputs"])


def test_ann_notebook_discloses_scope_and_matches_published_evidence() -> None:
    notebook = json.loads(Path("notebooks/06_ann_benchmark.ipynb").read_text())
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "official OTTO weights" in text
    assert "end-to-end serving latency" in text
    assert "Unknown catalogue positives remain misses" in text
    assert "exploratory validation" in text
    assert "baseline comparison is complete" in text
    assert "final learned-ranker quality remains unmeasured" in text
    assert "sha256(report_path.read_bytes())" in text
    report_path = Path("reports/metrics/two_tower_fold0_ann.json")
    report = json.loads(report_path.read_text())
    receipt = json.loads(Path(str(report_path) + ".json").read_text())
    audit = json.loads(Path("reports/metrics/two_tower_fold0_ann_audit.json").read_text())
    assert report["status"] == audit["status"] == "passed"
    assert report["input_id"] == receipt["input_id"] == audit["input_id"]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == receipt["sha256"]
    assert receipt["sha256"] == audit["metrics_sha256"]
    assert audit["verified_count_parts"] == 2 * report["prediction_export"]["parts"]
    images = [
        base64.b64decode(output["data"]["image/png"])
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    for name in ("two_tower_ann.png", "two_tower_ann_quality.png", "two_tower_ann_comparison.png"):
        assert (Path("reports/figures") / name).read_bytes() in images


def test_published_ann_comparison_is_audited_and_uses_the_frozen_base() -> None:
    path = Path("reports/metrics/two_tower_fold0_ann_comparison.json")
    report = json.loads(path.read_text())
    audit = json.loads(
        Path("reports/metrics/two_tower_fold0_ann_comparison_audit.json").read_text()
    )
    exact = json.loads(Path("reports/metrics/two_tower_fold0_retrieval.json").read_text())
    ann = json.loads(Path("reports/metrics/two_tower_fold0_ann.json").read_text())
    assert report["status"] == audit["status"] == "passed"
    assert report["input_id"] == audit["input_id"]
    assert report["prediction_input_id"] == ann["input_id"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["metrics_sha256"]
    assert report["contract"]["baseline_checksums"] == exact["contract"]["baseline_checksums"]
    assert report["sessions"] == exact["sessions"] == audit["sessions"]
    assert audit["verified_parts"] == 32
    assert "not ranked Recall@20" in report["interpretation"]
