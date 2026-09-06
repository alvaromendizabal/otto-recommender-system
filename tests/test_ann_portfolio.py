"""The public analytical notebook must be executable and honest about evidence."""

from __future__ import annotations

import ast
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


def test_ann_notebook_distinguishes_measured_evidence_and_pending_results() -> None:
    notebook = json.loads(Path("notebooks/06_ann_benchmark.ipynb").read_text())
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "official OTTO weights" in text
    assert "end-to-end serving latency" in text
    assert "Unknown catalogue positives remain misses" in text
    assert "ANN measurements are pending" in text
    assert "sha256(report_path.read_bytes())" in text
