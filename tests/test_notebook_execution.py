from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

EXECUTOR = runpy.run_path(str(Path(__file__).parents[1] / "scripts/execute_notebooks.py"))


def notebook() -> dict[str, Any]:
    return {"cells": [{"cell_type": "code", "source": ["print(1)"],
                       "execution_count": 1, "outputs": [
                           {"output_type": "stream", "name": "stdout", "text": ["1\n"]},
                       ]}]}


def test_notebook_receipt_requires_exact_content(tmp_path: Path) -> None:
    path = tmp_path / "01_analysis.ipynb"
    EXECUTOR["atomic_json"](path, notebook())
    receipt = {"input_id": "a", "sha256": EXECUTOR["sha256"](path), "code_cells": 1}
    EXECUTOR["atomic_json"](path.with_suffix(".json"), receipt)
    assert EXECUTOR["valid_receipt"](path, "a") == receipt
    assert EXECUTOR["valid_receipt"](path, "b") is None
    path.write_text("{}")
    assert EXECUTOR["valid_receipt"](path, "a") is None


@pytest.mark.parametrize("mutation", ["unexecuted", "error", "warning", "empty"])
def test_notebook_outputs_reject_incomplete_or_noisy_results(mutation: str) -> None:
    value = notebook()
    cell = value["cells"][0]
    if mutation == "unexecuted":
        cell["execution_count"] = None
    elif mutation == "error":
        cell["outputs"] = [{"output_type": "error", "ename": "ValueError"}]
    elif mutation == "warning":
        cell["outputs"][0]["text"] = ["FutureWarning: invalid API\n"]
    else:
        cell["source"] = [""]
    with pytest.raises(ValueError):
        EXECUTOR["validate_outputs"](value)
