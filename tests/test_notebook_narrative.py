"""Historical notebook captions must point to completed, scoped follow-up evidence."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def test_retrieval_captions_do_not_mark_completed_followups_pending() -> None:
    texts = {}
    for name in ("02_retrieval_benchmarks.ipynb", "05_two_tower_results.ipynb"):
        notebook = json.loads((Path("notebooks") / name).read_text())
        texts[name] = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    baseline = texts["02_retrieval_benchmarks.ipynb"]
    neural = texts["05_two_tower_results.ipynb"]
    assert "neural Fold 0 results pending" not in baseline
    assert "subsequent results: notebooks 05" in baseline
    assert "Final ranked Recall@20 remains unmeasured." not in neural
    assert "ANN serving performance remain open measurements" not in neural
    assert "**Next: ANN fidelity and latency.**" not in neural
    assert "08_ranking_evaluation.ipynb" in neural
    assert "06_ann_benchmark.ipynb" in neural
    assert "Independently selected neural-source ranking remains unmeasured" in neural
    assert "training-only screening" in neural
