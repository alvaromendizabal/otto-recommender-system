"""Public frontier review: arithmetic, provenance, rendering and method contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research.frontier import latent_features, path_features, retrieval, review

ROOT = Path(__file__).resolve().parents[1] / "research/frontier"


def test_all_public_point_scores_recompute() -> None:
    data = review.read_evidence(ROOT / "evidence.json")
    assert len(data["studies"]) == 5
    assert data["submission"]["private_score"] == 0.56842
    assert data["submission"]["winner_beaten"] is False
    assert data["latest_attempt"]["status"] == "BLOCKED_BEFORE_NEW_FITTING"
    assert data["latest_attempt"]["new_score"] is None


def test_changed_point_metric_is_rejected(tmp_path: Path) -> None:
    data = review.read_evidence(ROOT / "evidence.json")
    data["studies"][0]["metrics"]["control_shared"]["hits"][0] += 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metric mismatch"):
        review.read_evidence(path)


@pytest.mark.parametrize("hits,den", [([1, 2], [3, 4]), ([1.0, 1, 1], [2, 2, 2]),
                                     ([1, 1, 1], [0, 2, 2]), ([3, 1, 1], [2, 2, 2])])
def test_metric_rejects_invalid_totals(hits: list, den: list) -> None:
    with pytest.raises(ValueError):
        review.score(hits, den)


def test_reported_contributions_sum_to_latent_gain() -> None:
    data = review.read_evidence(ROOT / "evidence.json")
    latent = next(s for s in data["studies"] if s["id"] == "latent_affinity")
    total = sum(v for _, v in review.contribution_rows(data))
    assert abs(total - 100 * latent["comparison"]["gain"]) < 1e-10
    coverage, achieved = review.coverage_rows(data)
    assert coverage[1] > 0 > achieved[1]


def test_saved_notebook_is_executed_portable_and_bound_to_evidence() -> None:
    receipt = json.loads((ROOT / "execution.json").read_text())
    path = ROOT / "01_frontier_review.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["notebook_sha256"]
    assert hashlib.sha256((ROOT / "evidence.json").read_bytes()).hexdigest() == (
        receipt["evidence_sha256"]
    )
    nb = json.loads(path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(range(1, 8))
    outputs = [o for c in code for o in c["outputs"]]
    assert not any(o["output_type"] == "error" for o in outputs)
    charts = [o["data"] for o in outputs if "application/vnd.plotly.v1+json" in o.get("data", {})]
    assert len(charts) == 3
    for chart in charts:
        assert "image/svg+xml" in chart
        assert chart["application/vnd.plotly.v1+json"]["layout"]["width"] == 900
    assert "FRONTIER_REVIEW_COMPLETE" in str(code[-1]["outputs"])


def test_method_snapshots_match_their_published_receipts() -> None:
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for item in manifest["kernel_sources"]:
        assert item["function_ast_identical"] is True
        assert hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest() == (
            item["public_sha256"]
        )


def test_two_hop_propagation_and_path_feature_alignment() -> None:
    graph = csr_matrix(([2.0, 1.0], ([0, 1], [1, 2])), shape=(5, 5))
    ids, mass = retrieval.weighted_step(graph, np.array([0]), np.array([1.0]))
    ids, mass = retrieval.weighted_step(graph, ids, mass)
    np.testing.assert_array_equal(ids, [2])
    np.testing.assert_allclose(mass, [1.0])
    prefix = SimpleNamespace(aid=np.array([0]), kind=np.array([0]))
    out, info = retrieval.expand([graph] * 3, prefix, np.array([0, 3, 4]), replacements=1)
    np.testing.assert_array_equal(out, [0, 3, 2])
    assert info["inserted"] == 1
    x = path_features.transform([graph] * 3, prefix, out, np.array([0, 3, 4]))
    assert x.shape == (3, 18) and np.isfinite(x).all()
    assert x[2, 0] > 0 and x[0, 0] == 0


def test_graph_embedding_checkpoint_reuses_without_refactorization(tmp_path: Path) -> None:
    graph = csr_matrix(np.eye(8) + np.roll(np.eye(8), 1, axis=1))
    e, n, first = latent_features.embed(graph, tmp_path, {"fixture": 1}, rank=2, oversampling=1)
    ee, nn, second = latent_features.embed(graph, tmp_path, {"fixture": 1}, rank=2, oversampling=1)
    np.testing.assert_array_equal(e, ee)
    np.testing.assert_array_equal(n, nn)
    assert first == second
    assert e.shape == (8, 2) and np.isfinite(e).all()


def test_replay_signature_rejects_the_old_call() -> None:
    def replay(d, engine, matrix, external, out):
        return d, engine, matrix, external, out

    adapter = create_autospec(replay, side_effect=replay)
    with pytest.raises(TypeError):
        adapter({}, object(), object())
    resources = object(), object(), object()
    result = adapter({}, *resources, Path("output"))
    assert result[1:4] == resources
    patch = (ROOT / "training_scale_replay.patch").read_text()
    assert "+    replay(d,*resources_cache,out)" in patch


def test_evidence_has_no_session_level_identifiers_or_credentials() -> None:
    forbidden = {"session_ids", "cohort_ids", "labels", "access_token", "secret_access_key"}

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(json.loads((ROOT / "evidence.json").read_text()))
