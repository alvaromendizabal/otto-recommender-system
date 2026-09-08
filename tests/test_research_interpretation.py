"""Explanations are tied to native frozen models and use selection rows only."""

from __future__ import annotations

import logging

import polars as pl
from test_research_evaluation import build_evaluated_study

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.research.interpretation import explain


def test_post_selection_explanations_preserve_the_model_seal_and_prove_additivity(tmp_path):
    build_evaluated_study(tmp_path)
    before = sha256_file(tmp_path / "evaluation_seal.json")
    args = dict(
        root=tmp_path,
        seed=7,
        threads=1,
        logger=logging.getLogger("explain-test"),
        sample_sessions=20,
        shap_rows=64,
    )
    report = explain(**args)
    assert report["status"] == "passed"
    assert report["selection_sessions"] == 20
    assert report["shap_rows"] == 64
    assert max(report["shap_additivity_max_abs_error"].values()) < 1e-5
    assert report["group_permutation"]
    assert report["feature_benchmark"]["broad_catalog"]["features"] == 1482
    assert sha256_file(tmp_path / "evaluation_seal.json") == before
    frame = pl.read_parquet(tmp_path / "interpretation/feature_importance.parquet")
    assert set(frame["objective"]) == {"clicks", "carts", "orders"}
    assert explain(**args) == report
