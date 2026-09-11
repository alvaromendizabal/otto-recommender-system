"""Guards for the bounded early shared-feature smoke runner."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_shared_feature_smoke.py"
SPEC = importlib.util.spec_from_file_location("shared_feature_smoke", SOURCE)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeHelpersTests(unittest.TestCase):
    def test_matrix_digest_is_deterministic_and_content_bound(self) -> None:
        aids = np.array([2, 3], dtype=np.int64)
        target = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        first = smoke.matrix_digest(aids, target, matrix)
        self.assertEqual(first, smoke.matrix_digest(aids.copy(), target.copy(), matrix.copy()))
        changed = matrix.copy()
        changed[0, 0] += 1
        self.assertNotEqual(first, smoke.matrix_digest(aids, target, changed))

    def test_role_indices_require_exact_membership(self) -> None:
        queries = SimpleNamespace(session=np.array([1, 3, 5, 7], dtype=np.int64))
        np.testing.assert_array_equal(smoke.role_indices(queries, np.array([1, 5])), [0, 2])
        with self.assertRaisesRegex(ValueError, "differ"):
            smoke.role_indices(queries, np.array([1, 4]))

    def test_baseline_fusion_shape_and_zero_evidence(self) -> None:
        ranks = np.zeros((3, 5), dtype=np.float32)
        actual = smoke.baseline_fusion(ranks)
        self.assertEqual(actual.shape, (3, 3))
        self.assertTrue(np.array_equal(actual, np.zeros((3, 3), dtype=np.float32)))

    def test_baseline_fusion_positive_evidence_is_finite(self) -> None:
        ranks = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
        actual = smoke.baseline_fusion(ranks)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual > 0).all())

    def test_verify_cached_query_accepts_exact_sample(self) -> None:
        aids = np.array([10, 20, 30], dtype=np.int64)
        target = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
        chosen = np.array([0, 2], dtype=np.int64)
        stored = pl.DataFrame(
            {
                "session": [9, 9],
                "aid": [10, 30],
                "candidate_position": [0, 2],
                "target_clicks": [1, 0],
                "target_carts": [0, 0],
                "target_orders": [0, 1],
            }
        )
        smoke.verify_cached_query(
            stored, session=9, aids=aids, target=target, chosen=chosen
        )

    def test_verify_cached_query_rejects_changed_candidate(self) -> None:
        aids = np.array([10, 20], dtype=np.int64)
        target = np.zeros((2, 3), dtype=np.int8)
        stored = pl.DataFrame(
            {
                "session": [9, 9],
                "aid": [10, 99],
                "candidate_position": [0, 1],
                "target_clicks": [0, 0],
                "target_carts": [0, 0],
                "target_orders": [0, 0],
            }
        )
        with self.assertRaisesRegex(ValueError, "parity"):
            smoke.verify_cached_query(
                stored,
                session=9,
                aids=aids,
                target=target,
                chosen=np.array([0, 1], dtype=np.int64),
            )

    def test_diagnostics_marks_families_and_constant_added_feature(self) -> None:
        matrix = np.array(
            [[0.0, 1.0, 5.0], [1.0, 2.0, 5.0], [2.0, 3.0, 5.0]],
            dtype=np.float32,
        )
        frame = smoke.diagnostics(("b1", "b2", "a1"), matrix, 2)
        self.assertEqual(
            frame["family"].to_list(), ["baseline", "baseline", "shared_added"]
        )
        self.assertEqual(frame["std"][2], 0.0)
        self.assertEqual(frame["max_abs_corr_with_frozen_baseline"][2], 0.0)

    def test_selected_sessions_requires_exact_256_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "early_fit_queries.parquet"
            pl.DataFrame({"session": [1, 1, 2]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "256"):
                smoke.selected_sessions(Path(root), "fit")


if __name__ == "__main__":
    unittest.main()
