"""Tests for exact streaming feature diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from otto_recsys.research.feature_diagnostics import (
    DiagnosticThresholds,
    StreamingFeatureDiagnostics,
)


class StreamingFeatureDiagnosticsTests(unittest.TestCase):
    def test_exact_correlation_support_and_constant_flags(self) -> None:
        names = ("b1", "b2", "a_same", "a_inverse", "a_constant")
        diag = StreamingFeatureDiagnostics(names, baseline_count=2)
        diag.update(
            np.array(
                [
                    [1.0, 1.0, 1.0, 4.0, 5.0],
                    [2.0, 0.0, 2.0, 3.0, 5.0],
                ],
                dtype=np.float32,
            )
        )
        diag.update(
            np.array(
                [
                    [3.0, 1.0, 3.0, 2.0, 5.0],
                    [4.0, 0.0, 4.0, 1.0, 5.0],
                ],
                dtype=np.float32,
            )
        )
        result = diag.finalize()
        rows = {row["feature"]: row for row in result["features"]}
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["sessions"], 2)
        self.assertAlmostEqual(rows["a_same"]["max_abs_corr_with_baseline"], 1.0)
        self.assertAlmostEqual(rows["a_inverse"]["max_abs_corr_with_baseline"], 1.0)
        self.assertTrue(rows["a_constant"]["constant"])
        self.assertEqual(rows["b2"]["row_nonzero_share"], 0.5)
        self.assertEqual(rows["b2"]["session_nonzero_share"], 1.0)

    def test_added_redundancy_is_exact_and_diagonal_is_excluded(self) -> None:
        diag = StreamingFeatureDiagnostics(("b", "a1", "a2"), baseline_count=1)
        diag.update(
            np.array(
                [[0.0, 1.0, 2.0], [1.0, 2.0, 4.0], [0.0, 3.0, 6.0]],
                dtype=np.float32,
            )
        )
        rows = {row["feature"]: row for row in diag.finalize()["features"]}
        self.assertAlmostEqual(rows["a1"]["max_abs_corr_with_added"], 1.0)
        self.assertAlmostEqual(rows["a2"]["max_abs_corr_with_added"], 1.0)

    def test_nonfinite_wrong_shape_and_empty_finalize_rejected(self) -> None:
        diag = StreamingFeatureDiagnostics(("b", "a"), baseline_count=1)
        with self.assertRaisesRegex(ValueError, "shape"):
            diag.update(np.ones((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            diag.update(np.array([[1.0, np.nan]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "at least one"):
            diag.finalize()

    def test_threshold_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            StreamingFeatureDiagnostics(
                ("b", "a"),
                baseline_count=1,
                thresholds=DiagnosticThresholds(near_duplicate_correlation=1.1),
            )


if __name__ == "__main__":
    unittest.main()
