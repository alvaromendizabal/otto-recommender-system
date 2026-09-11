"""Pure tests for fitting-only OOF screen contracts."""

from __future__ import annotations

import unittest

import numpy as np

from otto_recsys.research.fit_screen_contracts import advancement, fold_assignment


class FitScreenContractsTests(unittest.TestCase):
    def test_fold_assignment_is_deterministic_and_complete(self) -> None:
        ids = np.arange(100, 220, dtype=np.int64)
        first = fold_assignment(ids, seed=20260911, folds=3)
        second = fold_assignment(ids.copy(), seed=20260911, folds=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first.tolist()), {0, 1, 2})

    def test_fold_assignment_rejects_duplicate_sessions(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            fold_assignment(np.array([1, 1, 2]), seed=1, folds=2)

    def test_advancement_requires_all_three_conditions(self) -> None:
        baseline = {
            "weighted_recall_at_20": 0.50,
            "objectives": {"orders": {"recall_at_20": 0.50}},
        }
        shared = {
            "weighted_recall_at_20": 0.502,
            "objectives": {"orders": {"recall_at_20": 0.501}},
        }
        passed = advancement(
            baseline,
            shared,
            [0.001, -0.001, 0.004],
            minimum_gain=0.001,
            minimum_nonnegative_folds=2,
        )
        self.assertTrue(passed["advance_to_fitting_only_block_ablation"])
        failed = advancement(
            baseline,
            shared,
            [-0.001, -0.002, 0.009],
            minimum_gain=0.001,
            minimum_nonnegative_folds=2,
        )
        self.assertFalse(failed["advance_to_fitting_only_block_ablation"])

    def test_advancement_rejects_orders_decline(self) -> None:
        baseline = {
            "weighted_recall_at_20": 0.50,
            "objectives": {"orders": {"recall_at_20": 0.50}},
        }
        shared = {
            "weighted_recall_at_20": 0.51,
            "objectives": {"orders": {"recall_at_20": 0.49}},
        }
        decision = advancement(
            baseline,
            shared,
            [0.01, 0.01, 0.01],
            minimum_gain=0.001,
            minimum_nonnegative_folds=2,
        )
        self.assertFalse(decision["advance_to_fitting_only_block_ablation"])


if __name__ == "__main__":
    unittest.main()
