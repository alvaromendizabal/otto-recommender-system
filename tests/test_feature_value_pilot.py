"""Leakage, normalization, metric and native checkpoint tests for the feature pilot."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_feature_value_pilot.py"
SPEC = importlib.util.spec_from_file_location("feature_value_pilot", SOURCE)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class RelativeDemandTests(unittest.TestCase):
    def test_complete_pool_shape(self) -> None:
        self.assertEqual(pilot.relative_demand(np.ones((400, 6))).shape, (400, 12))

    def test_zero_count_has_no_evidence(self) -> None:
        self.assertTrue((pilot.relative_demand(np.zeros((400, 6))) == 0).all())

    def test_ties_are_equal(self) -> None:
        values = np.repeat(np.array([[0.], [2.], [2.], [4.]]), 6, axis=1)
        result = pilot.relative_demand(values)
        np.testing.assert_allclose(result[:, 0], [0, 0.5, 0.5, 1])
        np.testing.assert_allclose(result[:, 1], [0, 0.25, 0.25, 0.5])

    def test_equal_positive_pool_is_midrank(self) -> None:
        result = pilot.relative_demand(np.ones((400, 6)))
        np.testing.assert_allclose(result[:, ::2], 0.5)
        np.testing.assert_allclose(result[:, 1::2].sum(axis=0), 1, atol=1e-5)

    def test_permutation_equivariant(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.integers(0, 8, (40, 6))
        order = rng.permutation(40)
        np.testing.assert_array_equal(pilot.relative_demand(values[order]),
                                      pilot.relative_demand(values)[order])

    def test_no_input_mutation(self) -> None:
        values = np.ones((20, 6))
        before = values.copy()
        pilot.relative_demand(values)
        np.testing.assert_array_equal(values, before)

    def test_bad_counts_rejected(self) -> None:
        for value in (-1, np.nan, np.inf):
            with self.subTest(value=value):
                values = np.ones((20, 6))
                values[0, 0] = value
                with self.assertRaises(ValueError):
                    pilot.relative_demand(values)

    def test_bad_shapes_rejected(self) -> None:
        for shape in ((6,), (1, 6), (20, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                pilot.relative_demand(np.ones(shape))

    def test_invariant_to_positive_scale(self) -> None:
        values = np.arange(1, 121).reshape(20, 6)
        np.testing.assert_allclose(pilot.relative_demand(values * 10),
                                   pilot.relative_demand(values))


class ProtocolTests(unittest.TestCase):
    def test_purged_forward_folds(self) -> None:
        query = np.arange(1024) * 10 + 100
        folds = pilot.purged_folds(query, embargo_ms=120)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0]["purged"], 12)
        for fold in folds:
            self.assertLess(query[fold["train"]].max(), fold["cutoff"] - 120)
            self.assertFalse(np.intersect1d(fold["train"], fold["valid"]).size)

    def test_tied_cutoff_purged(self) -> None:
        query = np.arange(1024)
        query[510:514] = 511
        first = pilot.purged_folds(query, embargo_ms=0)[0]
        self.assertTrue((query[first["train"]] < first["cutoff"]).all())

    def test_insufficient_purged_support_stops(self) -> None:
        with self.assertRaisesRegex(ValueError, "support"):
            pilot.purged_folds(np.arange(1024), embargo_ms=5000)

    def test_invalid_time_ledger_stops(self) -> None:
        with self.assertRaises(ValueError):
            pilot.purged_folds(np.arange(1024), embargo_ms=-1)
        with self.assertRaises(ValueError):
            pilot.purged_folds(np.arange(1023), embargo_ms=0)

    def test_training_labels_censored_before_sampling(self) -> None:
        ids, aids, q = np.array([1]), np.arange(400)[None, :], np.array([100])
        labels = {1: [(3, 0, 101), (4, 1, 199), (5, 2, 200), (6, 2, 201)]}
        target = pilot.targets_asof(ids, aids, q, labels, 200)
        self.assertEqual(target.sum(axis=(0, 1)).tolist(), [1, 1, 0])
        self.assertEqual(int(target[0, 5, 2]), 0)

    def test_unobserved_outcome_not_used_as_full_period_negative(self) -> None:
        ids, aids, q = np.array([1]), np.arange(400)[None, :], np.array([100])
        before = pilot.targets_asof(ids, aids, q, {1: [(3, 2, 205)]}, 200)
        after = pilot.targets_asof(ids, aids, q, {1: [(3, 2, 205)]}, 210)
        self.assertEqual(int(before.sum()), 0)
        self.assertEqual(int(after.sum()), 1)

    def test_training_label_precedes_query_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pilot.targets_asof(np.array([1]), np.arange(400)[None, :], np.array([100]),
                               {1: [(3, 1, 99)]}, 200)

    def test_training_query_at_validation_cutoff_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pilot.targets_asof(np.array([1]), np.arange(400)[None, :], np.array([200]),
                               {}, 200)

    def test_official_pooled_not_mean_session_recall(self) -> None:
        h = np.array([[1, 1, 1], [0, 0, 0]])
        d = np.array([[1, 1, 1], [9, 9, 9]])
        self.assertAlmostEqual(pilot.pooled(h, d)["weighted_recall_at_20"], 0.1)

    def test_missing_objective_denominator_stops(self) -> None:
        with self.assertRaises(ValueError):
            pilot.pooled(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_illegal_hits_stops(self) -> None:
        with self.assertRaises(ValueError):
            pilot.pooled(np.full((2, 3), 3), np.ones((2, 3)))

    def test_tie_break_by_aid(self) -> None:
        aids = np.arange(40)[::-1][None, :]
        truth = (aids < 20).astype(np.int8)
        result = pilot.session_hits(np.zeros((1, 40)), aids, truth)
        self.assertEqual(result.tolist(), [20])

    def test_metric_rejects_duplicate_candidates(self) -> None:
        with self.assertRaises(ValueError):
            pilot.session_hits(np.zeros((1, 40)), np.zeros((1, 40)), np.zeros((1, 40)))

    def test_metric_rejects_nonfinite_scores(self) -> None:
        with self.assertRaises(ValueError):
            pilot.session_hits(np.full((1, 40), np.nan), np.arange(40)[None, :],
                               np.zeros((1, 40)))

    def test_arms_keep_frozen_schema(self) -> None:
        config = {"baseline_features": [f"b{i}" for i in range(102)],
                  "added_features": [f"a{i}" for i in range(32)],
                  "ablation_blocks": {"block": [f"a{i}" for i in range(32)]}}
        schemas = pilot.arms(config)
        self.assertEqual(len(schemas["baseline_relative"]), 114)
        self.assertEqual(len(schemas["shared_relative"]), 146)
        self.assertEqual(schemas["without_block"], schemas["baseline"])

    def test_invalid_ablation_partition_stops(self) -> None:
        config = {"baseline_features": [f"b{i}" for i in range(102)],
                  "added_features": [f"a{i}" for i in range(32)], "ablation_blocks": {}}
        with self.assertRaises(ValueError):
            pilot.arms(config)

    def test_immutable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            pilot.preserve(path, b"same")
            pilot.preserve(path, b"same")
            with self.assertRaisesRegex(ValueError, "conflict"):
                pilot.preserve(path, b"different")
            self.assertEqual(path.read_bytes(), b"same")

    def test_symlink_checkpoint_stops(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "link"
            path.symlink_to(Path(root) / "missing")
            with self.assertRaises(ValueError):
                pilot.preserve(path, b"bytes")

    def test_native_model_reload_and_zero_fit_reuse(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 2))
        y = np.tile([1, 0, 0, 0, 0, 0], 20)
        params = {"objective": "lambdarank", "verbosity": -1, "num_threads": 1,
                  "num_leaves": 3, "min_data_in_leaf": 2, "deterministic": True,
                  "force_col_wise": True}
        with tempfile.TemporaryDirectory() as root:
            args = (Path(root), {"frozen": True}, x, y, [6] * 20, x, params, 3, ["a", "b"])
            prediction, fits = pilot.fit_or_reuse(*args)
            replay, reused_fits = pilot.fit_or_reuse(*args)
            self.assertEqual(fits, 1)
            self.assertEqual(reused_fits, 0)
            np.testing.assert_array_equal(prediction, replay)


if __name__ == "__main__":
    unittest.main()
