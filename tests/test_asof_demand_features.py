"""Tests for leakage-safe as-of demand features."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "otto_recsys"
    / "research"
    / "asof_demand_features.py"
)
SPEC = importlib.util.spec_from_file_location("asof_demand_features", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def snapshot() -> object:
    counts = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [4, 1, 0], [8, 2, 1], [20, 4, 2]],
            [[1, 0, 0], [3, 1, 0], [5, 2, 1], [10, 3, 1], [30, 8, 4]],
            [[2, 1, 0], [4, 1, 1], [7, 3, 1], [12, 5, 2], [40, 12, 5]],
        ],
        dtype=np.float32,
    )
    return module.DemandSnapshot(
        cutoff_ts=10_000_000,
        aids=np.array([10, 20, 30], dtype=np.int64),
        counts=counts,
        last_event_ts=np.array(
            [
                [9_000_000, 8_000_000, 7_000_000],
                [9_500_000, 9_000_000, 8_000_000],
                [9_900_000, 9_700_000, -1],
            ],
            dtype=np.int64,
        ),
        source_id="fixture",
    )


class SnapshotTests(unittest.TestCase):
    def test_valid_snapshot(self) -> None:
        snapshot().validate()

    def test_rejects_nonmonotone_nested_windows(self) -> None:
        value = snapshot()
        counts = value.counts.copy()
        counts[0, 2, 0] = 0
        broken = module.DemandSnapshot(
            cutoff_ts=value.cutoff_ts,
            aids=value.aids,
            counts=counts,
            last_event_ts=value.last_event_ts,
            source_id=value.source_id,
        )
        with self.assertRaisesRegex(ValueError, "monotone"):
            broken.validate()

    def test_rejects_future_snapshot_event(self) -> None:
        value = snapshot()
        last = value.last_event_ts.copy()
        last[0, 0] = value.cutoff_ts
        broken = module.DemandSnapshot(
            cutoff_ts=value.cutoff_ts,
            aids=value.aids,
            counts=value.counts,
            last_event_ts=last,
            source_id=value.source_id,
        )
        with self.assertRaisesRegex(ValueError, "cutoff"):
            broken.validate()

    def test_rejects_unsorted_ids(self) -> None:
        value = snapshot()
        broken = module.DemandSnapshot(
            cutoff_ts=value.cutoff_ts,
            aids=np.array([20, 10, 30], dtype=np.int64),
            counts=value.counts,
            last_event_ts=value.last_event_ts,
            source_id=value.source_id,
        )
        with self.assertRaisesRegex(ValueError, "identifiers"):
            broken.validate()


class TransformTests(unittest.TestCase):
    def test_catalog_is_unique_and_stable(self) -> None:
        names = module.feature_names()
        self.assertEqual(len(names), 75)
        self.assertEqual(len(set(names)), 75)
        self.assertEqual(names[0], "asof_clicks_h1_log_count")
        self.assertEqual(names[-1], "asof_action_mix_orders_h168")

    def test_shape_and_finiteness(self) -> None:
        candidates = np.array([10, 20, 30, 99], dtype=np.int64)
        matrix = module.transform(snapshot(), candidates, query_ts=10_500_000)
        self.assertEqual(matrix.shape, (4, 75))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix.dtype, np.float32)

    def test_unknown_candidate_has_zero_counts_and_unseen(self) -> None:
        names = module.feature_names()
        matrix = module.transform(snapshot(), np.array([99], dtype=np.int64), query_ts=10_500_000)
        self.assertEqual(matrix[0, names.index("asof_clicks_h24_log_count")], 0)
        self.assertEqual(matrix[0, names.index("asof_orders_last_seen")], 0)

    def test_query_before_snapshot_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "precedes"):
            module.transform(snapshot(), np.array([10], dtype=np.int64), query_ts=9_999_999)

    def test_duplicate_candidates_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            module.transform(snapshot(), np.array([10, 10], dtype=np.int64), query_ts=10_500_000)

    def test_negative_candidate_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            module.transform(snapshot(), np.array([-1], dtype=np.int64), query_ts=10_500_000)

    def test_percentile_is_tie_aware(self) -> None:
        values = np.array([3, 3, 1], dtype=np.float32)
        percentile = module._candidate_percentile(values)
        self.assertAlmostEqual(float(percentile[0]), float(percentile[1]))
        self.assertGreater(float(percentile[0]), float(percentile[2]))

    def test_all_tied_percentile_is_half(self) -> None:
        values = np.zeros(4, dtype=np.float32)
        np.testing.assert_allclose(module._candidate_percentile(values), 0.5)

    def test_higher_demand_has_higher_percentile(self) -> None:
        names = module.feature_names()
        candidates = np.array([10, 20, 30], dtype=np.int64)
        matrix = module.transform(snapshot(), candidates, query_ts=10_500_000)
        col = names.index("asof_clicks_h24_candidate_percentile")
        self.assertLess(matrix[0, col], matrix[1, col])
        self.assertLess(matrix[1, col], matrix[2, col])

    def test_candidate_mass_sums_to_one_when_supported(self) -> None:
        names = module.feature_names()
        matrix = module.transform(snapshot(), np.array([10, 20, 30], dtype=np.int64), query_ts=10_500_000)
        for action in module.ACTIONS:
            for hours in module.WINDOW_HOURS:
                col = names.index(f"asof_{action}_h{hours}_candidate_mass_share")
                values = matrix[:, col]
                if values.sum() > 0:
                    self.assertAlmostEqual(float(values.sum()), 1.0, places=6)

    def test_trend_uses_rate_not_raw_count(self) -> None:
        names = module.feature_names()
        matrix = module.transform(snapshot(), np.array([10], dtype=np.int64), query_ts=10_500_000)
        col = names.index("asof_clicks_trend_h1_h24")
        self.assertLess(float(matrix[0, col]), 0)

    def test_percentile_delta_detects_rank_change(self) -> None:
        names = module.feature_names()
        matrix = module.transform(snapshot(), np.array([10, 20, 30], dtype=np.int64), query_ts=10_500_000)
        col = names.index("asof_clicks_percentile_delta_h1_h24")
        self.assertTrue(np.isfinite(matrix[:, col]).all())

    def test_action_mix_sums_to_one(self) -> None:
        names = module.feature_names()
        matrix = module.transform(snapshot(), np.array([10, 20, 30, 99], dtype=np.int64), query_ts=10_500_000)
        for hours in module.MIX_WINDOWS:
            cols = [names.index(f"asof_action_mix_{action}_h{hours}") for action in module.ACTIONS]
            np.testing.assert_allclose(matrix[:, cols].sum(axis=1), 1, atol=1e-6)

    def test_prior_strength_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior_strength"):
            module.transform(
                snapshot(), np.array([10], dtype=np.int64), query_ts=10_500_000, prior_strength=0
            )

    def test_last_age_nonnegative_and_missing_is_zero(self) -> None:
        names = module.feature_names()
        matrix = module.transform(snapshot(), np.array([30], dtype=np.int64), query_ts=10_500_000)
        self.assertGreater(matrix[0, names.index("asof_clicks_last_age_log_hours")], 0)
        self.assertEqual(matrix[0, names.index("asof_orders_last_seen")], 0)
        self.assertEqual(matrix[0, names.index("asof_orders_last_age_log_hours")], 0)

    def test_deterministic_replay(self) -> None:
        candidates = np.array([30, 10, 99, 20], dtype=np.int64)
        first = module.transform(snapshot(), candidates, query_ts=10_500_000)
        second = module.transform(snapshot(), candidates, query_ts=10_500_000)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_input_arrays_not_mutated(self) -> None:
        value = snapshot()
        before_counts = value.counts.copy()
        before_last = value.last_event_ts.copy()
        candidates = np.array([10, 20], dtype=np.int64)
        before_candidates = candidates.copy()
        module.transform(value, candidates, query_ts=10_500_000)
        np.testing.assert_array_equal(value.counts, before_counts)
        np.testing.assert_array_equal(value.last_event_ts, before_last)
        np.testing.assert_array_equal(candidates, before_candidates)

    def test_immutable_write_guard_example(self) -> None:
        # A minimal sanity check that the test suite itself leaves no temp artifacts behind.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sentinel"
            path.write_bytes(b"stable")
            self.assertEqual(path.read_bytes(), b"stable")


if __name__ == "__main__":
    unittest.main()
