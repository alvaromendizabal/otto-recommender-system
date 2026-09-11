"""Guards for nested label-blind candidate frontier expansion."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from otto_recsys.research.candidate_frontier import CandidateFrontier, candidate_ceiling
from otto_recsys.research.features import Candidates, Prefix


class FakeEngine:
    def __init__(self) -> None:
        self.cutoff = 100
        self.known = np.ones(10, dtype=np.float32)
        # 1 -> 4 -> 7 creates a two-hop-only item 7.
        base = csr_matrix(
            (
                np.asarray([1.0, 1.0, 0.8], dtype=np.float32),
                (np.asarray([1, 4, 2]), np.asarray([4, 7, 5])),
            ),
            shape=(10, 10),
        )
        self.graphs = [base, base.copy(), base.copy()]
        self.popular = [np.asarray([8]), np.asarray([8]), np.asarray([9])]

    def candidates(self, prefix: Prefix, budget: int = 400) -> Candidates:
        del prefix
        ids = np.asarray([1, 2, 3], dtype=np.int64)[:budget]
        return Candidates(
            ids,
            np.zeros((ids.size, 5), dtype=np.float32),
            np.zeros((ids.size, 5), dtype=np.float32),
            np.zeros((1, ids.size, 3), dtype=np.float32),
        )


def wide(cutoff: int = 100) -> dict[str, SimpleNamespace]:
    graph = csr_matrix(
        (
            np.asarray([2.0, 1.0], dtype=np.float32),
            (np.asarray([1, 4]), np.asarray([4, 7])),
        ),
        shape=(10, 10),
    )
    value = SimpleNamespace(cutoff=cutoff, graphs=[graph, graph.copy(), graph.copy()])
    return {"symmetric": value, "forward": value}


class CandidateFrontierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prefix = Prefix(
            1,
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([101, 102], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
        )

    def test_expansion_preserves_exact_baseline_prefix(self) -> None:
        frontier = CandidateFrontier(FakeEngine(), wide(), bridge_per_source=2)
        expanded = frontier.candidates(self.prefix, budget=8, hops=2)
        np.testing.assert_array_equal(expanded.aid[:3], [1, 2, 3])
        self.assertEqual(np.unique(expanded.aid).size, expanded.aid.size)
        self.assertIn(7, expanded.aid.tolist())

    def test_budget_400_delegates_exact_baseline(self) -> None:
        frontier = CandidateFrontier(FakeEngine(), wide())
        baseline = frontier.candidates(self.prefix, budget=400, hops=1)
        np.testing.assert_array_equal(baseline.aid, [1, 2, 3])

    def test_cutoff_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cutoff"):
            CandidateFrontier(FakeEngine(), wide(cutoff=99))

    def test_candidate_ceiling_is_capped_and_label_only_metric(self) -> None:
        hits = candidate_ceiling(
            np.asarray([1, 2, 3], dtype=np.int64),
            (
                np.asarray([1]),
                np.asarray([2, 9]),
                np.asarray([1, 2, 3, 8]),
            ),
            np.asarray([1, 2, 4], dtype=np.int64),
        )
        np.testing.assert_array_equal(hits, [1, 1, 3])

    def test_invalid_frontier_settings_are_rejected(self) -> None:
        frontier = CandidateFrontier(FakeEngine(), wide())
        for budget, hops in ((399, 1), (1601, 1), (800, 3)):
            with self.subTest(budget=budget, hops=hops):
                with self.assertRaisesRegex(ValueError, "budget"):
                    frontier.candidates(self.prefix, budget=budget, hops=hops)


if __name__ == "__main__":
    unittest.main()
