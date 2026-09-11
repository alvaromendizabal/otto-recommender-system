"""Independent deterministic fixtures for the label-blind candidate-frontier library."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from otto_recsys.research.candidate_frontier import (
    ARMS,
    CandidateFrontier,
    candidate_ceiling,
    integer_ids,
    source_ranks,
)


@dataclass
class PrefixFixture:
    session: int
    aid: np.ndarray
    ts: np.ndarray
    kind: np.ndarray

    def validate(self) -> None:
        if (not self.aid.size or self.aid.shape != self.ts.shape
                or self.aid.shape != self.kind.shape or (np.diff(self.ts) < 0).any()
                or not np.isin(self.kind, [0, 1, 2]).all()):
            raise ValueError("invalid observed prefix")
        integer_ids(self.aid)


class FakeEngine:
    def __init__(self) -> None:
        self.cutoff = 100
        self.calls = 0
        # 1 -> 400..1399, followed by 400 -> 1499 (only a two-hop discovery).
        source = np.r_[np.ones(1000, dtype=np.int64), 400]
        target = np.r_[np.arange(400, 1400), 1499]
        values = np.r_[np.linspace(2, 1, 1000), 100.0].astype(np.float32)
        graph = csr_matrix((values, (source, target)), shape=(1600, 1600))
        self.graphs = [graph, graph.copy(), graph.copy()]
        self.popular = [np.asarray([0, 2]), np.asarray([3]), np.asarray([4])]
        self.baseline = np.arange(400, dtype=np.int64)
        self.calls = 0

    def candidates(self, prefix: Any, budget: int = 400) -> Any:
        del prefix
        self.calls += 1
        return SimpleNamespace(aid=self.baseline[:budget])


def fixture() -> tuple[Any, Any, Any]:
    engine = FakeEngine()
    wide = {family: SimpleNamespace(cutoff=100, graphs=[g.copy() for g in engine.graphs])
            for family in ("symmetric", "forward")}
    prefix = PrefixFixture(17, np.array([1, 2]), np.array([101, 102]), np.array([0, 1]))
    return cast(Any, engine), cast(Any, wide), cast(Any, prefix)


class CandidateFrontierTests(unittest.TestCase):
    def test_all_arms_preserve_exact_400_candidate_prefix(self) -> None:
        engine, wide, prefix = fixture()
        arms = CandidateFrontier(engine, wide).frontiers(prefix)
        self.assertEqual(tuple(arms), tuple(ARMS))
        for arm, (budget, _) in ARMS.items():
            with self.subTest(arm=arm):
                np.testing.assert_array_equal(arms[arm].aid[:400], engine.baseline)
                self.assertEqual(arms[arm].aid.size, budget)
                self.assertEqual(np.unique(arms[arm].aid).size, budget)

    def test_800_is_exact_prefix_of_1200_with_same_source_values(self) -> None:
        engine, wide, prefix = fixture()
        arms = CandidateFrontier(engine, wide).frontiers(prefix)
        for family in ("onehop", "twohop"):
            small, large = arms[f"wide_{family}_800"], arms[f"wide_{family}_1200"]
            np.testing.assert_array_equal(small.aid, large.aid[:800])
            np.testing.assert_array_equal(small.source_scores, large.source_scores[:800])
            np.testing.assert_array_equal(small.source_ranks, large.source_ranks[:800])

    def test_twohop_recovers_item_not_in_onehop_frontier(self) -> None:
        engine, wide, prefix = fixture()
        arms = CandidateFrontier(engine, wide).frontiers(prefix)
        self.assertNotIn(1499, arms["wide_onehop_1200"].aid)
        self.assertIn(1499, arms["wide_twohop_1200"].aid)

    def test_baseline_generation_runs_once_for_all_arms(self) -> None:
        engine, wide, prefix = fixture()
        CandidateFrontier(engine, wide).frontiers(prefix)
        self.assertEqual(engine.calls, 1)

    def test_deterministic_replay_and_input_nonmutation(self) -> None:
        engine, wide, prefix = fixture()
        before = [(g.data.copy(), g.indices.copy()) for g in engine.graphs]
        original = prefix.aid.copy()
        frontier = CandidateFrontier(engine, wide)
        first, second = frontier.frontiers(prefix), frontier.frontiers(prefix)
        for arm in ARMS:
            np.testing.assert_array_equal(first[arm].aid, second[arm].aid)
            np.testing.assert_array_equal(first[arm].source_scores, second[arm].source_scores)
        for graph, (values, indices) in zip(engine.graphs, before, strict=True):
            np.testing.assert_array_equal(graph.data, values)
            np.testing.assert_array_equal(graph.indices, indices)
        np.testing.assert_array_equal(prefix.aid, original)

    def test_baseline_control_does_not_claim_source_evidence(self) -> None:
        engine, wide, prefix = fixture()
        baseline = CandidateFrontier(engine, wide).candidates(prefix, budget=400, hops=1)
        self.assertFalse(baseline.evidence_available)
        np.testing.assert_array_equal(baseline.aid, engine.baseline)

    def test_unknown_item_ids_have_no_accidental_graph_edges(self) -> None:
        engine, wide, prefix = fixture()
        prefix.aid = np.array([9999, 9999])
        frontier = CandidateFrontier(engine, wide).candidates(prefix, budget=800, hops=2)
        self.assertIn(9999, frontier.aid)
        self.assertNotIn(1499, frontier.aid)
        self.assertEqual(frontier.aid.size, 401)

    def test_empty_graphs_do_not_pad_or_duplicate_candidates(self) -> None:
        engine, wide, prefix = fixture()
        empty = [csr_matrix((1600, 1600), dtype=np.float32) for _ in range(3)]
        engine.graphs = empty
        for graph in wide.values():
            graph.graphs = empty
        arms = CandidateFrontier(engine, wide).frontiers(prefix)
        for value in arms.values():
            np.testing.assert_array_equal(value.aid, engine.baseline)

    def test_cutoff_and_missing_graph_family_are_rejected(self) -> None:
        engine, wide, _ = fixture()
        wide["forward"].cutoff = 101
        with self.assertRaisesRegex(ValueError, "cutoff"):
            CandidateFrontier(engine, wide)
        del wide["forward"]
        with self.assertRaisesRegex(ValueError, "both"):
            CandidateFrontier(engine, wide)

    def test_query_before_cutoff_is_rejected(self) -> None:
        engine, wide, prefix = fixture()
        prefix.ts = np.array([99, 101])
        with self.assertRaisesRegex(ValueError, "cutoff"):
            CandidateFrontier(engine, wide).frontiers(prefix)

    def test_invalid_bridge_and_resource_settings_are_rejected(self) -> None:
        engine, wide, _ = fixture()
        for value in (0, 65, 1.5, True):
            with self.subTest(bridge=value), self.assertRaises(ValueError):
                CandidateFrontier(engine, wide, bridge_per_source=value)
        for value in (0, np.inf, np.nan):
            with self.subTest(rrf=value), self.assertRaises(ValueError):
                CandidateFrontier(engine, wide, rrf_k=value)

    def test_invalid_candidate_budget_and_hops_are_rejected(self) -> None:
        engine, wide, prefix = fixture()
        frontier = CandidateFrontier(engine, wide)
        for budget, hops in ((8, 2), (399, 1), (1601, 1), (800.0, 1), (800, 3), (800, True)):
            with self.subTest(budget=budget, hops=hops), self.assertRaises(ValueError):
                frontier.candidates(prefix, budget=budget, hops=hops)

    def test_bad_baseline_ids_are_rejected(self) -> None:
        engine, wide, prefix = fixture()
        for bad in (np.array([1, 1]), np.array([-1, 0]), np.array([1.5]), np.array([], dtype=int)):
            engine.baseline = bad
            with self.subTest(baseline=bad), self.assertRaises(ValueError):
                CandidateFrontier(engine, wide).frontiers(prefix)

    def test_negative_nonfinite_and_rectangular_graphs_rejected(self) -> None:
        for value in (-1.0, np.inf, np.nan):
            engine, wide, prefix = fixture()
            engine.graphs[0].data[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                CandidateFrontier(engine, wide).frontiers(prefix)
        engine, wide, _ = fixture()
        engine.graphs[0] = csr_matrix((2, 3))
        with self.assertRaisesRegex(ValueError, "square"):
            CandidateFrontier(engine, wide)

    def test_discovery_resource_limit_stops_without_silent_truncation(self) -> None:
        engine, wide, prefix = fixture()
        engine.popular[0] = np.arange(2000)
        with self.assertRaisesRegex(ValueError, "cap"):
            CandidateFrontier(engine, wide, max_discovery_items=1600).frontiers(prefix)

    def test_ties_rank_by_item_and_zero_evidence_has_no_rank(self) -> None:
        result = source_ranks(np.array([2.0, 0.0, 2.0, 1.0]), np.array([5, 1, 3, 9]))
        np.testing.assert_array_equal(result, [2, 0, 1, 3])

    def test_ceiling_uses_full_truth_and_caps_at_twenty(self) -> None:
        truth = (np.array([1]), np.array([2, 8]), np.arange(30))
        np.testing.assert_array_equal(candidate_ceiling(np.arange(30), truth, np.array([1, 2, 20])),
                                      [1, 2, 20])
        result = candidate_ceiling(np.array([1, 2]), truth, np.array([1, 2, 20]))
        np.testing.assert_array_equal(result, [1, 1, 2])

    def test_bad_denominators_and_duplicate_labels_rejected(self) -> None:
        truth = (np.array([1]), np.array([2]), np.array([3]))
        for denominator in (np.array([1, 1, 2]), np.array([1.0, 1.0, 1.0]), np.array([-1, 1, 1])):
            with self.subTest(denominator=denominator), self.assertRaises(ValueError):
                candidate_ceiling(np.arange(4), truth, denominator)
        with self.assertRaisesRegex(ValueError, "unique"):
            candidate_ceiling(np.arange(4), (truth[0], np.array([2, 2]), truth[2]), np.ones(3, int))
        with self.assertRaisesRegex(ValueError, "next-click"):
            candidate_ceiling(
                np.arange(4), (np.array([1, 2]), truth[1], truth[2]), np.array([2, 1, 1])
            )

    def test_empty_truth_and_missing_targets_are_not_dropped(self) -> None:
        empty = np.array([], dtype=np.int64)
        result = candidate_ceiling(empty, (empty, empty, empty), np.zeros(3, int))
        np.testing.assert_array_equal(result, [0, 0, 0])
        result = candidate_ceiling(empty, (np.array([9]), empty, empty), np.array([1, 0, 0]))
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_unsigned_overflow_and_float_ids_rejected(self) -> None:
        for values in (np.array([2**63], dtype=np.uint64), np.array([1.2]), np.array([[1]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                integer_ids(values)


if __name__ == "__main__":
    unittest.main()
