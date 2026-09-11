"""Label-blind nested retrieval tests; no real labels or external graph files."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from otto_recsys.research.candidate_frontier import (
    ARMS,
    CandidateFrontier,
    candidate_ceiling,
    positive_ranks,
)


class FakeEngine:
    def __init__(self) -> None:
        self.cutoff = 100
        self.graphs = [csr_matrix((np.array([2.0, 1.0, 0.8], dtype=np.float32),
                                  ([1, 4, 2], [4, 7, 5])), shape=(1500, 1500))] * 3
        self.popular = [np.array([8]), np.array([9]), np.array([10])]
        self.calls = 0

    def candidates(self, prefix: object, budget: int) -> SimpleNamespace:
        del prefix
        assert budget == 400
        self.calls += 1
        return SimpleNamespace(aid=np.array([1, 2, 3]))


def setup() -> tuple[CandidateFrontier, SimpleNamespace]:
    engine = FakeEngine()
    graphs = {f: SimpleNamespace(cutoff=100, graphs=engine.graphs)
              for f in ("symmetric", "forward")}
    prefix = SimpleNamespace(aid=np.array([1, 2]), ts=np.array([101, 102]),
                             validate=lambda: None)
    return CandidateFrontier(engine, graphs), prefix


def test_nested_frontiers_and_twohop_only_item() -> None:
    f, prefix = setup()
    result = f.frontiers(prefix)
    assert tuple(result) == ARMS
    assert f.engine.calls == 1
    for value in result.values():
        np.testing.assert_array_equal(value.aid[:3], [1, 2, 3])
        assert np.unique(value.aid).size == value.aid.size
        assert np.isfinite(value.source_scores).all()
    assert 7 not in result["wide_onehop_1200"].aid
    assert 7 in result["wide_twohop_1200"].aid


def test_replay_and_input_nonmutation() -> None:
    f, prefix = setup()
    before = prefix.aid.copy()
    a, b = f.frontiers(prefix), f.frontiers(prefix)
    for name in ARMS:
        np.testing.assert_array_equal(a[name].aid, b[name].aid)
        np.testing.assert_array_equal(a[name].source_scores, b[name].source_scores)
    np.testing.assert_array_equal(before, prefix.aid)


def test_budget_slices_are_nested_on_large_catalogue() -> None:
    f, prefix = setup()
    graph = csr_matrix((np.ones(1499, dtype=np.float32),
                        (np.ones(1499, dtype=int), np.arange(1, 1500))), shape=(1500, 1500))
    f.graphs = [(name, graph) for name, _ in f.graphs]
    out = f.frontiers(prefix)
    assert out["wide_onehop_800"].aid.size == 800
    assert out["wide_onehop_1200"].aid.size == 1200
    np.testing.assert_array_equal(out["wide_onehop_800"].aid,
                                  out["wide_onehop_1200"].aid[:800])


def test_unknown_seed_is_safe() -> None:
    f, prefix = setup()
    prefix.aid = np.array([9000, 9001])
    assert f.frontiers(prefix)["wide_twohop_1200"].aid.size >= 3


def test_cutoff_is_enforced() -> None:
    f, prefix = setup()
    prefix.ts[0] = 99
    with pytest.raises(ValueError, match="cutoff"):
        f.frontiers(prefix)
    engine = FakeEngine()
    wide = {x: SimpleNamespace(cutoff=99, graphs=engine.graphs)
            for x in ("symmetric", "forward")}
    with pytest.raises(ValueError, match="cutoff"):
        CandidateFrontier(engine, wide)


def test_discovery_cap_stops_without_silent_truncation() -> None:
    f, prefix = setup()
    f.max_discovery_items = 4
    with pytest.raises(ValueError, match="Discovery"):
        f.frontiers(prefix)


@pytest.mark.parametrize("bridge", [0, 65, True])
def test_bridge_cap(bridge: int) -> None:
    engine = FakeEngine()
    wide = {x: SimpleNamespace(cutoff=100, graphs=engine.graphs)
            for x in ("symmetric", "forward")}
    with pytest.raises(ValueError, match="Bridge"):
        CandidateFrontier(engine, wide, bridge_per_source=bridge)


def test_rank_ties_use_item_id_and_zero_is_missing() -> None:
    np.testing.assert_array_equal(positive_ranks(np.array([2., 2., 0.]),
                                                np.array([9, 3, 1])), [2, 1, 0])


def test_oracle_preserves_missing_truth_and_caps_twenty() -> None:
    values = candidate_ceiling(np.arange(30), (np.array([1]), np.arange(40), np.array([99])),
                               np.array([1, 20, 1]))
    np.testing.assert_array_equal(values, [1, 20, 0])


@pytest.mark.parametrize("truth,denom", [(np.array([1, 1]), 2), (np.array([1, 9]), 1)])
def test_duplicate_or_truncated_truth_rejected(truth: np.ndarray, denom: int) -> None:
    with pytest.raises(ValueError, match="denominator"):
        candidate_ceiling(np.array([1]), (truth, np.array([], dtype=int),
                                        np.array([], dtype=int)), np.array([denom, 0, 0]))


def test_empty_targets_and_empty_candidates() -> None:
    empty = np.array([], dtype=int)
    np.testing.assert_array_equal(candidate_ceiling(empty, (empty, empty, empty),
                                                    np.zeros(3, dtype=int)), [0, 0, 0])


def test_duplicate_candidates_and_float_ids_rejected() -> None:
    empty = np.array([], dtype=int)
    for ids in (np.array([1, 1]), np.array([1.0])):
        with pytest.raises(ValueError):
            candidate_ceiling(ids, (empty, empty, empty), np.zeros(3, dtype=int))
