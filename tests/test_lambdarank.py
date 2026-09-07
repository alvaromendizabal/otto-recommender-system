from __future__ import annotations

import logging

import numpy as np
import pytest

from otto_recsys.ranking.lambdarank import (
    QueryBatch,
    RankerConfig,
    aggregate_official,
    evaluate,
    fit_ranker,
)


def batch(offset: int = 0) -> QueryBatch:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(480, 3)).astype(np.float32)
    sessions = np.repeat(np.arange(12) + offset, 40)
    labels = (x[:, 0] > 0).astype(int)
    counts = {int(s): int(labels[sessions == s].sum()) + 2 for s in np.unique(sessions)}
    return QueryBatch.create(x, sessions, np.tile(np.arange(40), 12), labels,
                             counts, ("revisit_rank", "source_count", "item_age_ms"))


def test_grouping_is_stable() -> None:
    data = batch()
    assert data.groups.tolist() == [40] * 12
    assert data.groups.sum() == data.features.shape[0]
    assert not data.features.flags.writeable
    assert data.fingerprint() == batch().fingerprint()


def test_full_denominator_and_empty_queries() -> None:
    data = QueryBatch.create([[1.], [0.]], [1, 1], [10, 11], [1, 0],
                             {1: 3, 2: 4}, ["score"])
    result = evaluate(data, [2, 1])
    assert result["hits"] == 1
    assert result["denominator"] == 7
    assert result["recall_at_20"] == 1 / 7
    assert result["mrr_at_20"] == 0.5
    assert result["hit_rate_at_20"] == 0.5


def test_deterministic_ties_and_ranked_ceiling() -> None:
    data = QueryBatch.create(np.ones((21, 1)), [1] * 21, list(range(21)),
                             [0] * 20 + [1], {1: 1}, ["score"])
    result = evaluate(data, np.ones(21))
    assert result["recall_at_20"] == 0
    assert result["candidate_ceiling_at_20"] == 1


def test_unlabeled_metric_is_not_zero_success() -> None:
    data = QueryBatch.create(np.empty((0, 1)), np.array([], dtype=int),
                             np.array([], dtype=int), np.array([], dtype=int), {1: 0}, ["x"])
    assert evaluate(data, [])['recall_at_20'] is None


@pytest.mark.parametrize("name", ["target", "session", "true_items", "fold", "aid"])
def test_leakage_columns_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="exclude"):
        QueryBatch.create([[1]], [1], [1], [1], {1: 1}, [name])


@pytest.mark.parametrize("sessions,items,labels,counts", [
    ([1, 1], [3, 3], [1, 0], {1: 1}),
    ([1, 2], [3, 4], [1, 0], {1: 1}),
    ([1, 1], [3, 4], [1, 1], {1: 1}),
    ([1.5, 2], [3, 4], [1, 0], {1: 1, 2: 0}),
    ([1, 2], [3, 4], [2, 0], {1: 2, 2: 0}),
])
def test_invalid_candidates_rejected(sessions, items, labels, counts) -> None:
    with pytest.raises(ValueError):
        QueryBatch.create([[1], [2]], sessions, items, labels, counts, ["x"])


def test_aggregate_uses_pooled_denominators() -> None:
    first = {name: {"hits": 1, "denominator": 1} for name in ("clicks", "carts", "orders")}
    second = {name: {"hits": 0, "denominator": 9} for name in first}
    assert aggregate_official([first, second])["weighted_recall_at_20"] == pytest.approx(0.1)


def test_outer_overlap_fails_before_writing(tmp_path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        fit_ranker(batch(), batch(100), outer_sessions=[0], objective="orders",
                   directory=tmp_path, config=RankerConfig(), logger=logging.getLogger("test"))
    assert not (tmp_path / "contract.json").exists()


def test_feature_order_fails(tmp_path) -> None:
    inner = batch(100)
    inner = QueryBatch.create(inner.features, inner.session, inner.aid, inner.target,
                              inner.truth_counts, list(reversed(inner.feature_names)))
    with pytest.raises(ValueError, match="feature order"):
        fit_ranker(batch(), inner, outer_sessions=[500], objective="orders", directory=tmp_path,
                   config=RankerConfig(), logger=logging.getLogger("test"))


def test_training_resume_and_contract_change(tmp_path) -> None:
    config = RankerConfig(rounds=15, patience=4, checkpoint_every=2,
                           min_data_in_leaf=5, threads=1)
    kwargs = dict(outer_sessions=[500], objective="orders", directory=tmp_path,
                   config=config, logger=logging.getLogger("test"))
    model, state = fit_ranker(batch(), batch(100), **kwargs)
    assert state["complete"] and state["best_iteration"] >= 1
    repeated, reused = fit_ranker(batch(), batch(100), **kwargs)
    np.testing.assert_allclose(model.predict(batch(100).features),
                                repeated.predict(batch(100).features), atol=0, rtol=0)
    assert reused == state
    with pytest.raises(ValueError, match="contract mismatch"):
        fit_ranker(batch(1000), batch(100), **kwargs)


def test_interrupt_resume_matches_uninterrupted(tmp_path) -> None:
    config = RankerConfig(rounds=8, patience=8, checkpoint_every=2,
                           min_data_in_leaf=5, threads=1)
    kwargs = dict(outer_sessions=[500], objective="orders", config=config,
                   logger=logging.getLogger("test"))

    def fail_once(directory) -> None:
        raise RuntimeError("simulated upload interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        fit_ranker(batch(), batch(100), directory=tmp_path / "resumed", publish=fail_once, **kwargs)
    resumed, state = fit_ranker(batch(), batch(100), directory=tmp_path / "resumed", **kwargs)
    fresh, expected = fit_ranker(batch(), batch(100), directory=tmp_path / "fresh", **kwargs)
    assert state["best_iteration"] == expected["best_iteration"]
    np.testing.assert_allclose(resumed.predict(batch(100).features),
                                fresh.predict(batch(100).features), atol=0, rtol=0)


def test_corrupt_latest_checkpoint_falls_back(tmp_path) -> None:
    config = RankerConfig(rounds=8, patience=8, checkpoint_every=2,
                           min_data_in_leaf=5, threads=1)
    kwargs = dict(outer_sessions=[500], objective="orders", directory=tmp_path,
                   config=config, logger=logging.getLogger("test"))
    model, _ = fit_ranker(batch(), batch(100), **kwargs)
    latest = sorted((tmp_path / "checkpoints").glob("*.json"))[-1]
    latest.write_text("broken")
    recovered, _ = fit_ranker(batch(), batch(100), **kwargs)
    np.testing.assert_allclose(model.predict(batch(100).features),
                                recovered.predict(batch(100).features), atol=0, rtol=0)


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [1., 2.]])
def test_invalid_scores_rejected(scores) -> None:
    data = QueryBatch.create([[1.]], [1], [1], [1], {1: 1}, ["x"])
    with pytest.raises(ValueError, match="scores"):
        evaluate(data, scores)


def test_recall_caps_per_query_not_total() -> None:
    data = QueryBatch.create(np.ones((22, 1)), [1] * 21 + [2], list(range(22)),
                             [1] * 22, {1: 100, 2: 1}, ["x"])
    result = evaluate(data, np.ones(22))
    assert result["hits"] == result["denominator"] == 21
    assert result["recall_at_20"] == 1.0


def test_query_ledger_is_immutable() -> None:
    data = batch()
    with pytest.raises(TypeError):
        data.truth_counts[1] = 900


def test_changed_inner_features_invalidate_resume(tmp_path) -> None:
    config = RankerConfig(rounds=4, patience=4, checkpoint_every=2,
                           min_data_in_leaf=5, threads=1)
    kwargs = dict(outer_sessions=[500], objective="orders", directory=tmp_path,
                   config=config, logger=logging.getLogger("test"))
    fit_ranker(batch(), batch(100), **kwargs)
    inner = batch(100)
    changed = QueryBatch.create(inner.features + 1, inner.session, inner.aid,
                                inner.target, inner.truth_counts, inner.feature_names)
    with pytest.raises(ValueError, match="contract mismatch"):
        fit_ranker(batch(), changed, **kwargs)


def test_complete_fit_republishes_without_training(tmp_path) -> None:
    config = RankerConfig(rounds=4, patience=4, checkpoint_every=2,
                           min_data_in_leaf=5, threads=1)
    kwargs = dict(outer_sessions=[500], objective="orders", directory=tmp_path,
                   config=config, logger=logging.getLogger("test"))
    _, first = fit_ranker(batch(), batch(100), **kwargs)
    calls = []
    _, second = fit_ranker(batch(), batch(100), publish=calls.append, **kwargs)
    assert calls == [tmp_path]
    assert first == second


def test_workspace_lock_rejects_concurrent_writer(tmp_path) -> None:
    import fcntl

    with (tmp_path / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active writer"):
            fit_ranker(batch(), batch(100), outer_sessions=[500], objective="orders",
                       directory=tmp_path, config=RankerConfig(), logger=logging.getLogger("test"))


def test_fit_has_signal_and_uses_only_inner_metric(tmp_path) -> None:
    config = RankerConfig(rounds=8, patience=8, checkpoint_every=2,
                           min_data_in_leaf=5, threads=1)
    model, state = fit_ranker(batch(), batch(100), outer_sessions=[500], objective="orders",
                              directory=tmp_path, config=config, logger=logging.getLogger("test"))
    predicted = evaluate(batch(100), model.predict(batch(100).features))["recall_at_20"]
    unranked = evaluate(batch(100), np.zeros(480))["recall_at_20"]
    assert predicted > unranked
    assert predicted == pytest.approx(state["best_score"])
