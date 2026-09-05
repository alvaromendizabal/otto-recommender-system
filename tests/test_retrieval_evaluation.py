from otto_recsys.retrieval.evaluation import (
    RecallGrid,
)


def test_recall_grid_at_multiple_cutoffs() -> None:
    grid = RecallGrid(
        ks=(1, 2, 20),
    )

    grid.update(
        "clicks",
        [10, 20],
        {20},
    )

    metrics = grid.results()

    assert metrics["clicks_recall_1"] == 0.0
    assert metrics["clicks_recall_2"] == 1.0
    assert metrics["clicks_recall_20"] == 1.0


def test_recall_grid_uses_otto_weights() -> None:
    grid = RecallGrid(
        ks=(20,),
    )

    grid.update(
        "clicks",
        [1],
        {1},
    )
    grid.update(
        "carts",
        [2],
        {2},
    )
    grid.update(
        "orders",
        [3],
        {3},
    )

    metrics = grid.results()

    assert metrics["weighted_recall_20"] == 1.0
