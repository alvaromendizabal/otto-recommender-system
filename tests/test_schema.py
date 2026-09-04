from otto_recsys.data.schema import ACTION_TO_ID, EVENT_SCHEMA


def test_action_ids_are_stable() -> None:
    assert ACTION_TO_ID == {
        "clicks": 0,
        "carts": 1,
        "orders": 2,
    }


def test_event_schema_contract() -> None:
    assert EVENT_SCHEMA.names == [
        "session",
        "aid",
        "ts",
        "event_type",
        "event_index",
    ]
