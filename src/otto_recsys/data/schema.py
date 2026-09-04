from __future__ import annotations

import pyarrow as pa

ACTION_TO_ID: dict[str, int] = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}

EVENT_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("event_type", pa.int8(), nullable=False),
        pa.field("event_index", pa.uint16(), nullable=False),
    ]
)
