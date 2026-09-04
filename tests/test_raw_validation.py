import logging
from pathlib import Path

import pytest

from otto_recsys.data.raw_validation import validate_jsonl


def test_raw_validation_accepts_valid_session(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"

    source.write_text(
        '{"session":1,"events":['
        '{"aid":10,"ts":1000,"type":"clicks"},'
        '{"aid":11,"ts":2000,"type":"carts"},'
        '{"aid":11,"ts":3000,"type":"orders"}'
        ']}\n',
        encoding="utf-8",
    )

    summary = validate_jsonl(
        source,
        logger=logging.getLogger("test"),
        heartbeat_seconds=10.0,
    )

    assert summary.sessions == 1
    assert summary.events == 3
    assert summary.clicks == 1
    assert summary.carts == 1
    assert summary.orders == 1
    assert summary.min_ts == 1000
    assert summary.max_ts == 3000


def test_raw_validation_rejects_time_reversal(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"

    source.write_text(
        '{"session":1,"events":['
        '{"aid":10,"ts":2000,"type":"clicks"},'
        '{"aid":11,"ts":1000,"type":"carts"}'
        ']}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timestamps"):
        validate_jsonl(
            source,
            logger=logging.getLogger("test"),
        )


def test_raw_validation_rejects_unknown_action(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"

    source.write_text(
        '{"session":1,"events":['
        '{"aid":10,"ts":1000,"type":"unknown"}'
        ']}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid action"):
        validate_jsonl(
            source,
            logger=logging.getLogger("test"),
        )
