import json
import logging
from pathlib import Path

from otto_recsys.logging_utils import JsonFormatter, configure_logging


def test_json_formatter_contains_utc_timestamp() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.event = "test_event"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello"
    assert payload["event"] == "test_event"
    assert payload["timestamp"].endswith("+00:00")


def test_configure_logging_writes_jsonl(tmp_path: Path) -> None:
    logger = configure_logging("unit_log", log_dir=tmp_path)
    logger.info("hello", extra={"event": "unit"})

    log_path = tmp_path / "unit_log.jsonl"
    assert log_path.exists()

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "unit"
