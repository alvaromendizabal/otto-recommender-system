from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar


def utc_now_iso() -> str:
    """Return an offset-aware UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    """JSON-lines formatter for machine-readable experiment logs."""

    _reserved: ClassVar[set[str]] = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._reserved and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


class ConsoleFormatter(logging.Formatter):
    """Readable UTC console formatter."""

    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event", record.getMessage()))
        details: list[str] = []
        for key in (
            "stage",
            "status",
            "bucket",
            "buckets",
            "restored_parts",
            "uri",
            "elapsed_seconds",
            "rss_mb",
            "cpu_percent",
            "sessions",
            "events",
            "throughput",
        ):
            if hasattr(record, key):
                details.append(f"{key}={getattr(record, key)}")
        suffix = f" | {' '.join(details)}" if details else ""
        return f"[{utc_now_iso()}] {record.levelname:<7} {event}{suffix}"


def configure_logging(
    run_name: str,
    *,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure idempotent console plus JSONL logging."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(run_name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(ConsoleFormatter())

    json_handler = logging.FileHandler(
        directory / f"{run_name}.jsonl",
        encoding="utf-8",
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(JsonFormatter())

    logger.addHandler(console)
    logger.addHandler(json_handler)
    return logger
