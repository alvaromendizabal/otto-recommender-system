from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
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
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event", record.getMessage()))
        details: list[str] = []
        for key in (
            "stage",
            "epoch",
            "step",
            "examples",
            "loss",
            "mrr",
            "hit10",
            "elapsed_seconds",
        ):
            if hasattr(record, key):
                details.append(f"{key}={getattr(record, key)}")
        suffix = f" | {' '.join(details)}" if details else ""
        return f"[{utc_now_iso()}] {record.levelname:<7} {event}{suffix}"


def configure_logging(run_name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(_ConsoleFormatter())
    file_handler = logging.FileHandler(log_dir / f"{run_name}.jsonl", encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
