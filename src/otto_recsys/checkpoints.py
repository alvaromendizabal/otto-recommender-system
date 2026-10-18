"""Exclusive-writer checkpoint locking for the retained reference pipeline."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def workspace_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("ranking feature cache already has an active writer") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
