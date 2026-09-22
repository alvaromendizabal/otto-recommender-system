"""Deterministic local artifact contracts extracted from the executed research runner."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

class GateError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def identity(obj) -> str:
    return hashlib.sha256(json_bytes(obj)).hexdigest()

def atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f'OUTPUT_SYMLINK: {path}')
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)

def save(path: Path, obj) -> None:
    atomic(path, json_bytes(obj))

def load(path: Path):
    return json.loads(Path(path).read_text())
