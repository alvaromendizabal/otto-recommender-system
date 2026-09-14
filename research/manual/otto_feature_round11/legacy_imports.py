"""Load hash-verified legacy siblings by path, never from an arbitrary working directory."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_replication(directory, expected):
    """Import only the reviewed Round02 source; no data load or experiment execution.

    spec_from_file_location does not make a source file's directory importable.
    Explicit sibling loading avoids reliance on whichever round owns sys.path[0].
    An already-loaded sibling must have the expected source hash as well.
    """
    directory = Path(directory).resolve()
    for filename in ('core.py', 'intent_features.py', 'replication.py'):
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ImportError(f'Certified legacy file is absent or a symlink: {path}')
        if filename not in expected or _digest(path) != expected[filename]:
            raise ImportError(f'Certified legacy source hash differs: {filename}')
    loaded_here = []
    try:
        for name, filename in (('core', 'core.py'), ('intent_features', 'intent_features.py'),
                               ('round02_certified_helpers', 'replication.py')):
            existing = sys.modules.get(name)
            if existing is not None:
                origin = getattr(existing, '__file__', None)
                if not origin or not Path(origin).is_file() or _digest(Path(origin)) != expected[filename]:
                    raise ImportError(f'Conflicting loaded module {name}; use a clean worker process')
                module = existing
                continue
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot load certified sibling: {filename}')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loaded_here.append(name)
            spec.loader.exec_module(module)
        return sys.modules['round02_certified_helpers']
    except BaseException:
        for name in reversed(loaded_here):
            sys.modules.pop(name, None)
        raise
