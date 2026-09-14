"""Run fast synthetic tests, then mandatory installed DuckDB/Parquet smoke."""
from pathlib import Path
import unittest
import sys
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():sys.exit(1)
    from history_io import backend_smoke
    print(backend_smoke(),flush=True)
    print('ROUND07_TESTS_PASSED',flush=True)
