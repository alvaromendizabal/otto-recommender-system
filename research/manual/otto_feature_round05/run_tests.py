"""Run local contracts and the mandatory installed DuckDB/Parquet smoke, no installs."""
import json
from pathlib import Path
import unittest
from backend_smoke import run
ROOT=Path(__file__).resolve().parent
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
if not result.wasSuccessful():raise SystemExit(1)
print(json.dumps(run()),flush=True)
print('ROUND05_TESTS_PASSED',flush=True)
