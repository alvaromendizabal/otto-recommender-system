"""Contract tests and native synthetic LightGBM reload smoke; never install packages."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
if not result.wasSuccessful():raise SystemExit(1)
print(f'ROUND06_TESTS_PASSED tests={result.testsRun}',flush=True)
