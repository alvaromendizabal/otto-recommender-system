"""User-executed checks only. Synthetic native fits are tests, not OTTO scores."""
import importlib.metadata,json,sys,unittest
for package in ('numpy','scipy','scikit-learn','lightgbm','polars','plotly','duckdb'):
 try:print(package,importlib.metadata.version(package),flush=True)
 except importlib.metadata.PackageNotFoundError:raise SystemExit('MISSING DEPENDENCY: '+package+'. Stop and return the log; do not install automatically.')
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
if not result.wasSuccessful():raise SystemExit(1)
from history_io import backend_smoke
print(backend_smoke())
with open('protocol.json', encoding='utf-8') as handle:
 p=json.load(handle)
print(f"ROUND{p['round']:02d}_TESTS_PASSED",flush=True)
