"""Run local contracts, then mandatory actual DuckDB/Parquet smoke in AWS."""
import argparse,unittest
p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');a=p.parse_args()
suite=unittest.defaultTestLoader.discover('tests')
r=unittest.TextTestRunner(verbosity=2).run(suite)
if not r.wasSuccessful():raise SystemExit(1)
if a.local:
 print('LOCAL_CONTRACTS_PASSED; private AWS/Parquet integration NOT executed')
else:
 from history_io import backend_smoke
 print(backend_smoke())
 print('ROUND10_TESTS_PASSED')
