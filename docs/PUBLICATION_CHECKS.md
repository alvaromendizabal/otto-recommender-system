# Public snapshot checks

This repository preserves historical source bytes and their recorded checksums.
The local publisher records an empty-end-of-file notice only when both the exact
archived path and whole-file SHA-256 match the table below. It does not rewrite
historical source or pretend its hashes were produced by a reformatted copy.

All other whitespace errors, conflict markers, unexpected diagnostics, credential
findings and byte mismatches remain blocking. Git configuration, hooks and GitHub
checks are not disabled. This is not evidence of a new model result.

| Frozen source path | SHA-256 |
|---|---|
| `research/manual/otto_feature_round04/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round05/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round06/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round08/source_contract.py` | `2cd4d6a82e6cd71381bb083ba92eaef881e8b61c68536448821121766a18a8a0` |
| `research/manual/otto_feature_round08/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round09/source_contract.py` | `2cd4d6a82e6cd71381bb083ba92eaef881e8b61c68536448821121766a18a8a0` |
| `research/manual/otto_feature_round09/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round10/source_contract.py` | `2cd4d6a82e6cd71381bb083ba92eaef881e8b61c68536448821121766a18a8a0` |
| `research/manual/otto_feature_round10/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round11/source_contract.py` | `cbf7de4a4c8c3f7cc7b375562fe634a692ce72aff558fee59735dba8da5d0343` |
| `research/manual/otto_feature_round11/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
| `research/manual/otto_feature_round12/source_contract.py` | `cbf7de4a4c8c3f7cc7b375562fe634a692ce72aff558fee59735dba8da5d0343` |
| `research/manual/otto_feature_round12/tests/test_metric_contracts.py` | `9a254d8b14787a8ad19482b936bf5e196995f50ab7def6ce94827ed12983ab91` |
