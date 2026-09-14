# Legacy import repair

The verified Round02 intent_features module is explicitly loaded before replication.py. No feature formula, parameter, candidate, model, data, or source-commit requirement changes. The unclosed test configuration file is also closed properly.

The installer preserves the failed notebook and receipts. Only the diagnosed pre-data import failure may be archived after the lightweight import check succeeds. Rerun the notebook tests before a data stage. The original execution Git checkout remains pinned.

New tests are prepared, not executed by the assistant.
