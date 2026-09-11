# Shared-feature environment and early-prefix preflight

The owner authorizes direct, bounded execution in the existing `otto-dev` space and GitHub. This stage follows completed workspace synchronization; do not repeat old collectors or retrain saved models.

## Question and scope

Can the existing project environment import the canonical data reader under the recorded core/ML package pins, and does it reproduce the frozen early-window prefixes and timestamped target sets for 256 fitting and 256 selection sessions?

`preflight_shared_feature_environment.py` inspects the selected interpreter without installing anything. Its real-data mode requires Python 3.13 and exact recorded core/ML versions. It verifies recovered query-file hashes, composes hard-linked reader inputs outside Git, recomputes canonical cohort ordering, and checks full prefix/target ledgers independently with PyArrow against `dataset.Queries`. It writes six small complete Parquet partitions and demands byte-identical replay. Evaluation sessions are excluded from the selected audit rows.

## Availability and leakage contract

Early history ends at 2022-08-16 22:00 UTC. Fitting ends August 19 22:00 UTC and selection ends August 20 22:00 UTC. Session roles use the first event; period ends are exclusive. A future label can share a timestamp with the final observed event only when its event index is strictly later. Click targets contain at most one next item; carts/orders are distinct-item sets. Denominators retain every saved target, capped at 20 per session/action, without candidate-based removal.

This is not a raw-event replay, model comparison, new feature screen, or complete 102/134 feature construction. It does not certify the additional historical graphs. The frozen 32-feature hypothesis and the open feature-engineering gate remain unchanged.

## Execution gates

Execute through a one-time, bounded driver in the existing space only after the unchanged repository CI passes. Preserve non-clean worktrees rather than resetting. Inspect `.venv` before considering installation. The outer driver limits useful work to 240 seconds, emits UTC heartbeats every 15 seconds, and saves result/error records to the existing private S3 area. Stop the app after retrieving the result, then remove only the temporary lifecycle configuration. No IAM, storage, global environment, or model changes are required.

A missing interpreter, version mismatch, invalid source hash, corpus/cohort discrepancy, or inconsistent replay is a stop for diagnosis, not permission to retry unchanged. Existing successful partitions are immutable and reused. The reader input directory deliberately lacks the large history table and is not a complete corpus suitable for graph rebuilding.

## Tests and result publication

The new tests cover temporal boundaries, tied timestamps, wrong roles/session IDs, missing/reordered events, corrupted frozen metadata, duplicate targets/objectives, next-click cardinality, capped denominators, input nonmutation, symlink rejection, and immutable file replay. CI remains authoritative for Ruff, mypy and all project tests. Record actual runtime, memory, target counts, source hashes, S3 receipts, GitHub status and shutdown after execution. No competition score may be inferred from a passing preflight.
