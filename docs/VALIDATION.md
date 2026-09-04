# OTTO local validation

The project reproduces OTTO's published Kaggle local-test semantics.

1. Use a time-based cutoff.
2. Trim training sessions to events strictly before the cutoff.
3. Require at least two retained events for training sessions.
4. Build the known-item universe from that leakage-safe training history.
5. Select sessions beginning after the cutoff as validation candidates.
6. Remove validation events for items unseen in training.
7. Require at least two remaining validation events.
8. Deterministically retain a random observed prefix.
9. Use the next future click as the click target.
10. Use unique future cart/order items as their respective targets.

The default benchmark uses two holdout days and seed 42.
