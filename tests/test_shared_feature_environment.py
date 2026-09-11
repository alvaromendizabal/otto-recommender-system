"""Independent guards for the frozen early-query environment preflight."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[1] / "scripts/preflight_shared_feature_environment.py"
SPEC = importlib.util.spec_from_file_location("otto_environment_preflight", SOURCE)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def fixture(role: str = "fit") -> tuple[Any, Any, Any, Any, str]:
    start = preflight.HISTORY_END if role == "fit" else preflight.FIT_END
    end = preflight.FIT_END if role == "fit" else preflight.SELECTION_END
    frozen = {"session": 1, "first_ts": start + 10, "query_ts": start + 20,
              "observed_events": 2, "observed_last_index": 1, "period_end": end}
    observed = [
        {"session": 1, "split_role": role, "event_index": 0, "event_type": 0,
         "aid": 9, "ts": start + 10},
        {"session": 1, "split_role": role, "event_index": 1, "event_type": 1,
         "aid": 9, "ts": start + 20},
    ]
    labels = [{"session": 1, "split_role": role, "objective": "orders", "aid": 9,
               "label_ts": start + 20, "label_event_index": 2}]
    queries = [{**frozen, "split_role": role, "objective": objective,
                "true_items": int(objective == "orders"),
                "recall_denominator": int(objective == "orders")}
               for objective in preflight.OBJECTIVES]
    return observed, labels, queries, frozen, role


class QueryAuditTests(unittest.TestCase):
    def test_valid_fit_and_selection(self) -> None:
        for role in ("fit", "selection"):
            with self.subTest(role=role):
                result = preflight.audit_session(*fixture(role))
                self.assertEqual(result["denominators"], [0, 0, 1])
                self.assertEqual(result["observed_events"], 2)

    def test_equal_timestamp_with_later_event_index_is_valid(self) -> None:
        result = preflight.audit_session(*fixture())
        self.assertEqual(result["truth"]["orders"], [9])

    def test_end_boundary_label_is_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        labels[0]["label_ts"] = frozen["period_end"]
        with self.assertRaisesRegex(ValueError, "timestamp"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_label_index_cannot_overlap_prefix(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        labels[0]["label_event_index"] = frozen["observed_last_index"]
        with self.assertRaisesRegex(ValueError, "after"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_target_before_query_is_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        labels[0]["label_ts"] = frozen["query_ts"] - 1
        with self.assertRaisesRegex(ValueError, "timestamp"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_missing_event_is_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        observed.pop()
        with self.assertRaisesRegex(ValueError, "count"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_noncontiguous_event_index_is_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        observed[1]["event_index"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_wrong_role_and_session_rejected(self) -> None:
        for key, value in (("split_role", "evaluation"), ("session", 99)):
            with self.subTest(field=key):
                observed, labels, queries, frozen, role = fixture()
                labels[0][key] = value
                with self.assertRaisesRegex(ValueError, "identity"):
                    preflight.audit_session(observed, labels, queries, frozen, role)

    def test_invalid_actions_and_items_rejected(self) -> None:
        for key, value in (("event_type", 8), ("aid", -1)):
            with self.subTest(field=key):
                observed, labels, queries, frozen, role = fixture()
                observed[0][key] = value
                with self.assertRaisesRegex(ValueError, "action"):
                    preflight.audit_session(observed, labels, queries, frozen, role)

    def test_nonmonotone_timestamps_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        observed[0]["ts"] = observed[1]["ts"] + 1
        with self.assertRaisesRegex(ValueError, "timestamps"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_frozen_query_metadata_rejected(self) -> None:
        for key in ("first_ts", "query_ts", "observed_last_index", "period_end"):
            with self.subTest(field=key):
                observed, labels, queries, frozen, role = fixture()
                frozen[key] += 1
                with self.assertRaisesRegex(ValueError, "Frozen"):
                    preflight.audit_session(observed, labels, queries, frozen, role)

    def test_duplicate_objective_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        queries[0] = copy.deepcopy(queries[1])
        with self.assertRaisesRegex(ValueError, "objective"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_duplicate_target_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        labels.append(copy.deepcopy(labels[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_two_next_click_targets_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        labels = [{**labels[0], "objective": "clicks", "aid": aid} for aid in (1, 2)]
        with self.assertRaisesRegex(ValueError, "Next-click"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_denominator_tampering_rejected(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        queries[2]["recall_denominator"] = 0
        with self.assertRaisesRegex(ValueError, "denominator"):
            preflight.audit_session(observed, labels, queries, frozen, role)

    def test_denominator_is_capped_at_twenty(self) -> None:
        observed, labels, queries, frozen, role = fixture()
        labels = [{**labels[0], "aid": aid, "label_event_index": 2 + aid} for aid in range(25)]
        queries[2].update(true_items=25, recall_denominator=20)
        result = preflight.audit_session(observed, labels, queries, frozen, role)
        self.assertEqual(result["denominators"], [0, 0, 20])

    def test_evaluation_role_is_rejected(self) -> None:
        observed, labels, queries, frozen, _ = fixture()
        with self.assertRaisesRegex(ValueError, "Only"):
            preflight.audit_session(observed, labels, queries, frozen, "evaluation")

    def test_input_data_not_mutated(self) -> None:
        inputs = fixture()
        before = copy.deepcopy(inputs)
        preflight.audit_session(*inputs)
        self.assertEqual(inputs, before)

    def test_empty_action_targets_are_preserved(self) -> None:
        observed, _, queries, frozen, role = fixture()
        for query in queries:
            query.update(true_items=0, recall_denominator=0)
        result = preflight.audit_session(observed, [], queries, frozen, role)
        self.assertEqual(result["denominators"], [0, 0, 0])

    def test_groups_have_no_lost_rows(self) -> None:
        rows = [{"session": 1}, {"session": 2}, {"session": 1}]
        self.assertEqual(preflight.group_rows(rows), {1: [rows[0], rows[2]], 2: [rows[1]]})

    def test_immutable_write_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "checkpoint.json"
            preflight.save_new(path, b"verified")
            preflight.save_new(path, b"verified")
            with self.assertRaisesRegex(ValueError, "preserved"):
                preflight.save_new(path, b"changed")
            self.assertEqual(path.read_bytes(), b"verified")

    def test_symlink_destination_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "link"
            path.symlink_to(Path(root) / "missing")
            with self.assertRaisesRegex(ValueError, "symlink"):
                preflight.save_new(path, b"data")


if __name__ == "__main__":
    unittest.main()
