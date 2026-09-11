"""Guards for the 1,024 fitting-session shared-feature scale runner."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_shared_feature_scale.py"
SPEC = importlib.util.spec_from_file_location("shared_feature_scale", SOURCE)
assert SPEC is not None and SPEC.loader is not None
scale = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scale)


class ScaleHelpersTests(unittest.TestCase):
    def test_role_indices_require_exact_membership(self) -> None:
        queries = SimpleNamespace(session=np.array([1, 3, 5, 7], dtype=np.int64))
        np.testing.assert_array_equal(scale.role_indices(queries, np.array([1, 5])), [0, 2])
        with self.assertRaisesRegex(ValueError, "differs"):
            scale.role_indices(queries, np.array([1, 4]))

    def test_matrix_digest_is_content_bound(self) -> None:
        aids = np.array([1, 2], dtype=np.int64)
        target = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        first = scale.matrix_digest(aids, target, matrix)
        self.assertEqual(first, scale.matrix_digest(aids.copy(), target.copy(), matrix.copy()))
        matrix[0, 0] += 1
        self.assertNotEqual(first, scale.matrix_digest(aids, target, matrix))

    def test_family_labels_require_exact_disjoint_cover(self) -> None:
        actual = scale.family_labels(
            ("b",),
            ("a1", "a2"),
            {"funnel": ["a1"], "episode": ["a2"]},
        )
        self.assertEqual(actual, ("baseline", "funnel", "episode"))
        with self.assertRaisesRegex(ValueError, "multiple"):
            scale.family_labels(
                ("b",),
                ("a1", "a2"),
                {"funnel": ["a1"], "episode": ["a1", "a2"]},
            )
        with self.assertRaisesRegex(ValueError, "cover"):
            scale.family_labels(("b",), ("a1", "a2"), {"funnel": ["a1"]})


if __name__ == "__main__":
    unittest.main()
