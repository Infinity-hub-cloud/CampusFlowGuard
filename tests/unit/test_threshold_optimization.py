from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.threshold_optimization import (
    POLICY_F1,
    POLICY_FAR,
    binary_metrics,
    evaluate_locked_thresholds,
    scan_thresholds,
    select_validation_thresholds,
)


class ValidationThresholdOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validation_labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)
        self.validation_probabilities = np.array([0.05, 0.20, 0.45, 0.80, 0.60, 0.90])

    def test_test_data_does_not_participate_in_selection(self) -> None:
        search = scan_thresholds(self.validation_labels, self.validation_probabilities, minimum=0.0, maximum=1.0, step=0.1)
        locked = select_validation_thresholds(search, far_limit=0.25)
        first_test = evaluate_locked_thresholds(np.array([0, 1], dtype=np.int8), np.array([0.1, 0.9]), locked)
        second_test = evaluate_locked_thresholds(np.array([0, 1], dtype=np.int8), np.array([0.9, 0.1]), locked)
        self.assertAlmostEqual(locked[POLICY_F1]["threshold"], 0.5)
        self.assertAlmostEqual(locked[POLICY_FAR]["threshold"], 0.5)
        self.assertNotEqual(first_test, second_test)
        self.assertEqual(locked, select_validation_thresholds(search, far_limit=0.25))

    def test_far_is_fp_divided_by_fp_plus_tn(self) -> None:
        result = binary_metrics(
            np.array([0, 0, 0, 1], dtype=np.int8),
            np.array([0.9, 0.8, 0.1, 0.9]),
            threshold=0.5,
        )
        self.assertEqual((result["FP"], result["TN"]), (2, 1))
        self.assertAlmostEqual(result["far"], 2 / 3)

    def test_identical_inputs_are_reproducible(self) -> None:
        first = scan_thresholds(self.validation_labels, self.validation_probabilities, minimum=0.0, maximum=1.0, step=0.1)
        second = scan_thresholds(self.validation_labels.copy(), self.validation_probabilities.copy(), minimum=0.0, maximum=1.0, step=0.1)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(select_validation_thresholds(first, far_limit=0.25), select_validation_thresholds(second, far_limit=0.25))


if __name__ == "__main__":
    unittest.main()
