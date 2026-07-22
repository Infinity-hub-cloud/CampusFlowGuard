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

from campus_flow_guard.inference import (
    build_detection_output,
    build_inference_windows,
    validate_required_columns,
)
from campus_flow_guard.risk import RiskMapping


class LocalInferenceTests(unittest.TestCase):
    def test_windows_target_the_last_input_row(self) -> None:
        values = np.arange(15, dtype=np.float32).reshape(5, 3)
        windows, targets = build_inference_windows(values, window_size=3)
        self.assertEqual(windows.shape, (3, 3, 3))
        np.testing.assert_array_equal(targets, np.array([2, 3, 4]))
        np.testing.assert_array_equal(windows[0, -1], values[2])

    def test_missing_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_required_columns(pd.DataFrame({"feature_a": [1]}), ["feature_a", "feature_b"])

    def test_detection_summary_and_risk_levels(self) -> None:
        mapping = RiskMapping(1, "Low", "Medium", "High", 0.75)
        document = build_detection_output(
            np.array([0.2, 0.8, 0.95]),
            np.array([7, 8, 9]),
            decision_threshold=0.659,
            risk_mapping=mapping,
            metadata={"input_row_count": 10},
        )
        self.assertEqual(
            [item["risk_level"] for item in document["detections"]],
            ["Low", "Medium", "High"],
        )
        summary = document["detection_summary"]
        self.assertEqual(summary["window_count"], 3)
        self.assertEqual(summary["detected_attack_count"], 2)
        self.assertEqual(summary["risk_level_counts"], {"Low": 1, "Medium": 1, "High": 1})


if __name__ == "__main__":
    unittest.main()
