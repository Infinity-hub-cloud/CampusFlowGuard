from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.risk import (
    RiskMapping,
    assess_risk,
    extract_decision_threshold,
    load_validation_threshold,
)


class RiskMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = RiskMapping(
            version=1,
            low_label="Low",
            medium_label="Medium",
            high_label="High",
            high_cutoff_fraction=0.75,
        )

    def test_low_medium_high_and_boundary_inclusion(self) -> None:
        threshold = {"threshold": 0.6}
        self.assertEqual(assess_risk(0.59, threshold, self.mapping).risk_level, "Low")
        self.assertEqual(assess_risk(0.60, threshold, self.mapping).risk_level, "Medium")
        self.assertEqual(assess_risk(0.89, threshold, self.mapping).risk_level, "Medium")
        high = assess_risk(0.90, threshold, self.mapping)
        self.assertEqual(high.risk_level, "High")
        self.assertAlmostEqual(high.high_cutoff, 0.9)

    def test_invalid_probability_and_threshold_are_rejected(self) -> None:
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(probability=value), self.assertRaises(ValueError):
                assess_risk(value, 0.5, self.mapping)
        with self.assertRaises(ValueError):
            extract_decision_threshold({"not_threshold": 0.5})

    def test_only_validation_threshold_documents_are_accepted(self) -> None:
        document = {
            "selection_scope": "validation_only",
            "test_used_for_threshold_selection": False,
            "seed_results": [
                {
                    "seed": 17,
                    "validation_selection": {"validation_f1_max": {"threshold": 0.7}},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "thresholds.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            result = load_validation_threshold(path, seed=17, policy="validation_f1_max")
            self.assertEqual(result["threshold"], 0.7)
            document["test_used_for_threshold_selection"] = True
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_validation_threshold(path, seed=17, policy="validation_f1_max")


if __name__ == "__main__":
    unittest.main()
