from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.inspect_dataset import inspect_dataset


class InspectDatasetTests(unittest.TestCase):
    def test_real_csv_inspection_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-02"],
                "category": ["web", "dns", "dns", "dns"],
                "label": ["Benign", "Attack", "Attack", "Attack"],
                "value": [1.0, float("inf"), 2.0, 2.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            frame.to_csv(csv_path, index=False)
            report = inspect_dataset(
                csv_path,
                label_column="label",
                time_column="timestamp",
            )

        self.assertEqual(report["row_count_loaded"], 4)
        self.assertEqual(report["column_count"], 4)
        self.assertEqual(report["duplicate_row_count"], 1)
        self.assertEqual(report["infinite_value_total"], 1)
        self.assertEqual(len(report["file_sha256"]), 64)
        self.assertEqual(report["constant_column_count"], 0)
        self.assertEqual(report["label_distribution"][0]["count"], 3)
        self.assertEqual(report["category_counts"]["category"]["unique_count_in_loaded_rows"], 2)
        self.assertTrue(report["time_column_exists"])


if __name__ == "__main__":
    unittest.main()
