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

from campus_flow_guard.train import _build_windows, _split_contiguous


class TrainDataContractTests(unittest.TestCase):
    def test_windows_end_at_their_target_and_do_not_cross_input_partition(self) -> None:
        values = np.arange(15, dtype=np.float32).reshape(5, 3)
        labels = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        windows, window_labels = _build_windows(values, labels, window_size=3)

        self.assertEqual(windows.shape, (3, 3, 3))
        np.testing.assert_array_equal(windows[0, :, 0], np.array([0, 3, 6], dtype=np.float32))
        np.testing.assert_array_equal(window_labels, np.array([1, 1, 1], dtype=np.int32))

    def test_contiguous_split_preserves_row_order(self) -> None:
        frame = pd.DataFrame({"row": range(20)})
        train, validation, test = _split_contiguous(frame, 0.7, 0.15, 0.15)

        self.assertEqual(train["row"].tolist(), list(range(14)))
        self.assertEqual(validation["row"].tolist(), [14, 15, 16])
        self.assertEqual(test["row"].tolist(), [17, 18, 19])


if __name__ == "__main__":
    unittest.main()
