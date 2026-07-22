from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.data_protocol import build_partition_window_indices


class StrictWindowGenerationTests(unittest.TestCase):
    def test_window_target_is_last_row(self) -> None:
        partition = np.array([14, 2, 9, 22, 30], dtype=np.int64)
        windows, targets = build_partition_window_indices(partition, window_size=3)
        np.testing.assert_array_equal(windows, np.array([[2, 9, 14], [9, 14, 22], [14, 22, 30]]))
        np.testing.assert_array_equal(targets, windows[:, -1])

    def test_windows_never_include_another_partition(self) -> None:
        train = np.array([0, 3, 6, 9, 12], dtype=np.int64)
        validation = np.array([1, 4, 7, 10, 13], dtype=np.int64)
        test = np.array([2, 5, 8, 11, 14], dtype=np.int64)
        split_sets = [set(values.tolist()) for values in (train, validation, test)]
        for position, partition in enumerate((train, validation, test)):
            windows, _ = build_partition_window_indices(partition, window_size=3)
            self.assertTrue(all(set(window.tolist()) <= split_sets[position] for window in windows))
            other_indices = set.union(*(split_sets[index] for index in range(3) if index != position))
            self.assertTrue(all(set(window.tolist()).isdisjoint(other_indices) for window in windows))
