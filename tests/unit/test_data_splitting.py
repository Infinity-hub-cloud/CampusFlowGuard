from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.data_protocol import (
    assert_split_integrity,
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
)


class StrictDataSplittingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.array([0] * 900 + [1] * 100, dtype=np.int8)

    def test_split_indices_are_mutually_exclusive_and_stratified(self) -> None:
        split = create_stratified_split(
            self.labels,
            pool_size=500,
            train_fraction=0.8,
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=17,
        )
        assert_split_integrity(split, expected_pool_size=500)
        self.assertEqual((len(split.train), len(split.validation), len(split.test)), (400, 50, 50))
        for indices in (split.train, split.validation, split.test):
            self.assertAlmostEqual(float(self.labels[indices].mean()), 0.1, places=2)

    def test_saved_indices_are_reused_exactly(self) -> None:
        split = create_stratified_split(
            self.labels,
            pool_size=500,
            train_fraction=0.8,
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            npz_path = root / "split.npz"
            json_path = root / "split.json"
            save_split_manifest(
                split,
                npz_path=npz_path,
                json_path=json_path,
                metadata={"pool_size": 500},
                labels=self.labels,
            )
            restored = load_split_manifest(npz_path, expected_pool_size=500)
        np.testing.assert_array_equal(restored.train, split.train)
        np.testing.assert_array_equal(restored.validation, split.validation)
        np.testing.assert_array_equal(restored.test, split.test)
