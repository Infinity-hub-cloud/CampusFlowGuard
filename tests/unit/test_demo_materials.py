from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.data_protocol import sha256_file
from scripts.prepare_demo_data import _read_mapping, compose_demo_frames


DEMO_DIR = PROJECT_ROOT / "artifacts" / "demo"
DEMO_NAMES = ("benign_demo.csv", "attack_demo.csv", "mixed_demo.csv")


class DemoMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "datasets" / "dataset_schema.yaml").read_text(
                encoding="utf-8"
            )
        )
        cls.manifest = json.loads(
            (DEMO_DIR / "demo_manifest.json").read_text(encoding="utf-8")
        )

    def test_demo_csvs_have_exact_feature_schema_and_no_sensitive_fields(self) -> None:
        expected = self.schema["feature_columns"]
        forbidden = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack", "Label"}
        for name in DEMO_NAMES:
            with self.subTest(name=name):
                frame = pd.read_csv(DEMO_DIR / name)
                self.assertEqual(frame.columns.tolist(), expected)
                self.assertTrue(forbidden.isdisjoint(frame.columns))
                self.assertEqual(len(frame), 16)
                self.assertEqual(int(frame.isna().sum().sum()), 0)
                self.assertFalse(np.isinf(frame.to_numpy(dtype=np.float64)).any())

    def test_manifest_sources_are_validation_only_and_hashes_match(self) -> None:
        split_path = (
            PROJECT_ROOT
            / "artifacts"
            / "baseline_freeze"
            / "unsw_trusted_baseline_v1"
            / "splits"
            / "split_indices.npz"
        )
        with np.load(split_path, allow_pickle=False) as loaded:
            validation = set(loaded["validation"].astype(np.int64).tolist())
            test = set(loaded["test"].astype(np.int64).tolist())
        self.assertEqual(self.manifest["source"]["partition"], "validation")
        self.assertFalse(self.manifest["selection"]["test_rows_used"])
        for name in DEMO_NAMES:
            details = self.manifest["files"][name]
            indices = set(details["source_indices"])
            self.assertTrue(indices.issubset(validation))
            self.assertTrue(indices.isdisjoint(test))
            self.assertEqual(details["sha256"], sha256_file(DEMO_DIR / name))

    def test_composition_is_deterministic(self) -> None:
        features = ["feature_a", "feature_b"]
        labels = ["Benign"] * 20 + ["Attack"] * 20
        frame = pd.DataFrame(
            {
                "feature_a": np.arange(40),
                "feature_b": np.arange(40) * 2,
                "Attack": labels,
            },
            index=np.arange(100, 140),
        ).sample(frac=1.0, random_state=7)
        first_frames, first_indices = compose_demo_frames(
            frame,
            feature_columns=features,
            label_column="Attack",
            benign_label="Benign",
        )
        second_frames, second_indices = compose_demo_frames(
            frame,
            feature_columns=features,
            label_column="Attack",
            benign_label="Benign",
        )
        self.assertEqual(first_indices, second_indices)
        for name in DEMO_NAMES:
            pd.testing.assert_frame_equal(first_frames[name], second_frames[name])

    def test_json_mapping_reader_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text('{"frozen": true}\n', encoding="utf-8-sig")
            self.assertEqual(_read_mapping(path), {"frozen": True})


if __name__ == "__main__":
    unittest.main()
