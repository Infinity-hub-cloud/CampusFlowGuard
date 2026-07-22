from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


try:
    from fastapi.testclient import TestClient

    from campus_flow_guard.api import create_app
except ImportError:
    TestClient = None
    create_app = None


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class LocalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        config_path = root / "configs" / "api" / "local_demo.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "run": {"host": "127.0.0.1", "port": 8000},
                    "inference": {"config_path": "configs/inference/local_focal.yaml"},
                    "upload": {
                        "maximum_size_mib": 1,
                        "sample_prediction_limit": 1,
                        "allowed_suffixes": [".csv"],
                    },
                }
            ),
            encoding="utf-8",
        )

        def fake_inference(_config_path: Path, input_path: Path) -> dict:
            self.assertTrue(input_path.is_file())
            return {
                "metadata": {
                    "model": "Focal Loss Record Projection Transformer",
                    "model_seed": 20260717,
                },
                "detection_summary": {
                    "input_row_count": 8,
                    "window_count": 1,
                    "detected_attack_count": 1,
                    "detected_attack_ratio": 1.0,
                    "maximum_attack_probability": 0.8,
                    "mean_attack_probability": 0.8,
                    "decision_threshold": 0.7751080393791199,
                    "risk_level_counts": {"Low": 0, "Medium": 1, "High": 0},
                },
                "detections": [
                    {
                        "target_row_number": 7,
                        "attack_probability": 0.8,
                        "detected_attack": True,
                        "risk_level": "Medium",
                    }
                ],
            }

        self.client = TestClient(create_app(config_path, inference_runner=fake_inference))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_index_serves_upload_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("上传检测文件", response.text)
        self.assertIn("Local Processing", response.text)
        self.assertNotIn("https://", response.text)

    def test_local_static_assets_are_served(self) -> None:
        stylesheet = self.client.get("/static/styles.css")
        script = self.client.get("/static/app.js")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertIn("--background", stylesheet.text)
        self.assertIn("textContent", script.text)

    def test_predict_returns_required_summary(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("flows.csv", b"feature\n1\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_flows"], 8)
        self.assertEqual(data["attack_count"], 1)
        self.assertEqual(data["attack_ratio"], 1.0)
        self.assertEqual(data["maximum_attack_probability"], 0.8)
        self.assertEqual(data["mean_attack_probability"], 0.8)
        self.assertAlmostEqual(data["decision_threshold"], 0.7751080393791199)
        self.assertEqual(
            data["model_version"],
            "Focal Loss Record Projection Transformer · Seed 20260717",
        )
        self.assertEqual(data["risk_level_counts"]["Medium"], 1)
        self.assertEqual(len(data["sample_predictions"]), 1)

    def test_non_csv_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("flows.txt", b"not csv", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
