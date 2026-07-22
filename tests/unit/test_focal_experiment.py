from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from campus_flow_guard.baseline import _build_record_projection_model
from campus_flow_guard.focal_experiment import compile_with_focal_loss


class FocalLossExperimentTests(unittest.TestCase):
    def test_gamma_zero_matches_binary_crossentropy(self) -> None:
        labels = np.array([[0.0], [1.0], [1.0]], dtype=np.float32)
        probabilities = np.array([[0.1], [0.8], [0.6]], dtype=np.float32)
        bce = keras.losses.BinaryCrossentropy()(labels, probabilities)
        focal = keras.losses.BinaryFocalCrossentropy(
            gamma=0.0,
            apply_class_balancing=False,
        )(labels, probabilities)
        self.assertAlmostEqual(float(bce), float(focal), places=6)

    def test_recompile_changes_loss_without_changing_structure(self) -> None:
        model_config = {
            "internal_size": 8,
            "num_heads": 2,
            "categorical_embedding_dim": 2,
            "transformer_layers": 2,
            "dropout": 0.1,
            "mlp_units": 4,
        }
        model = _build_record_projection_model(
            window_size=3,
            numeric_feature_count=2,
            vocabulary_sizes=[5],
            model_config=model_config,
            learning_rate=0.001,
        )
        architecture = model.get_config()
        parameter_count = model.count_params()
        compile_with_focal_loss(
            model,
            learning_rate=0.001,
            gamma=2.0,
            apply_class_balancing=False,
            alpha=0.25,
            from_logits=False,
        )
        self.assertEqual(model.get_config(), architecture)
        self.assertEqual(model.count_params(), parameter_count)
        self.assertIsInstance(model.loss, keras.losses.BinaryFocalCrossentropy)


if __name__ == "__main__":
    unittest.main()
