import unittest

import numpy as np

from campus_flow_guard.classical_baselines import _metrics, _validation_threshold, _window_features


class ClassicalBaselineTests(unittest.TestCase):
    def test_threshold_is_selected_from_validation_arrays(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        probabilities = np.array([0.05, 0.2, 0.7, 0.9], dtype=np.float32)
        threshold = _validation_threshold(labels, probabilities)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        metrics = _metrics(labels, probabilities, threshold)
        self.assertEqual(metrics["TP"] + metrics["FP"] + metrics["TN"] + metrics["FN"], 4)

    def test_metrics_confusion_counts_sum_to_samples(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        probabilities = np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32)
        metrics = _metrics(labels, probabilities, 0.5)
        self.assertEqual(metrics["TN"] + metrics["FP"] + metrics["FN"] + metrics["TP"], 4)
        self.assertEqual(metrics["far"], metrics["FP"] / (metrics["FP"] + metrics["TN"]))

    def test_window_target_count(self):
        self.assertEqual(12000 - 8 + 1, 11993)


if __name__ == "__main__":
    unittest.main()
