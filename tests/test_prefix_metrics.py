from __future__ import annotations

import unittest

import torch

from utils.prefix_metrics import (
    consecutive_regression_rate,
    ever_regressed_rate,
    first_correct_timestep,
    mean_negative_temporal_gain,
    negative_temporal_gain,
    prefix_accuracy_auc,
    prefix_accuracy_curve,
    stable_correct_timestep,
    worst_prefix_accuracy,
)


class PrefixMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = torch.tensor([0, 1])
        self.logits = torch.tensor([
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
            [[0.0, 3.0], [0.0, 3.0], [3.0, 0.0]],
        ])

    def test_accuracy_and_regression_metrics(self) -> None:
        curve = prefix_accuracy_curve(self.logits, self.targets)
        self.assertTrue(torch.equal(curve, torch.tensor([100.0, 50.0, 50.0])))
        regression = consecutive_regression_rate(self.logits, self.targets)
        self.assertTrue(torch.equal(regression["population_per_transition"], torch.tensor([50.0, 50.0])))
        self.assertTrue(torch.equal(regression["conditional_per_transition"], torch.tensor([50.0, 100.0])))
        self.assertEqual(float(regression["mean_population"]), 50.0)
        self.assertEqual(float(regression["mean_conditional"]), 75.0)
        self.assertTrue(torch.equal(regression["per_transition"], regression["population_per_transition"]))
        self.assertEqual(float(ever_regressed_rate(self.logits, self.targets)), 100.0)

    def test_curve_summaries(self) -> None:
        curve = torch.tensor([80.0, 70.0, 75.0, 65.0])
        self.assertEqual(float(negative_temporal_gain(curve)), 20.0)
        self.assertAlmostEqual(float(mean_negative_temporal_gain(curve)), 20.0 / 3.0, places=5)
        self.assertEqual(float(worst_prefix_accuracy(curve)), 65.0)
        self.assertEqual(float(prefix_accuracy_auc(curve)), 72.5)

    def test_first_and_stable_correct_timesteps(self) -> None:
        self.assertEqual(first_correct_timestep(self.logits, self.targets).tolist(), [1, 1])
        self.assertEqual(stable_correct_timestep(self.logits, self.targets).tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
