import unittest

import torch

from utils.stopping_policy_evaluation import binary_metrics, binary_ranking_metrics


class StoppingPolicyEvaluationTests(unittest.TestCase):
    def test_ranking_accepts_raw_scores_without_calibration_fields(self):
        scores = torch.tensor([-4.0, 2.0, -1.0, 8.0])
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        metrics = binary_ranking_metrics(scores, targets)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertNotIn("brier_score", metrics)
        self.assertNotIn("ece", metrics)
        calibrated = binary_metrics(scores.sigmoid(), targets)
        self.assertIn("brier_score", calibrated)
        self.assertIn("ece", calibrated)

    def test_auroc_is_invariant_to_monotonic_score_transform(self):
        scores = torch.tensor([-2.0, 0.5, 3.0, 1.0])
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
        first = binary_ranking_metrics(scores, targets)["auroc"]
        second = binary_ranking_metrics(scores * 7 + 11, targets)["auroc"]
        self.assertEqual(first, second)

    def test_auroc_ties_receive_half_credit(self):
        metrics = binary_ranking_metrics(torch.zeros(4), torch.tensor([0.0, 1.0, 0.0, 1.0]))
        self.assertEqual(metrics["auroc"], 0.5)

    def test_ranking_metrics_scale_without_pairwise_matrix(self):
        scores = torch.arange(100_000, dtype=torch.float32) % 100
        targets = torch.arange(100_000).remainder(2).float()
        metrics = binary_ranking_metrics(scores, targets)
        self.assertTrue(metrics["valid"])
        self.assertTrue(0.0 <= metrics["auroc"] <= 1.0)


if __name__ == "__main__": unittest.main()
