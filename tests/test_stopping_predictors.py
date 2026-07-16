import unittest

import torch

from utils.stopping_predictors import (StoppingMLP, masked_bce_with_logits, multi_horizon_stops,
    one_step_stops, predictor_output_dim, recoverability_stops)
from utils.kill_test_selection import planned_configurations


class StoppingPredictorTests(unittest.TestCase):
    def test_predictor_feature_mode_cross_product(self):
        configurations = planned_configurations(["one_step", "multi_horizon"], ["current_logits", "logit_history"])
        self.assertEqual([item[2] for item in configurations], [
            "one_step__current_logits", "one_step__logit_history",
            "multi_horizon__current_logits", "multi_horizon__logit_history"])
        self.assertEqual(len(planned_configurations(["one_step"], ["current_logits"])), 1)
        with self.assertRaises(ValueError):
            planned_configurations(["one_step"], ["current_logits", "current_logits"])

    def test_forward_shapes(self):
        features = torch.randn(2, 4, 7)
        for name in ("recoverability_final", "final_horizon_gain", "one_step", "multi_horizon"):
            model = StoppingMLP(7, predictor_output_dim(name, 4), hidden_dim=8, dropout=0)
            self.assertEqual(model(features).shape, (2, 4, predictor_output_dim(name, 4)))

    def test_masked_loss_ignores_invalid_horizons(self):
        logits = torch.zeros(1, 2, 2)
        targets = torch.tensor([[[0., 1.], [1., 1.]]])
        mask = torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool)
        loss = masked_bce_with_logits(logits, targets, mask)
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)

    def test_sequential_decisions_and_final_stop(self):
        one = torch.tensor([[[0.9, 0.1], [0.2, 0.1], [0.1, 0.1]]])
        self.assertEqual(one_step_stops(one, 0.0).item(), 3)
        multi = torch.tensor([[[0.9, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.0, 0.1]]])
        self.assertEqual(multi_horizon_stops(multi, 0.0).item(), 2)
        recover = torch.ones(1, 3)
        self.assertEqual(recoverability_stops(recover, 0.5).item(), 3)
        for stops in (one_step_stops(one, 0), multi_horizon_stops(multi, 0)):
            self.assertTrue(((stops >= 1) & (stops <= 3)).all())


if __name__ == "__main__": unittest.main()
