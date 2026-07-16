import unittest

import torch

from utils.stopping_targets import action_margin_weights, build_stopping_targets, oracle_future_choice


class StoppingTargetTests(unittest.TestCase):
    def setUp(self):
        predictions = torch.tensor([[1, 0, 0], [0, 0, 1], [1, 1, 1]])
        self.targets = torch.tensor([0, 0, 0])
        self.logits = torch.nn.functional.one_hot(predictions, 2).float() * 4

    def test_targets_and_masks(self):
        result = build_stopping_targets(self.logits, self.targets)
        self.assertEqual(result["recoverable_final"].tolist(), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(result["final_horizon_outcome"].tolist(), [[1, 0, 0], [2, 2, 0], [0, 0, 0]])
        self.assertEqual(result["next_error"].tolist(), [[0, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.assertFalse(result["continue_mask"][:, -1].any())
        self.assertFalse(result["next_target_mask"][:, -1].any())
        mask = result["future_horizon_mask"]
        self.assertTrue(mask[0].equal(torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=torch.bool)))

    def test_oracle_ties_early_and_lambda_changes_choice(self):
        error = torch.tensor([[1.0, 0.0, 0.0]])
        choices, actions = oracle_future_choice(error, 0.0)
        self.assertEqual(choices[0, 0].item(), 1)
        self.assertTrue(actions[0, 0])
        expensive, _ = oracle_future_choice(error, 4.0)
        self.assertEqual(expensive[0, 0].item(), 0)

    def test_action_weights_are_bounded_and_final_zero(self):
        weights = action_margin_weights(torch.tensor([[1.0, 0.0, 1.0]]), 0.5, 0.25)
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())
        self.assertEqual(weights[0, -1].item(), 0.0)


if __name__ == "__main__": unittest.main()
