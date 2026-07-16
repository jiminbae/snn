import unittest

import torch
import torch.nn.functional as F

from utils.temporal_reliability_loss import (all_prefix_cross_entropy, selective_regression_loss,
    combine_temporal_objective, symmetric_temporal_kl)


class TemporalReliabilityLossTests(unittest.TestCase):
    def test_final_ce_matches_existing_behavior(self):
        logits = torch.randn(3, 4, 2, requires_grad=True); targets = torch.tensor([0, 1, 0])
        existing_total = F.cross_entropy(logits[:, -1], targets)
        total, prefix_ce, temporal_loss, diagnostics = combine_temporal_objective(
            "final_ce", existing_total, existing_total, None, targets,
            prefix_loss_weight=0.0, temporal_loss_weight=1.0, margin=0.0,
            confidence_threshold=0.8, temperature=1.0, selection_mode="hard",
        )
        self.assertIs(total, existing_total)
        self.assertEqual(prefix_ce.item(), 0.0)
        self.assertEqual(temporal_loss.item(), 0.0)
        self.assertEqual(diagnostics["selected_transition_fraction"].item(), 0.0)

    def test_all_prefix_ce_has_finite_gradients(self):
        logits = torch.randn(2, 3, 2, requires_grad=True)
        loss = all_prefix_cross_entropy(logits, torch.tensor([0, 1])); loss.backward()
        self.assertEqual(loss.ndim, 0); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_symmetric_kl_is_zero_for_identical_logits(self):
        logits = torch.randn(2, 1, 3).expand(-1, 4, -1).clone()
        self.assertAlmostEqual(symmetric_temporal_kl(logits).item(), 0.0, places=6)

    def test_selective_regression_conditions(self):
        target = torch.tensor([0])
        decreasing = torch.tensor([[[5., 0.], [1., 4.]]], requires_grad=True)
        loss, diagnostics = selective_regression_loss(decreasing, target, 0.8, 0.0)
        self.assertGreater(loss.item(), 0); self.assertEqual(diagnostics["selected_transition_count"].item(), 1)
        loss.backward(); self.assertTrue(torch.isfinite(decreasing.grad).all()); self.assertNotEqual(decreasing.grad[:, 0].abs().sum(), 0); self.assertNotEqual(decreasing.grad[:, 1].abs().sum(), 0)
        increasing = torch.tensor([[[2., 0.], [5., 0.]]])
        self.assertEqual(selective_regression_loss(increasing, target, 0.8, 0.0)[0].item(), 0.0)

    def test_wrong_low_confidence_and_beneficial_transitions_are_unselected(self):
        target = torch.tensor([0])
        wrong_to_correct = torch.tensor([[[0., 5.], [5., 0.]]])
        loss, diagnostics = selective_regression_loss(wrong_to_correct, target, 0.0, 0.0)
        self.assertEqual(loss.item(), 0.0); self.assertEqual(diagnostics["selected_transition_count"].item(), 0)
        low_confidence = torch.tensor([[[0.1, 0.0], [0.0, 0.1]]])
        self.assertEqual(selective_regression_loss(low_confidence, target, 0.8, 0.0)[0].item(), 0.0)

    def test_no_selected_transition_is_finite(self):
        logits = torch.tensor([[[0., 1.], [0., 1.]]], requires_grad=True)
        loss, diagnostics = selective_regression_loss(logits, torch.tensor([0]), 0.8, 0.0)
        loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(diagnostics["violating_transition_fraction"].item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        logits = torch.randn(2, 3, 2, device="cuda", requires_grad=True)
        all_prefix_cross_entropy(logits, torch.tensor([0, 1], device="cuda")).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__": unittest.main()
