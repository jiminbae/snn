import unittest

import torch

from utils.stopping_features import build_causal_features, fit_feature_normalization, normalize_features


class StoppingFeatureTests(unittest.TestCase):
    def test_future_changes_do_not_affect_past_features(self):
        logits = torch.randn(2, 4, 3)
        changed = logits.clone(); changed[:, 2:] += 100
        first = build_causal_features(logits, "logit_history")
        second = build_causal_features(changed, "logit_history")
        self.assertTrue(torch.equal(first[:, :2], second[:, :2]))

    def test_padding_mask_switches_persistence_and_t1(self):
        logits = torch.tensor([[[5., 0.], [4., 0.], [0., 5.]]])
        features = build_causal_features(logits, "logit_history")
        tmax, classes = 3, 2
        mask_start = tmax * classes
        self.assertEqual(features[0, 0, mask_start:mask_start+tmax].tolist(), [1, 0, 0])
        self.assertTrue(torch.isfinite(features).all())
        self.assertFalse(torch.equal(features[:, 1], features[:, 2]))

    def test_normalization_uses_supplied_train_statistics_and_zero_std(self):
        train = torch.ones(2, 3, 4)
        stats = fit_feature_normalization(train)
        self.assertTrue(torch.equal(stats["feature_std"], torch.ones(4)))
        test = torch.full((1, 3, 4), 3.0)
        self.assertTrue(torch.equal(normalize_features(test, stats), torch.full_like(test, 2.0)))

    def test_feature_modes_have_distinct_dimensions_and_train_only_statistics(self):
        train = torch.randn(2, 3, 2)
        current = build_causal_features(train, "current_logits")
        history = build_causal_features(train, "logit_history")
        self.assertNotEqual(current.shape[-1], history.shape[-1])
        current_stats = fit_feature_normalization(current)
        history_stats = fit_feature_normalization(history)
        changed_validation = build_causal_features(torch.full((2, 3, 2), 1000.0), "current_logits")
        self.assertTrue(torch.equal(current_stats["feature_mean"], fit_feature_normalization(current)["feature_mean"]))
        self.assertFalse(torch.equal(current_stats["feature_mean"], fit_feature_normalization(changed_validation)["feature_mean"]))
        self.assertNotEqual(current_stats["feature_mean"].numel(), history_stats["feature_mean"].numel())


if __name__ == "__main__": unittest.main()
