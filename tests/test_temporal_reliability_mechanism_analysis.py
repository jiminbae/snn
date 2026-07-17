import unittest

import torch

from analyze_temporal_reliability_mechanism import (
    MIN_BIN_SUPPORT,
    mechanism_recommendation,
    paired_state_rows,
    seed_metrics,
    timing_row,
)
from export_confirmatory_prefix_trajectories import (
    build_trajectory,
    validate_alignment,
    validate_trajectory,
)


def trajectory(correct, probability=None, method="final_ce", seed=3, targets=None, indices=None):
    correct = torch.tensor(correct, dtype=torch.bool)
    samples, timesteps = correct.shape
    targets = torch.zeros(samples, dtype=torch.long) if targets is None else torch.tensor(targets)
    logits = torch.full((samples, timesteps, 2), -2.0)
    logits[..., 0] = torch.where(correct, torch.tensor(2.0), torch.tensor(-2.0))
    logits[..., 1] = torch.where(correct, torch.tensor(-2.0), torch.tensor(2.0))
    result = build_trajectory(
        logits, targets, method=method, seed=seed,
        checkpoint_path="checkpoint.pt", config={},
    )
    if probability is not None:
        result["true_class_probability"] = torch.tensor(probability, dtype=torch.float32)
    if indices is not None:
        result["sample_index"] = torch.tensor(indices)
    return result


def recommendation_row(seed, regression=-1.0, eligible=-1.0, drop=-0.1, drop_fraction=-0.1, recovery=1.0, support=100):
    return {
        "seed": seed,
        "common_correct_support": support,
        "common_wrong_support": support,
        "micro_matched_regression_difference": regression,
        "eligible_micro_regression_difference": eligible,
        "mean_drop_magnitude_difference": drop,
        "probability_drop_fraction_difference": drop_fraction,
        "recovery_preservation_ratio": recovery,
    }


class MechanismAnalysisTests(unittest.TestCase):
    def test_lower_matched_regression_supports_mechanism(self):
        rows = [recommendation_row(seed) for seed in (3, 4, 5)]
        self.assertEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_opportunity_only_does_not_support(self):
        rows = [recommendation_row(seed, regression=0.0) for seed in (3, 4, 5)]
        self.assertNotEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_worse_common_correct_is_not_supported(self):
        rows = [recommendation_row(seed, regression=1.0) for seed in (3, 4, 5)]
        self.assertEqual(mechanism_recommendation(rows)[0], "mechanism_not_supported")

    def test_regression_down_and_recovery_preserved_can_support(self):
        rows = [recommendation_row(seed, recovery=0.95) for seed in (3, 4, 5)]
        self.assertEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_low_recovery_blocks_support(self):
        rows = [recommendation_row(seed, recovery=0.79) for seed in (3, 4, 5)]
        self.assertNotEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_probability_drop_magnitude_reduced(self):
        selective = trajectory([[1, 1], [1, 1]], [[0.9, 0.88], [0.8, 0.79]], "selective_regression_thr0.6")
        final = trajectory([[1, 1], [1, 1]], [[0.9, 0.5], [0.8, 0.4]])
        row = paired_state_rows(selective, final, "final_ce")["probability"][0]
        self.assertLess(row["mean_drop_magnitude_difference"], 0)

    def test_eligibility_matched_regression_reduced(self):
        selective = trajectory([[1, 1], [1, 1]], [[0.8, 0.8], [0.7, 0.7]], "selective_regression_thr0.6")
        final = trajectory([[1, 0], [1, 0]], [[0.8, 0.4], [0.7, 0.4]])
        row = paired_state_rows(selective, final, "final_ce")["confidence"][0]
        self.assertLess(row["paired_difference"], 0)

    def test_small_confidence_bin_is_not_interpretable(self):
        selective = trajectory([[1, 1]], [[0.7, 0.7]], "selective_regression_thr0.6")
        final = trajectory([[1, 1]], [[0.7, 0.7]])
        bins = paired_state_rows(selective, final, "final_ce")["confidence"]
        fixed = [row for row in bins if row["bin"].startswith("[")]
        self.assertTrue(all(not row["interpretable"] for row in fixed))
        self.assertEqual(MIN_BIN_SUPPORT, 100)

    def test_target_mismatch_raises(self):
        a = trajectory([[1, 1]], method="selective_regression_thr0.6")
        b = trajectory([[1, 1]], targets=[1])
        with self.assertRaisesRegex(ValueError, "targets"):
            validate_alignment([a, b])

    def test_sample_order_mismatch_raises(self):
        a = trajectory([[1, 1], [1, 1]], method="selective_regression_thr0.6")
        b = trajectory([[1, 1], [1, 1]], indices=[1, 0])
        with self.assertRaisesRegex(ValueError, "sample_index"):
            validate_alignment([a, b])

    def test_prefix_shape_mismatch_raises(self):
        a = trajectory([[1, 1]], method="selective_regression_thr0.6")
        b = trajectory([[1, 1, 1]])
        with self.assertRaisesRegex(ValueError, "prefix shape"):
            validate_alignment([a, b])

    def test_final_accuracy_mismatch_raises(self):
        item = trajectory([[1, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "final accuracy"):
            validate_trajectory(
                item, expected_samples=2, expected_timesteps=2,
                expected_classes=2, expected_final_accuracy=100.0,
                expected_prefix_curve=[100.0, 50.0],
            )

    def test_zero_common_correct_does_not_crash(self):
        a = trajectory([[0, 0]], method="selective_regression_thr0.6")
        b = trajectory([[0, 0]])
        rows = paired_state_rows(a, b, "final_ce")
        self.assertEqual(rows["regression"][0]["common_correct_count"], 0)

    def test_zero_common_wrong_does_not_crash(self):
        a = trajectory([[1, 1]], method="selective_regression_thr0.6")
        b = trajectory([[1, 1]])
        rows = paired_state_rows(a, b, "final_ce")
        self.assertEqual(rows["recovery"][0]["common_wrong_count"], 0)

    def test_seed_metrics_are_computed_separately(self):
        a = trajectory([[1, 1], [1, 1]], method="selective_regression_thr0.6", seed=3)
        b = trajectory([[1, 0], [1, 1]], seed=3)
        row = seed_metrics(paired_state_rows(a, b, "final_ce"))
        self.assertEqual(row["seed"], 3)
        self.assertEqual(row["common_correct_support"], 2)

    def test_one_seed_improvement_is_not_enough(self):
        rows = [
            recommendation_row(3, regression=-1, eligible=-1),
            recommendation_row(4, regression=1, eligible=1),
            recommendation_row(5, regression=1, eligible=1),
        ]
        self.assertNotEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_two_of_three_direction_passes(self):
        rows = [
            recommendation_row(3, regression=-2, eligible=-2),
            recommendation_row(4, regression=-2, eligible=-2),
            recommendation_row(5, regression=1, eligible=1),
        ]
        self.assertEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_first_correct_delay_detected(self):
        selective = trajectory([[0, 1, 1]], method="selective_regression_thr0.6")
        final = trajectory([[1, 1, 1]])
        row = timing_row(selective, final, "final_ce")
        self.assertEqual(row["first_correct_delay_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()

