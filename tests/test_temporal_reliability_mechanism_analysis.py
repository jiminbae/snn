import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from analyze_temporal_reliability_mechanism import (
    MIN_BIN_SUPPORT,
    mechanism_recommendation,
    paired_bootstrap_intervals,
    paired_state_rows,
    seed_metrics,
    timing_row,
)
from export_confirmatory_prefix_trajectories import (
    build_trajectory,
    validate_alignment,
    validate_trajectory,
    source_fingerprints,
)
from run_temporal_reliability_mechanism_analysis import parse_gpus, trajectory_valid


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


def recommendation_row(
    seed,
    regression=-1.0,
    eligible=-1.0,
    drop=-0.1,
    drop_fraction=-0.1,
    recovery=1.0,
    support=100,
    selective_recovery_count=None,
    comparator_recovery_count=None,
):
    comparator_recovery_count = support if comparator_recovery_count is None else comparator_recovery_count
    selective_recovery_count = (
        round(recovery * comparator_recovery_count)
        if selective_recovery_count is None
        else selective_recovery_count
    )
    return {
        "seed": seed,
        "common_correct_support": support,
        "common_wrong_support": support,
        "eligible_common_correct_support": support,
        "valid_eligible_transition_count": 1,
        "micro_matched_regression_difference": regression,
        "eligible_micro_regression_difference": eligible,
        "macro_mean_drop_magnitude_difference": drop,
        "probability_drop_fraction_difference": drop_fraction,
        "recovery_preservation_ratio": (
            selective_recovery_count / comparator_recovery_count
            if comparator_recovery_count else float("nan")
        ),
        "selective_matched_recovery_count": selective_recovery_count,
        "comparator_matched_recovery_count": comparator_recovery_count,
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

    def test_pooled_recovery_ratio_blocks_unrepresentative_seed_average(self):
        rows = [
            recommendation_row(3, support=1000, selective_recovery_count=800, comparator_recovery_count=1000),
            recommendation_row(4, selective_recovery_count=10, comparator_recovery_count=10),
            recommendation_row(5, selective_recovery_count=10, comparator_recovery_count=10),
        ]
        self.assertGreaterEqual(
            sum(row["recovery_preservation_ratio"] for row in rows) / len(rows), 0.9
        )
        self.assertTrue(all(row["recovery_preservation_ratio"] >= 0.8 for row in rows))
        self.assertNotEqual(mechanism_recommendation(rows)[0], "mechanism_supported")

    def test_low_eligibility_support_is_insufficient(self):
        rows = [recommendation_row(seed, support=99) for seed in (3, 4, 5)]
        self.assertEqual(mechanism_recommendation(rows)[0], "insufficient_support")

    def test_uninterpretable_eligibility_is_excluded_from_seed_micro(self):
        selective = trajectory([[1, 1], [1, 1]], method="selective_regression_thr0.6")
        final = trajectory([[1, 0], [1, 1]])
        row = seed_metrics(paired_state_rows(selective, final, "final_ce"))
        self.assertEqual(row["eligible_common_correct_total_support"], 2)
        self.assertEqual(row["eligible_common_correct_support"], 0)
        self.assertEqual(row["valid_eligible_transition_count"], 0)
        self.assertTrue(torch.isnan(torch.tensor(row["eligible_micro_regression_difference"])))

    def test_paired_bootstrap_produces_nonempty_intervals(self):
        selective = trajectory([[1, 1]] * 100, method="selective_regression_thr0.6")
        final = trajectory([[1, 0]] * 100)
        rows = paired_bootstrap_intervals(selective, final, "final_ce", iterations=10)
        self.assertEqual(len(rows), 5)
        regression = next(row for row in rows if row["metric"] == "micro_matched_regression_difference")
        self.assertEqual(regression["bootstrap_iterations"], 10)
        self.assertEqual(regression["point_estimate"], -100.0)
        self.assertEqual(regression["resampling_unit"], "paired_sample_id")

    def test_bootstrap_macro_drop_matches_recommendation_metric(self):
        selective = trajectory(
            [[1, 1, 1], [1, 0, 0]],
            [[0.9, 0.8, 0.7], [0.9, 0.1, 0.1]],
            "selective_regression_thr0.6",
        )
        final = trajectory(
            [[1, 1, 1], [1, 0, 0]],
            [[0.9, 0.5, 0.4], [0.9, 0.1, 0.1]],
        )
        metrics = seed_metrics(paired_state_rows(selective, final, "final_ce"))
        intervals = paired_bootstrap_intervals(selective, final, "final_ce", iterations=10)
        macro = next(
            row for row in intervals if row["metric"] == "macro_mean_drop_magnitude_difference"
        )
        micro = next(
            row for row in intervals if row["metric"] == "micro_mean_drop_magnitude_difference"
        )
        self.assertAlmostEqual(
            macro["point_estimate"], metrics["macro_mean_drop_magnitude_difference"]
        )
        self.assertAlmostEqual(
            micro["point_estimate"], metrics["micro_mean_drop_magnitude_difference"]
        )
        self.assertNotEqual(macro["point_estimate"], micro["point_estimate"])

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
        with self.assertRaisesRegex(ValueError, "final correct count"):
            validate_trajectory(
                item, expected_samples=2, expected_timesteps=2,
                expected_classes=2, expected_final_accuracy=100.0,
                expected_prefix_curve=[100.0, 50.0],
            )

    def test_prefix_count_drift_within_tolerance_passes(self):
        logits = torch.full((10000, 2, 2), -2.0)
        logits[..., 0] = 2.0
        logits[:3, 0, 0] = -2.0
        logits[:3, 0, 1] = 2.0
        item = build_trajectory(
            logits, torch.zeros(10000, dtype=torch.long), method="final_ce", seed=3,
            checkpoint_path="checkpoint.pt", config={},
        )
        validation = validate_trajectory(
            item, expected_samples=10000, expected_timesteps=2,
            expected_classes=2, expected_final_accuracy=100.0,
            expected_prefix_curve=[100.0, 100.0],
        )
        self.assertEqual(validation["prefix_correct_count_drift"], [-3, 0])
        self.assertEqual(validation["max_abs_prefix_correct_count_drift"], 3)
        self.assertAlmostEqual(validation["max_abs_prefix_curve_drift_pp"], 0.03)
        self.assertTrue(validation["final_correct_count_exact"])

    def test_prefix_count_drift_above_tolerance_fails(self):
        logits = torch.full((10000, 2, 2), -2.0)
        logits[..., 0] = 2.0
        logits[:6, 0, 0] = -2.0
        logits[:6, 0, 1] = 2.0
        item = build_trajectory(
            logits, torch.zeros(10000, dtype=torch.long), method="final_ce", seed=3,
            checkpoint_path="checkpoint.pt", config={},
        )
        with self.assertRaisesRegex(ValueError, "correct-count drift exceeds tolerance"):
            validate_trajectory(
                item, expected_samples=10000, expected_timesteps=2,
                expected_classes=2, expected_final_accuracy=100.0,
                expected_prefix_curve=[100.0, 100.0],
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
        self.assertEqual(row["selective_matched_recovery_count"], 0)
        self.assertEqual(row["comparator_matched_recovery_count"], 0)

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

    def test_never_correct_is_separate_from_delay(self):
        selective = trajectory([
            [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1],
        ], method="selective_regression_thr0.6")
        final = trajectory([
            [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1],
        ])
        row = timing_row(selective, final, "final_ce")
        self.assertEqual(row["first_correct_selective_only_never_count"], 1)
        self.assertEqual(row["first_correct_comparator_only_never_count"], 1)
        self.assertEqual(row["first_correct_both_never_count"], 1)
        self.assertEqual(row["first_correct_both_valid_count"], 1)
        self.assertEqual(row["first_correct_delay_mean"], 1.0)
        self.assertEqual(row["first_correct_both_valid_differences"], [1.0])

    def test_first_correct_delay_detected(self):
        selective = trajectory([[0, 1, 1]], method="selective_regression_thr0.6")
        final = trajectory([[1, 1, 1]])
        row = timing_row(selective, final, "final_ce")
        self.assertEqual(row["first_correct_delay_mean"], 1.0)

    def test_parallel_gpus_are_split_from_environment(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            self.assertEqual(parse_gpus(None), ["0", "1", "2", "3"])

    def test_trajectory_cache_requires_current_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "final_ce" / "seed_3"
            run_dir.mkdir(parents=True)
            config = {"dataset": "nmnist", "model": "fixed_lif", "tmax": 2, "seed": 3, "batch_size": 2}
            summary = {"final_accuracy": 100.0}
            prefix_metrics = {"num_samples": 2, "prefix_accuracy_curve": [100.0, 100.0]}
            (run_dir / "config.json").write_text(json.dumps(config))
            (run_dir / "temporal_reliability_summary.json").write_text(json.dumps(summary))
            (run_dir / "prefix_metrics.json").write_text(json.dumps(prefix_metrics))
            (run_dir / "best_checkpoint.pt").write_bytes(b"checkpoint")
            logits = torch.zeros(2, 2, 10)
            logits[..., 0] = 1.0
            cached = build_trajectory(
                logits, torch.zeros(2, dtype=torch.long), method="final_ce", seed=3,
                checkpoint_path=str(run_dir / "best_checkpoint.pt"), config=config,
                fingerprints=source_fingerprints(run_dir),
                export_settings={"batch_size": 2, "cudnn_benchmark": True},
            )
            cached["validation"] = validate_trajectory(
                cached, expected_samples=2, expected_timesteps=2,
                expected_classes=10, expected_final_accuracy=100.0,
                expected_prefix_curve=[100.0, 100.0],
            )
            cache_path = Path(tmp) / "trajectory.pt"
            torch.save(cached, cache_path)
            self.assertTrue(trajectory_valid(cache_path, run_dir, "final_ce", 3))
            config["tmax"] = 3
            (run_dir / "config.json").write_text(json.dumps(config))
            self.assertFalse(trajectory_valid(cache_path, run_dir, "final_ce", 3))


if __name__ == "__main__":
    unittest.main()
