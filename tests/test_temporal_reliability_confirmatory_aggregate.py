import unittest

from aggregate_temporal_reliability_confirmatory import (
    COMMON_PROTOCOL,
    build_matched_records,
    recommendation_from_records,
    validate_confirmatory_config,
)


def record(
    method,
    seed,
    regression=10.0,
    beneficial=10.0,
    recovered=10.0,
    accuracy=70.0,
    destructive=10.0,
):
    return {
        "method": method,
        "seed": seed,
        "final_accuracy": accuracy,
        "mean_prefix_accuracy": 65.0,
        "minimum_prefix_accuracy": 55.0,
        "ever_regressed_fraction": regression,
        "mean_population_regression": regression,
        "mean_conditional_regression": regression,
        "correct_to_wrong_transition_count": regression,
        "destructive_transition_fraction": destructive,
        "ever_recovered_fraction": recovered,
        "wrong_to_correct_transition_count": beneficial,
        "beneficial_transition_fraction": beneficial,
        "prefix_accuracy_curve": [60.0] * 8,
    }


def rows(selective_regressions=(6.0, 6.0, 6.0), **selective):
    result = []
    for seed, regression in zip((3, 4, 5), selective_regressions):
        result.extend(
            [
                record("final_ce", seed),
                record(
                    "symmetric_kl",
                    seed,
                    regression=8.0,
                    beneficial=8.0,
                    recovered=8.0,
                    destructive=8.0,
                ),
                record(
                    "selective_regression_thr0.6",
                    seed,
                    regression=regression,
                    beneficial=selective.get("beneficial", 9.5),
                    recovered=selective.get("recovered", 9.5),
                    accuracy=selective.get("accuracy", 70.0),
                    destructive=selective.get("destructive", 6.0),
                ),
            ]
        )
    return result


def valid_config(method="final_ce"):
    config = dict(COMMON_PROTOCOL)
    config.update(seed=3, temporal_training_mode=method)
    if method == "selective_regression":
        config["temporal_confidence_threshold"] = 0.6
    return config


class ConfirmatoryRecommendationTests(unittest.TestCase):
    def test_all_three_seed_reductions_are_go(self):
        self.assertEqual(recommendation_from_records(rows())[0], "go")

    def test_two_seed_reductions_are_go(self):
        self.assertEqual(
            recommendation_from_records(rows((6.0, 6.0, 11.0)))[0],
            "go",
        )

    def test_preservation_between_point_eight_and_point_nine_is_weak_go(self):
        self.assertEqual(
            recommendation_from_records(rows(beneficial=8.5, recovered=8.5))[0],
            "weak_go",
        )

    def test_low_beneficial_preservation_is_no_go(self):
        self.assertEqual(
            recommendation_from_records(rows(beneficial=7.9))[0],
            "no_go",
        )

    def test_low_recovered_preservation_is_no_go(self):
        self.assertEqual(
            recommendation_from_records(rows(recovered=7.9))[0],
            "no_go",
        )

    def test_accuracy_drop_over_one_pp_is_no_go(self):
        self.assertEqual(
            recommendation_from_records(rows(accuracy=68.9))[0],
            "no_go",
        )

    def test_one_reduced_seed_is_no_go(self):
        self.assertEqual(
            recommendation_from_records(rows((6.0, 11.0, 11.0)))[0],
            "no_go",
        )

    def test_worse_destructive_than_symmetric_is_not_go(self):
        self.assertNotEqual(
            recommendation_from_records(rows(destructive=8.1))[0],
            "go",
        )

    def test_worse_beneficial_than_symmetric_is_not_go(self):
        self.assertNotEqual(
            recommendation_from_records(rows(beneficial=7.9))[0],
            "go",
        )

    def test_two_matched_seeds_are_no_go(self):
        partial = [row for row in rows() if row["seed"] in (3, 4)]
        self.assertEqual(recommendation_from_records(partial)[0], "no_go")

    def test_threshold_point_eight_does_not_substitute(self):
        wrong = [
            {**row, "method": "selective_regression_thr0.8"}
            if row["method"] == "selective_regression_thr0.6"
            else row
            for row in rows()
        ]
        self.assertEqual(recommendation_from_records(wrong)[0], "no_go")

    def test_zero_matches_does_not_crash(self):
        decision, reasons = recommendation_from_records([])
        self.assertEqual(decision, "no_go")
        self.assertEqual(
            reasons,
            ["No complete matched confirmatory seeds were found."],
        )

    def test_duplicate_method_seed_raises(self):
        duplicate = rows() + [record("final_ce", 3)]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_matched_records(duplicate)


class ConfirmatoryProtocolTests(unittest.TestCase):
    def test_dvs_gesture_config_raises(self):
        config = valid_config()
        config["dataset"] = "dvs_gesture"
        with self.assertRaisesRegex(ValueError, "dataset.*nmnist.*dvs_gesture"):
            validate_confirmatory_config("run", config, "final_ce")

    def test_wrong_split_seed_raises(self):
        config = valid_config()
        config["split_seed"] = 42
        with self.assertRaisesRegex(ValueError, "split_seed.*123.*42"):
            validate_confirmatory_config("run", config, "final_ce")

    def test_wrong_selective_threshold_raises(self):
        config = valid_config("selective_regression")
        config["temporal_confidence_threshold"] = 0.7
        with self.assertRaisesRegex(ValueError, "threshold.*0.6.*0.7"):
            validate_confirmatory_config(
                "run",
                config,
                "selective_regression_thr0.6",
            )

    def test_wrong_temporal_weight_raises(self):
        config = valid_config()
        config["temporal_loss_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "temporal_loss_weight.*1.0.*0.5"):
            validate_confirmatory_config("run", config, "final_ce")


if __name__ == "__main__":
    unittest.main()

