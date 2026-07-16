import unittest

from aggregate_temporal_reliability_results import recommendation_from_records


def record(method, seed, regression, beneficial, accuracy=70.0, prefix=65.0, destructive=None):
    return {"method": method, "seed": seed, "ever_regressed_fraction": regression,
            "beneficial_transition_fraction": beneficial, "final_accuracy": accuracy,
            "mean_prefix_accuracy": prefix, "destructive_transition_fraction": regression if destructive is None else destructive,
            "minimum_prefix_accuracy": prefix, "mean_population_regression": regression,
            "mean_conditional_regression": regression, "correct_to_wrong_transition_count": regression,
            "ever_recovered_fraction": beneficial, "wrong_to_correct_transition_count": beneficial,
            "stable_correct_fraction": accuracy}


class TemporalReliabilityAggregateTests(unittest.TestCase):
    def test_regression_reduction_with_preserved_recovery_is_go(self):
        rows = []
        for seed in range(3):
            rows += [record("final_ce", seed, 10, 10), record("symmetric_kl", seed, 8, 9),
                     record("selective_regression_thr0.8", seed, 6, 9.5, prefix=66)]
        self.assertEqual(recommendation_from_records(rows)[0], "go")

    def test_recovery_collapse_or_accuracy_drop_is_not_go(self):
        for beneficial, accuracy in ((7.0, 70.0), (10.0, 68.9)):
            rows = []
            for seed in range(3):
                rows += [record("final_ce", seed, 10, 10),
                         record("selective_regression_thr0.8", seed, 5, beneficial, accuracy=accuracy)]
            self.assertNotEqual(recommendation_from_records(rows)[0], "go")

    def test_one_improved_seed_is_no_go(self):
        rows = []
        for seed in range(3):
            rows += [record("final_ce", seed, 10, 10),
                     record("selective_regression_thr0.8", seed, 5 if seed == 0 else 12, 10)]
        self.assertEqual(recommendation_from_records(rows)[0], "no_go")


if __name__ == "__main__": unittest.main()
