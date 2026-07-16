import unittest

from aggregate_stopping_predictor_results import aggregate_method_success
from utils.kill_test_selection import (apply_selected_parameters, provisional_recommendation,
    select_validation_operating_points, tolerance_matched_comparisons)


class KillTestSelectionTests(unittest.TestCase):
    def test_selection_is_independent_per_tolerance_and_reuses_parameter(self):
        validation = [
            {"Accuracy": 90.0, "Average Timestep": 4.0, "Threshold": 0.8},
            {"Accuracy": 89.5, "Average Timestep": 3.0, "Threshold": 0.6},
            {"Accuracy": 88.0, "Average Timestep": 2.0, "Threshold": 0.4},
        ]
        selected = select_validation_operating_points(validation, 90.0, "Threshold")
        self.assertEqual([row["Threshold"] for row in selected], [0.8, 0.6, 0.6, 0.4])
        test = [{"Accuracy": 1.0, "Average Timestep": value, "Threshold": threshold}
                for value, threshold in [(8, 0.8), (6, 0.6), (4, 0.4)]]
        applied = apply_selected_parameters(selected, test, "Threshold")
        self.assertEqual([row["Threshold"] for row in applied], [0.8, 0.6, 0.6, 0.4])

    def test_comparisons_never_mix_tolerances(self):
        rows = [
            {"Accuracy Tolerance PP": 0.0, "Method": "Confidence", "Predictor": "baseline", "Feature Mode": "none", "Accuracy": 90, "Average Timestep": 5},
            {"Accuracy Tolerance PP": 2.0, "Method": "Confidence", "Predictor": "baseline", "Feature Mode": "none", "Accuracy": 88, "Average Timestep": 2},
            {"Accuracy Tolerance PP": 0.0, "Method": "multi_horizon__current_logits", "Predictor": "multi_horizon", "Feature Mode": "current_logits", "Accuracy": 90, "Average Timestep": 4, "Meets Validation Tolerance": True},
        ]
        comparison = tolerance_matched_comparisons(rows)[-1]
        self.assertEqual(comparison["Timestep Gain vs Confidence"], 1.0)

    def test_recommendation_requires_two_tolerances(self):
        rows = []
        for tolerance in (0.0, 0.5):
            rows.append({"Accuracy Tolerance PP": tolerance, "Predictor": "multi_horizon", "Feature Mode": "logit_history",
                         "Timestep Gain vs Final-Horizon Same Feature": 1.0, "Timestep Gain vs Confidence": 1.0,
                         "Timestep Gain vs Confidence Stability": 0.0, "Timestep Gain vs Same-Predictor Current Logits": 1.0,
                         "Meets Validation Tolerance": True})
        recommendation, _, _ = provisional_recommendation(rows)
        self.assertEqual(recommendation, "provisional_go")
        recommendation, _, _ = provisional_recommendation(rows[:1])
        self.assertEqual(recommendation, "provisional_weak_go")

    def test_recommendation_does_not_mix_feature_modes(self):
        rows = [
            {"Accuracy Tolerance PP": 0.0, "Predictor": "multi_horizon", "Feature Mode": "current_logits",
             "Timestep Gain vs Final-Horizon Same Feature": 1.0, "Timestep Gain vs Confidence": -1.0,
             "Timestep Gain vs Confidence Stability": -1.0, "Timestep Gain vs Same-Predictor Current Logits": 0.0,
             "Meets Validation Tolerance": True, "Meets Test Tolerance": True},
            {"Accuracy Tolerance PP": 0.0, "Predictor": "multi_horizon", "Feature Mode": "logit_history",
             "Timestep Gain vs Final-Horizon Same Feature": -1.0, "Timestep Gain vs Confidence": 1.0,
             "Timestep Gain vs Confidence Stability": 1.0, "Timestep Gain vs Same-Predictor Current Logits": 1.0,
             "Meets Validation Tolerance": True, "Meets Test Tolerance": True},
        ]
        recommendation, results, _ = provisional_recommendation(rows)
        self.assertFalse(results["0.0"]["provisional_success"])
        self.assertEqual(recommendation, "provisional_no_go")

    def test_aggregate_requires_positive_mean_gain(self):
        rows = [{"Timestep Gain vs Final-Horizon Same Feature": gain,
                 "Timestep Gain vs Confidence": gain,
                 "Timestep Gain vs Confidence Stability": gain,
                 "Meets Test Tolerance": "True"} for gain in (0.1, 0.1, -2.0)]
        self.assertFalse(aggregate_method_success(rows))


if __name__ == "__main__": unittest.main()
