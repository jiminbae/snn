from __future__ import annotations

import unittest

import torch

from utils.stopping_analysis import (
    best_policy_by_accuracy_tolerance,
    confidence_stopping,
    cost_aware_oracle,
    earliest_correct_oracle,
    earliest_stable_correct_oracle,
    evaluate_stopping_policy,
    global_pareto_frontier,
    pareto_frontier,
    timestep_confidence_outcome_rows,
    trajectory_outcomes,
    validate_trajectory_payload,
)


class StoppingAnalysisTest(unittest.TestCase):
    def test_earliest_correct_oracle_and_never_correct_fallback(self) -> None:
        correct = torch.tensor([[False, True, False], [False, False, False]])
        self.assertEqual(earliest_correct_oracle(correct).tolist(), [2, 3])

    def test_earliest_stable_correct_oracle(self) -> None:
        correct = torch.tensor([
            [False, True, True],
            [True, False, True],
            [False, False, False],
        ])
        self.assertEqual(earliest_stable_correct_oracle(correct).tolist(), [2, 3, 3])

    def test_cost_aware_oracle_changes_with_lambda(self) -> None:
        correct = torch.tensor([[False, False, True]])
        self.assertEqual(cost_aware_oracle(correct, 0.0).item(), 3)
        self.assertEqual(cost_aware_oracle(correct, 3.0).item(), 1)

    def test_cost_aware_oracle_breaks_cost_ties_early(self) -> None:
        correct = torch.tensor([[False, False, True]])
        self.assertEqual(cost_aware_oracle(correct, 1.5).item(), 1)

    def test_confidence_stopping_and_final_fallback(self) -> None:
        confidence = torch.tensor([[0.2, 0.8, 0.9], [0.1, 0.2, 0.3]])
        self.assertEqual(confidence_stopping(confidence, 0.7).tolist(), [2, 3])

    def test_trajectory_outcomes_cover_four_types(self) -> None:
        correct = torch.tensor([
            [True, True, True],
            [False, True, True],
            [True, False, False],
            [False, False, False],
        ])
        outcomes = trajectory_outcomes(correct)
        self.assertTrue(outcomes["safe_stop"][0, 0])
        self.assertTrue(outcomes["beneficial_continuation"][1, 0])
        self.assertTrue(outcomes["destructive_continuation"][2, 0])
        self.assertTrue(outcomes["futile_continuation"][3, 0])
        partition = sum(mask.to(torch.int64) for mask in outcomes.values())
        self.assertTrue(torch.equal(partition, torch.ones_like(partition)))

    def test_pareto_frontier_removes_dominated_points_per_family(self) -> None:
        rows = [
            {"Policy": "Confidence", "Accuracy": 80.0, "Average Timestep": 2.0},
            {"Policy": "Confidence", "Accuracy": 80.0, "Average Timestep": 3.0},
            {"Policy": "Confidence", "Accuracy": 70.0, "Average Timestep": 1.0},
            {"Policy": "Entropy", "Accuracy": 60.0, "Average Timestep": 4.0},
        ]
        frontier = pareto_frontier(rows)
        self.assertEqual(len(frontier), 3)
        self.assertFalse(any(row["Policy"] == "Confidence" and row["Average Timestep"] == 3.0 for row in frontier))

    def test_global_pareto_removes_points_dominated_by_other_families(self) -> None:
        rows = [
            {"Policy": "Confidence", "Accuracy": 80.0, "Average Timestep": 3.0},
            {"Policy": "Entropy", "Accuracy": 80.0, "Average Timestep": 2.0},
        ]
        frontier = global_pareto_frontier(rows)
        self.assertEqual(frontier, [rows[1]])

    def test_accuracy_tolerance_selects_fastest_eligible_policy(self) -> None:
        rows = [
            {"Policy": "Confidence", "Accuracy": 96.4, "Average Timestep": 1.5},
            {"Policy": "Confidence", "Accuracy": 96.5, "Average Timestep": 2.0},
            {"Policy": "Confidence", "Accuracy": 97.0, "Average Timestep": 3.0},
        ]
        selected = best_policy_by_accuracy_tolerance(rows, final_accuracy=97.0, tolerances_pp=[0.5])
        self.assertIs(selected["0.5"], rows[1])

    def test_timestep_confidence_outcomes_respect_bin_boundaries(self) -> None:
        confidence = torch.tensor([[0.50, 1.00], [0.499, 0.99]])
        correct = torch.tensor([[True, True], [False, False]])
        rows = timestep_confidence_outcome_rows(confidence, trajectory_outcomes(correct))
        for timestep in (1, 2):
            selected = [row for row in rows if row["Timestep"] == timestep]
            self.assertEqual(sum(row["Number of Samples"] for row in selected), 2)
            for row in selected:
                if row["Number of Samples"]:
                    self.assertAlmostEqual(sum(row[name] for name in (
                        "safe_stop", "beneficial_continuation", "destructive_continuation", "futile_continuation"
                    )), 100.0)
        last_bin_t2 = [row for row in rows if row["Timestep"] == 2 and row["Lower Bound"] == 0.99][0]
        self.assertEqual(last_bin_t2["Number of Samples"], 2)

    def test_policy_evaluation_rejects_out_of_range_stops(self) -> None:
        with self.assertRaisesRegex(ValueError, "within"):
            evaluate_stopping_policy(
                torch.zeros((1, 2), dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                torch.tensor([0]),
                policy="invalid",
            )

    def _valid_payload(self) -> dict[str, torch.Tensor]:
        return {
            "prefix_logits": torch.zeros((2, 3, 2)),
            "targets": torch.zeros(2, dtype=torch.long),
            "predictions": torch.zeros((2, 3), dtype=torch.long),
            "confidence": torch.ones((2, 3)),
            "entropy": torch.zeros((2, 3)),
            "margin": torch.ones((2, 3)),
            "correct": torch.ones((2, 3), dtype=torch.bool),
        }

    def test_trajectory_payload_shape_and_finite_validation(self) -> None:
        with self.assertRaises(ValueError):
            validate_trajectory_payload([])  # type: ignore[arg-type]
        invalid_payloads = []
        wrong_predictions = self._valid_payload()
        wrong_predictions["predictions"] = torch.zeros((2, 2), dtype=torch.long)
        invalid_payloads.append(wrong_predictions)
        wrong_targets = self._valid_payload()
        wrong_targets["targets"] = torch.zeros(1, dtype=torch.long)
        invalid_payloads.append(wrong_targets)
        nan_confidence = self._valid_payload()
        nan_confidence["confidence"][0, 0] = torch.nan
        invalid_payloads.append(nan_confidence)
        empty_n = self._valid_payload()
        empty_n["prefix_logits"] = torch.zeros((0, 3, 2))
        invalid_payloads.append(empty_n)
        empty_t = self._valid_payload()
        empty_t["prefix_logits"] = torch.zeros((2, 0, 2))
        invalid_payloads.append(empty_t)
        for payload in invalid_payloads:
            with self.subTest(shape=tuple(payload["prefix_logits"].shape)):
                with self.assertRaises(ValueError):
                    validate_trajectory_payload(payload)


if __name__ == "__main__":
    unittest.main()
