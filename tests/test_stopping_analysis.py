from __future__ import annotations

import unittest

import torch

from utils.stopping_analysis import (
    confidence_stopping,
    cost_aware_oracle,
    earliest_correct_oracle,
    earliest_stable_correct_oracle,
    pareto_frontier,
    trajectory_outcomes,
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


if __name__ == "__main__":
    unittest.main()
