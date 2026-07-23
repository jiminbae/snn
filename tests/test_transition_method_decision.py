import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from export_confirmatory_prefix_trajectories import build_trajectory
from utils.trajectory_export import export_trajectory_payload
from models import FixedLIFSNN
from utils.transition_method_decision import (
    DecisionGuardrails,
    FilterSpec,
    HiddenRCCED,
    candidate_guardrail_pass,
    direction_metrics,
    direction_selector_rollout,
    first_regression_timestep,
    hidden_rc_ced_loss,
    recommend_branch,
    regression_event_comparison,
    regression_survival_rows,
    simple_filter_rollout,
    split_payload_to_trajectory,
    threshold_at_false_veto_budget,
    validate_hidden_trajectory,
    validate_split_trajectories,
)


def make(probabilities, seed=3, sample_index=None, split=None):
    probability = torch.tensor(probabilities, dtype=torch.float32)
    if probability.ndim == 1:
        probability = probability.unsqueeze(0)
    logits = torch.stack(
        [torch.logit(probability.clamp(1e-5, 1 - 1e-5)), torch.zeros_like(probability)],
        dim=-1,
    )
    trajectory = build_trajectory(
        logits,
        torch.zeros(logits.shape[0], dtype=torch.long),
        method="final_ce",
        seed=seed,
        checkpoint_path="x",
        config={},
    )
    if sample_index is not None:
        trajectory["sample_index"] = torch.tensor(sample_index, dtype=torch.long)
    if split is not None:
        trajectory["split"] = split
    return trajectory


def good_rows(**overrides):
    rows = []
    for seed in (3, 4, 5):
        row = {
            "seed": seed,
            "micro_matched_regression_difference": -1.0,
            "candidate_matched_recovery_count": 96,
            "raw_matched_recovery_count": 100,
            "seed_recovery_preservation_ratio": 0.96,
            "final_accuracy_change": 0.0,
            "mean_prefix_accuracy_change": 0.0,
            "minimum_prefix_accuracy_change": 0.0,
            "first_delay_mean": 0.0,
            "ever_regressed_fraction_change": -0.01,
            "raw_matched_regression_count": 100,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def guardrails():
    return DecisionGuardrails(
        0.5,
        minimum_pooled_raw_regression_count=100,
        minimum_pooled_raw_recovery_count=100,
    )


class HiddenPrefixModel(nn.Module):
    def forward(self, images, **kwargs):
        first = torch.cat([images, -images], dim=1)
        second = torch.cat([images + 1, -images], dim=1)
        logits = torch.stack([first, second], dim=1)
        hidden = torch.stack([images, images + 1], dim=1)
        return {
            "prefix_logits": logits,
            "logits": logits[:, -1],
            "temporal_features": hidden,
        }


class TransitionMethodDecisionTests(unittest.TestCase):
    def test_arithmetic_alpha_one_matches_raw_probabilities(self):
        raw = make([[0.8, 0.2, 0.9]])
        candidate, _ = simple_filter_rollout(
            raw, FilterSpec("arithmetic_probability_ema", (("alpha", 1.0),))
        )
        self.assertTrue(
            torch.allclose(
                candidate["prefix_logits"].softmax(-1),
                raw["prefix_logits"].softmax(-1),
                atol=1e-6,
            )
        )

    def test_geometric_alpha_one_matches_raw_probabilities(self):
        raw = make([[0.8, 0.2, 0.9]])
        candidate, _ = simple_filter_rollout(
            raw, FilterSpec("geometric_probability_ema", (("alpha", 1.0),))
        )
        self.assertTrue(
            torch.allclose(
                candidate["prefix_logits"].softmax(-1),
                raw["prefix_logits"].softmax(-1),
                atol=1e-6,
            )
        )

    def test_clipped_infinite_innovation_telescopes_to_raw(self):
        raw = make([[0.5, 0.98, 0.98]])
        candidate, _ = simple_filter_rollout(
            raw, FilterSpec("clipped_logit_innovation", (("clip", math.inf),))
        )
        self.assertTrue(
            torch.allclose(
                candidate["prefix_logits"].softmax(-1),
                raw["prefix_logits"].softmax(-1),
                atol=1e-6,
            )
        )

    def test_clipped_innovation_uses_raw_previous(self):
        logits = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [4.0, 0.0]]])
        raw = build_trajectory(
            logits, torch.zeros(1, dtype=torch.long),
            method="final_ce", seed=3, checkpoint_path="x", config={}
        )
        candidate, _ = simple_filter_rollout(
            raw, FilterSpec("clipped_logit_innovation", (("clip", 1.0),))
        )
        centered = candidate["prefix_logits"][0, :, 0] - candidate["prefix_logits"][0, :, 1]
        self.assertTrue(torch.allclose(centered, torch.tensor([0.0, 2.0, 2.0])))

    def test_filters_are_causal(self):
        first = make([[0.8, 0.2, 0.9]])
        second = make([[0.8, 0.2, 0.01]])
        spec = FilterSpec("geometric_probability_ema", (("alpha", 0.5),))
        first_filtered, _ = simple_filter_rollout(first, spec)
        second_filtered, _ = simple_filter_rollout(second, spec)
        self.assertTrue(
            torch.allclose(
                first_filtered["prefix_logits"][:, :2],
                second_filtered["prefix_logits"][:, :2],
            )
        )

    def test_low_fpr_threshold_handles_ties_conservatively(self):
        scores = torch.tensor([0.90, 0.90, 0.10, 0.85])
        targets = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
        threshold = threshold_at_false_veto_budget(scores, targets, 1 / 3)
        metrics = direction_metrics(scores, targets, threshold)
        self.assertGreater(threshold, 0.90)
        self.assertEqual(metrics["W_TO_C_false_veto_rate"], 0.0)

    def test_low_fpr_threshold_maximizes_recall_under_budget(self):
        scores = torch.tensor([0.90, 0.85, 0.70, 0.10])
        targets = torch.tensor([0, 1, 0, 0], dtype=torch.bool)
        threshold = threshold_at_false_veto_budget(scores, targets, 1 / 3)
        self.assertAlmostEqual(threshold, 0.85)
        metrics = direction_metrics(scores, targets, threshold)
        self.assertAlmostEqual(metrics["C_TO_W_recall"], 1.0)
        self.assertAlmostEqual(metrics["W_TO_C_false_veto_rate"], 1 / 3)

    def test_direction_selector_never_vetoes_same_class(self):
        raw = make([[0.6, 0.9, 0.1]])
        candidate, log = direction_selector_rollout(
            raw, lambda features: torch.ones(features.shape[0]), 0.5, "ranker"
        )
        self.assertFalse(log["keep"][0, 0])
        self.assertTrue(log["keep"][0, 1])
        self.assertEqual(candidate["predictions"][0].tolist(), [0, 0, 0])

    def test_regression_survival_curve_exact(self):
        trajectory = make(
            [
                [0.9, 0.1, 0.1, 0.9],
                [0.1, 0.9, 0.9, 0.1],
                [0.9, 0.9, 0.9, 0.9],
            ]
        )
        rows = regression_survival_rows(
            trajectory, trajectory,
            family="x", candidate_id="x", seed=3, split="test"
        )
        expected = [1.0, 2 / 3, 2 / 3, 1 / 3]
        self.assertTrue(
            all(abs(row["candidate_regression_free_survival"] - value) < 1e-6
                for row, value in zip(rows, expected))
        )

    def test_delayed_regression_is_not_prevention(self):
        raw = make([[0.9, 0.1, 0.1]])
        candidate = make([[0.9, 0.9, 0.1]])
        comparison = regression_event_comparison(candidate, raw)
        self.assertEqual(comparison["regression_prevented_count"], 0)
        self.assertEqual(comparison["regression_both_event_count"], 1)
        self.assertEqual(comparison["both_event_delay_mean"], 1.0)

    def test_never_event_not_averaged(self):
        raw = make([[0.9, 0.1], [0.9, 0.9]])
        candidate = make([[0.9, 0.1], [0.9, 0.1]])
        comparison = regression_event_comparison(candidate, raw)
        self.assertEqual(comparison["regression_both_event_count"], 1)
        self.assertEqual(comparison["regression_induced_count"], 1)
        self.assertEqual(comparison["both_event_delay_mean"], 0.0)

    def test_valid_hidden_trajectory(self):
        trajectory = make([[0.8, 0.9]])
        trajectory["hidden_features"] = torch.randn(1, 2, 4)
        trajectory["hidden_feature_metadata"] = {
            "format_version": 1,
            "uses_target": False,
            "causal": True,
            "dimension": 4,
            "groups": [{"name": "u2_mean"}],
        }
        result = validate_hidden_trajectory(trajectory)
        self.assertEqual(result["dimension"], 4)

    def test_target_dependent_hidden_schema_rejected(self):
        trajectory = make([[0.8, 0.9]])
        trajectory["hidden_features"] = torch.randn(1, 2, 4)
        trajectory["hidden_feature_metadata"] = {
            "format_version": 1,
            "uses_target": False,
            "causal": True,
            "dimension": 4,
            "groups": [{"name": "is_correct"}],
        }
        with self.assertRaisesRegex(ValueError, "target-dependent"):
            validate_hidden_trajectory(trajectory)

    def test_split_overlap_rejected(self):
        by_split = {}
        for split, indices in (("train", [0, 1]), ("val", [1, 2]), ("test", [0, 1])):
            by_split[split] = [
                make([[0.8, 0.9], [0.2, 0.9]], seed=seed, sample_index=indices, split=split)
                for seed in (3, 4, 5)
            ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_trajectories(by_split)

    def test_hidden_decoder_shape_and_alpha_bounds(self):
        model = HiddenRCCED(torch.zeros(4), torch.ones(4), alpha_min=0.1)
        decoded, alpha = model(torch.randn(3, 4, 2), torch.randn(3, 4, 4))
        self.assertEqual(decoded.shape, (3, 4, 2))
        self.assertEqual(alpha.shape, (3, 4))
        self.assertTrue(torch.all(alpha[:, 1:] >= 0.1))
        self.assertTrue(torch.all(alpha <= 1.0))

    def test_hidden_loss_is_finite_and_differentiable(self):
        decoded = torch.randn(3, 4, 2, requires_grad=True)
        raw = torch.randn(3, 4, 2)
        loss, parts = hidden_rc_ced_loss(
            decoded, raw, torch.zeros(3, dtype=torch.long),
            destructive_weight=1.0, intervention_weight=0.02
        )
        loss.backward()
        self.assertTrue(math.isfinite(parts["loss"]))
        self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_pooled_recovery_below_095_fails(self):
        passed, failures, _ = candidate_guardrail_pass(
            good_rows(candidate_matched_recovery_count=94,
                      seed_recovery_preservation_ratio=0.94),
            guardrails(),
        )
        self.assertFalse(passed)
        self.assertIn("pooled_recovery", failures)

    def test_any_seed_recovery_below_090_fails(self):
        rows = good_rows()
        rows[0]["seed_recovery_preservation_ratio"] = 0.89
        passed, failures, _ = candidate_guardrail_pass(rows, guardrails())
        self.assertFalse(passed)
        self.assertIn("per_seed_recovery", failures)

    def test_delayed_only_candidate_fails(self):
        passed, failures, _ = candidate_guardrail_pass(
            good_rows(ever_regressed_fraction_change=0.0), guardrails()
        )
        self.assertFalse(passed)
        self.assertIn("ever_regressed_not_merely_delayed", failures)

    def test_simple_filter_selected_by_parsimony(self):
        simple = {
            "guardrail_pass": True, "regression_reduction_pp": 0.10,
            "pooled_matched_recovery_preservation": 0.96,
            "ever_regressed_fraction_change_mean": -0.01,
            "mean_prefix_accuracy_change_mean": 0.0,
            "final_accuracy_change_mean": 0.0,
        }
        selector = dict(simple, regression_reduction_pp=0.11)
        result = recommend_branch(simple, selector, None)
        self.assertEqual(result["recommended_branch"], "finish_with_simple_filter")

    def test_selector_requires_material_dominance(self):
        simple = {
            "guardrail_pass": True, "regression_reduction_pp": 0.10,
            "pooled_matched_recovery_preservation": 0.96,
            "ever_regressed_fraction_change_mean": -0.01,
            "mean_prefix_accuracy_change_mean": 0.0,
            "final_accuracy_change_mean": 0.0,
        }
        selector = dict(simple, regression_reduction_pp=0.13)
        result = recommend_branch(simple, selector, None)
        self.assertEqual(result["recommended_branch"], "continue_output_only_selector")

    def test_hidden_selected_only_if_it_passes(self):
        result = recommend_branch(
            {"guardrail_pass": False},
            {"guardrail_pass": False},
            {"guardrail_pass": True},
        )
        self.assertEqual(result["recommended_branch"], "move_to_hidden_state_rc_ced")

    def test_hidden_export_payload(self):
        dataset = TensorDataset(
            torch.tensor([[1.0], [2.0]]), torch.zeros(2, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=2)
        metadata = {
            "hidden_feature_metadata": {
                "format_version": 1,
                "uses_target": False,
                "causal": True,
                "dimension": 1,
                "groups": [{"name": "u2_mean"}],
            }
        }
        payload = export_trajectory_payload(
            HiddenPrefixModel(),
            loader,
            torch.arange(2),
            split="train",
            device=torch.device("cpu"),
            forward_args=SimpleNamespace(
                gate_threshold=0.5,
                min_prefix_steps=1,
                temporal_prefix_steps=0,
                temporal_prefix_mode="none",
            ),
            metadata=metadata,
            include_hidden_features=True,
        )
        self.assertEqual(payload["hidden_features"].shape, (2, 2, 1))
        self.assertFalse(payload["hidden_feature_metadata"]["uses_target"])

    def test_fixed_lif_temporal_feature_shape(self):
        model = FixedLIFSNN("nmnist", tmax=2)
        output = model(
            torch.zeros(1, 2, 2, 34, 34),
            return_prefix_logits=True,
            return_temporal_features=True,
        )
        self.assertEqual(output["prefix_logits"].shape, (1, 2, 10))
        self.assertEqual(output["temporal_features"].shape, (1, 2, 192))
        self.assertTrue(torch.isfinite(output["temporal_features"]).all())



if __name__ == "__main__":
    unittest.main()
