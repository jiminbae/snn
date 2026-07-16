import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from train import is_better_validation_checkpoint, main, restore_model_checkpoint
from utils.data import build_dataloaders, build_train_val_test_dataloaders


def synthetic_datasets(*args, **kwargs):
    train = TensorDataset(torch.arange(40).float().unsqueeze(1), torch.arange(40) % 2)
    test = TensorDataset(torch.arange(10).float().unsqueeze(1), torch.arange(10) % 2)
    return train, test


class ValidationSplitTests(unittest.TestCase):
    def build(self, seed=42, ratio=0.25):
        with patch("utils.data._build_datasets", side_effect=synthetic_datasets):
            return build_train_val_test_dataloaders(
                "nmnist", ".", 4, tmax=8, val_ratio=ratio, split_seed=seed, num_workers=0
            )

    def test_split_is_deterministic_disjoint_and_complete(self):
        *_, first = self.build(seed=42)
        *_, repeated = self.build(seed=42)
        *_, different = self.build(seed=43)
        self.assertTrue(torch.equal(first["train_indices"], repeated["train_indices"]))
        self.assertTrue(torch.equal(first["val_indices"], repeated["val_indices"]))
        self.assertFalse(torch.equal(first["val_indices"], different["val_indices"]))
        train = set(first["train_indices"].tolist())
        val = set(first["val_indices"].tolist())
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(range(40)))

    def test_invalid_or_empty_splits_fail(self):
        for ratio in (0.0, 1.0, -0.1, 1.1, 0.01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.build(ratio=ratio)

    def test_legacy_builder_still_returns_two_loaders(self):
        with patch("utils.data._build_datasets", side_effect=synthetic_datasets):
            loaders = build_dataloaders("nmnist", ".", 4, tmax=8, num_workers=0)
        self.assertEqual(len(loaders), 2)


class SelectionTests(unittest.TestCase):
    def test_validation_comparator_ties_and_non_finite_values(self):
        better = is_better_validation_checkpoint
        def compare(candidate_acc, candidate_loss, best_acc, best_loss, metric):
            return better(
                candidate_acc=candidate_acc,
                candidate_loss=candidate_loss,
                best_acc=best_acc,
                best_loss=best_loss,
                selection_metric=metric,
            )

        self.assertTrue(compare(80.0, 0.5, None, None, "val_acc"))
        self.assertTrue(compare(81.0, 0.7, 80.0, 0.5, "val_acc"))
        self.assertTrue(compare(80.0, 0.4, 80.0, 0.5, "val_acc"))
        self.assertFalse(compare(80.0, 0.5, 80.0, 0.5, "val_acc"))
        self.assertTrue(compare(79.0, 0.4, 80.0, 0.5, "val_loss"))
        self.assertTrue(compare(81.0, 0.5, 80.0, 0.5, "val_loss"))
        self.assertFalse(compare(80.0, 0.5, 80.0, 0.5, "val_loss"))
        for value in (math.nan, math.inf, -math.inf):
            self.assertFalse(compare(value, 0.5, 80.0, 0.5, "val_acc"))
            self.assertFalse(compare(80.0, value, 80.0, 0.5, "val_acc"))

    def test_checkpoint_restores_model_parameters(self):
        model = nn.Linear(2, 1)
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, path)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            checkpoint = restore_model_checkpoint(model, path, torch.device("cpu"))
        self.assertEqual(checkpoint["epoch"], 3)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]))

    def test_best_val_flow_evaluates_test_once_after_restore(self):
        dataset = TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))
        loaders = tuple(torch.utils.data.DataLoader(dataset, batch_size=2) for _ in range(3))
        split = {
            "train_indices": torch.tensor([0, 1]),
            "val_indices": torch.tensor([2, 3]),
            "split_seed": 42,
            "val_ratio": 0.5,
        }
        model = nn.Linear(2, 1)
        validation_calls = []
        test_calls = []

        def fake_train(model, loader, optimizer, scaler, device, args, epoch):
            with torch.no_grad():
                model.weight.fill_(float(epoch))
            return {
                "loss": 1.0, "ce_loss": 1.0, "ce_hard": 0.0, "consistency_loss": 0.0,
                "spike_cost": 0.0, "time_cost": 0.0, "hard_budget_cost": 0.0,
                "target_budget_loss": 0.0, "min_target_loss": 0.0, "hard_budget_proxy": 0.0,
                "acc": 0.0, "acc_hard": 0.0,
            }

        def fake_validation(*args, **kwargs):
            validation_calls.append(1)
            epoch = len(validation_calls)
            return {"loss": float(epoch), "acc": 100.0 - epoch, "soft_acc": 0.0, "hard_acc": 0.0}

        def fake_test(model, *args, **kwargs):
            test_calls.append(float(model.weight.detach().flatten()[0]))
            return {
                "test_acc": 0.0, "loss": 0.0, "soft_acc": 0.0, "hard_acc": 0.0,
                "raw_spike_rate": 0.0, "gated_spike_rate": 0.0, "prefix_spike_rate": 0.0,
                "effective_timestep": 0.0, "hard_effective_timestep": 0.0,
                "layer1_effective_timestep": 0.0, "layer2_effective_timestep": 0.0,
                "layer1_hard_timestep": 0.0, "layer2_hard_timestep": 0.0,
                "energy_proxy": 0.0, "prefix_energy_proxy": 0.0,
                "executed_timestep": 0.0, "loop_energy_proxy": 0.0,
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "sys.argv",
            ["train.py", "--epochs", "2", "--device", "cpu", "--checkpoint-selection", "best_val",
             "--val-ratio", "0.5", "--results-dir", tmp, "--run-name", "flow", "--num-workers", "0"],
        ), patch("train.build_train_val_test_dataloaders", return_value=(*loaders, split)), patch(
            "train.build_model", return_value=model
        ), patch("train.train_one_epoch", side_effect=fake_train), patch(
            "train.evaluate_loader", side_effect=fake_validation
        ), patch("train.evaluate", side_effect=fake_test), patch("train.plot_training_curves"), patch(
            "train.plot_timestep_gates"
        ):
            main()

        self.assertEqual(len(validation_calls), 2)
        self.assertEqual(test_calls, [1.0])


if __name__ == "__main__":
    unittest.main()
