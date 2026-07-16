import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset

from utils.trajectory_export import build_export_loaders, save_split_trajectories


class PrefixModel(nn.Module):
    def forward(self, images, **kwargs):
        logits = torch.stack([torch.cat([images, -images], dim=1), torch.cat([images + 1, -images], dim=1)], dim=1)
        return {"prefix_logits": logits, "logits": logits[:, -1]}


class SplitTrajectoryExportTests(unittest.TestCase):
    def test_export_preserves_indices_and_accuracy(self):
        train = TensorDataset(torch.arange(6).float().unsqueeze(1), torch.zeros(6, dtype=torch.long))
        test = TensorDataset(torch.arange(3).float().unsqueeze(1), torch.zeros(3, dtype=torch.long))
        split = {"train_indices": torch.tensor([0, 2, 4]), "val_indices": torch.tensor([1, 3, 5])}
        loaders, indices = build_export_loaders(train, test, split, 2, 0)
        args = SimpleNamespace(gate_threshold=.5, min_prefix_steps=1, temporal_prefix_steps=0, temporal_prefix_mode="none")
        with tempfile.TemporaryDirectory() as tmp:
            summary = save_split_trajectories(PrefixModel(), loaders, indices, tmp, device=torch.device("cpu"),
                                              forward_args=args, metadata={"tmax": 2}, expected_test_accuracy=100.0)
            payload = torch.load(Path(tmp) / "train_trajectories.pt", weights_only=True)
            self.assertEqual(payload["sample_indices"].tolist(), [0, 2, 4])
            self.assertEqual(summary["splits"]["test"]["samples"], 3)
            test_payload = torch.load(Path(tmp) / "test_trajectories.pt", weights_only=True)
            expected = test_payload["prefix_logits"].argmax(-1)[:, -1].eq(test_payload["targets"]).float().mean().item() * 100
            saved = __import__("json").loads((Path(tmp) / "trajectory_export_summary.json").read_text())["splits"]["test"]["final_accuracy"]
            self.assertEqual(expected, 100.0)
            self.assertEqual(saved, expected)
            second = list(iter(loaders["train"]))[0][0]
            self.assertTrue(torch.equal(second.flatten(), torch.tensor([0.0, 2.0])))

    def test_overlap_is_rejected(self):
        dataset = TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))
        with self.assertRaises(ValueError):
            build_export_loaders(dataset, dataset, {"train_indices": [0, 1], "val_indices": [1]}, 1, 0)


if __name__ == "__main__": unittest.main()
