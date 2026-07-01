"""Fixed convolutional backbones shared by all SNN variants."""

from __future__ import annotations

import torch
from torch import nn


class ConvSNNBackbone(nn.Module):
    def __init__(self, dataset: str) -> None:
        super().__init__()
        dataset = dataset.lower()
        if dataset in {"fashionmnist", "fashion-mnist", "mnist"}:
            self.dataset = "fashionmnist"
            in_channels = 1
            c1, c2 = 32, 64
            classifier_in = 64 * 7 * 7
        elif dataset in {"cifar10", "cifar-10"}:
            self.dataset = "cifar10"
            in_channels = 3
            c1, c2 = 64, 128
            classifier_in = 128 * 8 * 8
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(classifier_in, 10)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        return x1, x2

    def classify(self, s2: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.pool(s2), start_dim=1))
