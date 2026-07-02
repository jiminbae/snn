"""Fixed convolutional backbones shared by all SNN variants."""

from __future__ import annotations

import torch
from torch import nn

from utils.data import canonical_dataset_name, get_dataset_spec


class ConvSNNBackbone(nn.Module):
    def __init__(self, dataset: str) -> None:
        super().__init__()
        self.dataset = canonical_dataset_name(dataset)
        spec = get_dataset_spec(self.dataset)
        self.num_classes = int(spec["num_classes"])
        in_channels = int(spec["input_channels"])

        if self.dataset == "fashionmnist":
            c1, c2 = 32, 64
            classifier_pool_size = 7
        elif self.dataset == "cifar10":
            c1, c2 = 64, 128
            classifier_pool_size = 8
        elif self.dataset == "nmnist":
            c1, c2 = 32, 64
            classifier_pool_size = 7
        elif self.dataset == "dvs_gesture":
            c1, c2 = 64, 128
            classifier_pool_size = 8
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.classifier_pool = nn.AdaptiveAvgPool2d((classifier_pool_size, classifier_pool_size))
        self.classifier = nn.Linear(c2 * classifier_pool_size * classifier_pool_size, self.num_classes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        return x1, x2

    def classify(self, s2: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.classifier_pool(s2), start_dim=1))
