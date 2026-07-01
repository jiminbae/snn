"""Metric helpers for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def accuracy(logits: Tensor, target: Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean().item() * 100.0


def energy_proxy(spike_rate: float, effective_timestep: float) -> float:
    # This is a proxy only: lower spike activity and fewer active timesteps are
    # common indicators for potential neuromorphic efficiency.
    return spike_rate * effective_timestep
