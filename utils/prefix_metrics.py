"""Project diagnostics for prefix-wise SNN predictions."""

from __future__ import annotations

import torch
from torch import Tensor


def _correct_by_prefix(prefix_logits: Tensor, targets: Tensor) -> Tensor:
    if prefix_logits.ndim != 3:
        raise ValueError(f"Expected prefix_logits [B,T,C], got {tuple(prefix_logits.shape)}.")
    if targets.ndim != 1 or targets.shape[0] != prefix_logits.shape[0]:
        raise ValueError("targets must have shape [B] matching prefix_logits.")
    return prefix_logits.argmax(dim=-1).eq(targets[:, None])


def prefix_accuracy_curve(prefix_logits: Tensor, targets: Tensor) -> Tensor:
    """Return dataset prefix accuracies in percentage points."""
    return _correct_by_prefix(prefix_logits, targets).float().mean(dim=0) * 100.0


def consecutive_regression_rate(prefix_logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
    """Return population and correct-conditioned regression percentages."""
    correct = _correct_by_prefix(prefix_logits, targets)
    regressed = correct[:, :-1] & ~correct[:, 1:]
    population = regressed.float().mean(dim=0) * 100.0
    correct_count = correct[:, :-1].sum(dim=0)
    regressed_count = regressed.sum(dim=0)
    valid = correct_count > 0
    conditional = torch.zeros_like(population)
    conditional[valid] = regressed_count[valid].float() / correct_count[valid].float() * 100.0
    mean_population = population.mean() if population.numel() else prefix_logits.new_tensor(0.0)
    mean_conditional = conditional[valid].mean() if valid.any() else prefix_logits.new_tensor(0.0)
    return {
        "population_per_transition": population,
        "conditional_per_transition": conditional,
        "correct_count_per_transition": correct_count,
        "conditional_valid_transition_mask": valid,
        "mean_population": mean_population,
        "mean_conditional": mean_conditional,
        # Backward-compatible aliases for the original population metric.
        "per_transition": population,
        "mean": mean_population,
    }


def ever_regressed_rate(prefix_logits: Tensor, targets: Tensor) -> Tensor:
    """Percentage of ever-correct samples that become incorrect later."""
    correct = _correct_by_prefix(prefix_logits, targets)
    ever_correct = correct.any(dim=1)
    seen_correct = correct.cumsum(dim=1) > 0
    later_incorrect = (seen_correct & ~correct).any(dim=1)
    denominator = ever_correct.sum()
    if denominator.item() == 0:
        return prefix_logits.new_tensor(0.0)
    return later_incorrect[ever_correct].float().mean() * 100.0


def negative_temporal_gain(accuracy_curve: Tensor) -> Tensor:
    """Project diagnostic: summed prefix-to-prefix accuracy decreases."""
    if accuracy_curve.ndim != 1:
        raise ValueError("accuracy_curve must be one-dimensional.")
    return torch.relu(accuracy_curve[:-1] - accuracy_curve[1:]).sum()


def mean_negative_temporal_gain(accuracy_curve: Tensor) -> Tensor:
    transitions = max(1, accuracy_curve.numel() - 1)
    return negative_temporal_gain(accuracy_curve) / transitions


def worst_prefix_accuracy(accuracy_curve: Tensor) -> Tensor:
    return accuracy_curve.min()


def prefix_accuracy_auc(accuracy_curve: Tensor) -> Tensor:
    """Discrete mean of prefix accuracies, not a continuous integral."""
    return accuracy_curve.mean()


def first_correct_timestep(prefix_logits: Tensor, targets: Tensor) -> Tensor:
    """Return 1-based first-correct timesteps; T+1 means never correct."""
    correct = _correct_by_prefix(prefix_logits, targets)
    timestep = torch.arange(1, correct.shape[1] + 1, device=correct.device)
    sentinel = correct.shape[1] + 1
    candidates = torch.where(correct, timestep[None, :], sentinel)
    return candidates.min(dim=1).values


def stable_correct_timestep(prefix_logits: Tensor, targets: Tensor) -> Tensor:
    """Return 1-based stable-correct timesteps; T+1 means never stable."""
    correct = _correct_by_prefix(prefix_logits, targets)
    stable = torch.flip(torch.cumprod(torch.flip(correct.to(torch.int64), dims=[1]), dim=1), dims=[1]).bool()
    timestep = torch.arange(1, correct.shape[1] + 1, device=correct.device)
    sentinel = correct.shape[1] + 1
    candidates = torch.where(stable, timestep[None, :], sentinel)
    return candidates.min(dim=1).values
