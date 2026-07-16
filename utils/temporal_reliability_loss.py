"""Losses and diagnostics for temporal reliability kill tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate(prefix_logits: Tensor, targets: Tensor | None = None) -> None:
    if prefix_logits.ndim != 3:
        raise ValueError("prefix_logits must have shape [B,T,C].")
    if targets is not None and targets.shape != (prefix_logits.shape[0],):
        raise ValueError("targets must have shape [B].")


def all_prefix_cross_entropy(prefix_logits: Tensor, targets: Tensor) -> Tensor:
    _validate(prefix_logits, targets)
    expanded_targets = targets[:, None].expand(-1, prefix_logits.shape[1])
    return F.cross_entropy(prefix_logits.flatten(0, 1), expanded_targets.reshape(-1))


def symmetric_temporal_kl(prefix_logits: Tensor, temperature: float = 1.0) -> Tensor:
    _validate(prefix_logits)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if prefix_logits.shape[1] < 2:
        return prefix_logits.sum() * 0.0
    scaled = prefix_logits / temperature
    log_probabilities = F.log_softmax(scaled, dim=-1)
    probabilities = log_probabilities.exp()
    forward = F.kl_div(log_probabilities[:, 1:], probabilities[:, :-1], reduction="none").sum(-1)
    backward = F.kl_div(log_probabilities[:, :-1], probabilities[:, 1:], reduction="none").sum(-1)
    return 0.5 * (forward + backward).mean() * temperature**2


def selective_regression_loss(
    prefix_logits: Tensor,
    targets: Tensor,
    confidence_threshold: float,
    margin: float,
    selection_mode: str = "hard",
) -> tuple[Tensor, dict[str, Tensor]]:
    _validate(prefix_logits, targets)
    if selection_mode not in {"hard", "soft"}:
        raise ValueError("selection_mode must be 'hard' or 'soft'.")
    if prefix_logits.shape[1] < 2:
        zero = prefix_logits.sum() * 0.0
        return zero, {"selected_transition_count": zero.detach(), "selected_transition_fraction": zero.detach(),
                      "mean_target_probability_drop": zero.detach(), "violating_transition_fraction": zero.detach()}

    probabilities = prefix_logits.softmax(dim=-1)
    target_probabilities = probabilities.gather(2, targets[:, None, None].expand(-1, prefix_logits.shape[1], 1)).squeeze(-1)
    current_correct = prefix_logits[:, :-1].argmax(dim=-1).eq(targets[:, None])
    current_confident = target_probabilities[:, :-1] >= confidence_threshold
    selected = current_correct & current_confident
    if selection_mode == "hard":
        weights = selected.float()
    else:
        confidence_weight = ((target_probabilities[:, :-1] - confidence_threshold) /
                             max(1e-8, 1.0 - confidence_threshold)).clamp(0.0, 1.0)
        weights = current_correct.float() * confidence_weight

    drops = target_probabilities[:, :-1] - target_probabilities[:, 1:]
    violations = drops + margin > 0
    denominator = weights.sum()
    loss = (weights * torch.relu(drops + margin)).sum() / denominator.clamp_min(1.0)
    selected_count = selected.sum()
    selected_fraction = selected.float().mean()
    mean_drop = (weights * drops).sum() / denominator.clamp_min(1.0)
    violating_fraction = (weights * violations.float()).sum() / denominator.clamp_min(1.0)
    diagnostics = {
        "selected_transition_count": selected_count.detach(),
        "selected_transition_fraction": selected_fraction.detach(),
        "mean_target_probability_drop": mean_drop.detach(),
        "violating_transition_fraction": violating_fraction.detach(),
    }
    return loss, diagnostics


def combine_temporal_objective(
    mode: str,
    base_total: Tensor,
    final_ce: Tensor,
    prefix_logits: Tensor | None,
    targets: Tensor,
    *,
    prefix_loss_weight: float,
    temporal_loss_weight: float,
    margin: float,
    confidence_threshold: float,
    temperature: float,
    selection_mode: str,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Add a temporal objective while preserving the exact final-CE base path."""
    zero = final_ce * 0.0
    diagnostics = {"selected_transition_fraction": zero.detach(), "violating_transition_fraction": zero.detach()}
    if mode == "final_ce":
        return base_total, zero, zero, diagnostics
    if prefix_logits is None:
        raise ValueError("Temporal training modes require prefix logits.")
    prefix_ce = all_prefix_cross_entropy(prefix_logits, targets)
    if mode == "all_prefix_ce":
        return base_total - final_ce + prefix_ce, prefix_ce, zero, diagnostics
    if mode == "symmetric_kl":
        temporal_loss = symmetric_temporal_kl(prefix_logits, temperature)
    elif mode == "selective_regression":
        temporal_loss, full_diagnostics = selective_regression_loss(
            prefix_logits, targets, confidence_threshold, margin, selection_mode
        )
        diagnostics = {key: full_diagnostics[key] for key in diagnostics}
    else:
        raise ValueError(f"Unknown temporal training mode: {mode}")
    return base_total + prefix_loss_weight * prefix_ce + temporal_loss_weight * temporal_loss, prefix_ce, temporal_loss, diagnostics


def temporal_reliability_metrics(prefix_logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
    """Return percentage-point dataset metrics for regression and recovery."""
    _validate(prefix_logits, targets)
    correct = prefix_logits.argmax(-1).eq(targets[:, None])
    correct_to_wrong = correct[:, :-1] & ~correct[:, 1:]
    wrong_to_correct = ~correct[:, :-1] & correct[:, 1:]
    ever_correct = correct.any(1)
    ever_regressed = (correct.cumsum(1).gt(0) & ~correct).any(1)
    ever_wrong = (~correct).any(1)
    ever_recovered = ((~correct).cumsum(1).gt(0) & correct).any(1)
    stable_correct = torch.flip(torch.cumprod(torch.flip(correct.long(), [1]), 1), [1]).bool().any(1)
    conditional_denominator = correct[:, :-1].sum()
    return {
        "final_accuracy": correct[:, -1].float().mean() * 100,
        "mean_prefix_accuracy": correct.float().mean() * 100,
        "minimum_prefix_accuracy": correct.float().mean(0).min() * 100,
        "ever_regressed_fraction": ever_regressed[ever_correct].float().mean() * 100 if ever_correct.any() else prefix_logits.new_tensor(0.0),
        "mean_population_regression": correct_to_wrong.float().mean() * 100,
        "mean_conditional_regression": correct_to_wrong.sum().float() / conditional_denominator.clamp_min(1) * 100,
        "correct_to_wrong_transition_count": correct_to_wrong.sum(),
        "destructive_transition_fraction": correct_to_wrong.float().mean() * 100,
        "ever_recovered_fraction": ever_recovered[ever_wrong].float().mean() * 100 if ever_wrong.any() else prefix_logits.new_tensor(0.0),
        "wrong_to_correct_transition_count": wrong_to_correct.sum(),
        "beneficial_transition_fraction": wrong_to_correct.float().mean() * 100,
        "stable_correct_fraction": stable_correct.float().mean() * 100,
    }
