"""Leakage-free supervision targets for temporal stopping predictors."""

from __future__ import annotations

import torch
from torch import Tensor

OUTCOME_CLASS_NAMES = {0: "same", 1: "improve", 2: "harm"}


def build_stopping_targets(prefix_logits: Tensor, targets: Tensor) -> dict[str, Tensor | dict[int, str]]:
    if prefix_logits.ndim != 3 or targets.shape != (prefix_logits.shape[0],):
        raise ValueError("Expected prefix_logits [N,T,C] and targets [N].")
    predictions = prefix_logits.argmax(dim=-1)
    error = predictions.ne(targets[:, None])
    n, tmax = error.shape
    continue_mask = torch.ones((n, tmax), dtype=torch.bool, device=error.device)
    continue_mask[:, -1] = False

    recoverable = error & ~error[:, -1:]
    recoverable[:, -1] = False
    outcomes = torch.zeros_like(error, dtype=torch.long)
    outcomes[error & ~error[:, -1:]] = 1
    outcomes[~error & error[:, -1:]] = 2

    next_error = torch.zeros_like(error)
    next_error[:, :-1] = error[:, 1:]
    horizon = torch.arange(tmax, device=error.device)
    future_mask = horizon[None, None, :] >= horizon[None, :, None]
    future_mask = future_mask.expand(n, -1, -1).clone()
    future_error = error[:, None, :].expand(-1, tmax, -1).clone()
    return {
        "predictions": predictions,
        "error": error.float(),
        "continue_mask": continue_mask,
        "recoverable_final": recoverable.float(),
        "final_horizon_outcome": outcomes,
        "outcome_class_names": OUTCOME_CLASS_NAMES,
        "current_error": error.float(),
        "next_error": next_error.float(),
        "next_target_mask": continue_mask,
        "future_error_target": future_error.float(),
        "future_horizon_mask": future_mask,
    }


def oracle_future_choice(error: Tensor, lambda_cost: float) -> tuple[Tensor, Tensor]:
    """Return zero-based best reachable horizon and whether to continue at every state."""
    if error.ndim != 2:
        raise ValueError("error must have shape [N,T].")
    n, tmax = error.shape
    time_cost = torch.arange(1, tmax + 1, device=error.device, dtype=torch.float32) / tmax
    costs = error.float() + float(lambda_cost) * time_cost[None, :]
    choices = torch.empty((n, tmax), dtype=torch.long, device=error.device)
    for timestep in range(tmax):
        choices[:, timestep] = costs[:, timestep:].argmin(dim=1) + timestep
    current = torch.arange(tmax, device=error.device)[None, :]
    return choices, choices.ne(current)


def action_margin_weights(error: Tensor, lambda_cost: float, margin_temperature: float) -> Tensor:
    if margin_temperature <= 0:
        raise ValueError("margin_temperature must be positive.")
    n, tmax = error.shape
    time_cost = torch.arange(1, tmax + 1, device=error.device, dtype=torch.float32) / tmax
    costs = error.float() + float(lambda_cost) * time_cost[None, :]
    weights = torch.zeros((n, tmax), device=error.device)
    for timestep in range(tmax - 1):
        margin = (costs[:, timestep] - costs[:, timestep + 1 :].min(dim=1).values).abs()
        weights[:, timestep] = (margin / margin_temperature).clamp(0.0, 1.0)
    return weights
