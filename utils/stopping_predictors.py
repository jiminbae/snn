"""Small matched-capacity MLPs and sequential stopping decisions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StoppingMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def predictor_output_dim(name: str, tmax: int) -> int:
    return {"recoverability_final": 1, "final_horizon_gain": 3, "one_step": 2, "multi_horizon": tmax}[name]


def masked_bce_with_logits(logits: Tensor, targets: Tensor, mask: Tensor, weights: Tensor | None = None) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    effective = mask.float()
    if weights is not None:
        effective = effective * weights
    per_state = (loss * effective).sum(dim=-1) / effective.sum(dim=-1).clamp_min(1.0) if loss.ndim == 3 else loss * effective
    valid = effective.sum(dim=-1) > 0 if effective.ndim == loss.ndim and loss.ndim == 3 else effective > 0
    return per_state[valid].mean() if valid.any() else loss.sum() * 0.0


def recoverability_stops(probabilities: Tensor, threshold: float) -> Tensor:
    n, tmax = probabilities.shape
    stops = torch.full((n,), tmax, dtype=torch.long, device=probabilities.device)
    active = torch.ones(n, dtype=torch.bool, device=probabilities.device)
    for timestep in range(tmax - 1):
        stop = active & (probabilities[:, timestep] < threshold)
        stops[stop] = timestep + 1
        active &= ~stop
    return stops


def one_step_stops(error_probabilities: Tensor, lambda_cost: float) -> Tensor:
    n, tmax, _ = error_probabilities.shape
    stops = torch.full((n,), tmax, dtype=torch.long, device=error_probabilities.device)
    active = torch.ones(n, dtype=torch.bool, device=error_probabilities.device)
    for timestep in range(tmax - 1):
        stop_cost = error_probabilities[:, timestep, 0] + lambda_cost * (timestep + 1) / tmax
        continue_cost = error_probabilities[:, timestep, 1] + lambda_cost * (timestep + 2) / tmax
        stop = active & (continue_cost >= stop_cost)
        stops[stop] = timestep + 1
        active &= ~stop
    return stops


def multi_horizon_stops(error_probabilities: Tensor, lambda_cost: float) -> Tensor:
    n, tmax, horizons = error_probabilities.shape
    if horizons != tmax:
        raise ValueError("multi-horizon probabilities must have shape [N,T,T].")
    stops = torch.full((n,), tmax, dtype=torch.long, device=error_probabilities.device)
    active = torch.ones(n, dtype=torch.bool, device=error_probabilities.device)
    time_cost = lambda_cost * torch.arange(1, tmax + 1, device=error_probabilities.device) / tmax
    for timestep in range(tmax - 1):
        choice = (error_probabilities[:, timestep, timestep:] + time_cost[timestep:]).argmin(dim=1) + timestep
        stop = active & choice.eq(timestep)
        stops[stop] = timestep + 1
        active &= ~stop
    return stops
