"""Pure tensor utilities for post-hoc temporal stopping diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch
from torch import Tensor


def first_satisfied_timestep(condition: Tensor, min_timestep: int = 1) -> Tensor:
    """Return 1-based first true timestep, falling back to the final timestep."""
    if condition.ndim != 2:
        raise ValueError("condition must have shape [N,T].")
    n, tmax = condition.shape
    if tmax < 1:
        raise ValueError("condition must contain at least one timestep.")
    minimum = max(1, min(int(min_timestep), tmax))
    allowed = condition.clone()
    allowed[:, : minimum - 1] = False
    timesteps = torch.arange(1, tmax + 1, device=condition.device).expand(n, -1)
    candidates = torch.where(allowed, timesteps, tmax + 1)
    first = candidates.min(dim=1).values
    return torch.where(first <= tmax, first, tmax)


def evaluate_stopping_policy(
    predictions: Tensor,
    targets: Tensor,
    stop_timesteps: Tensor,
    *,
    policy: str,
    threshold: float | None = None,
    lambda_cost: float | None = None,
) -> dict[str, Any]:
    if predictions.ndim != 2 or targets.ndim != 1:
        raise ValueError("predictions and targets must have shapes [N,T] and [N].")
    n, tmax = predictions.shape
    selected = predictions.gather(1, (stop_timesteps.long() - 1).unsqueeze(1)).squeeze(1)
    accuracy = selected.eq(targets).float().mean().item() * 100.0
    average_timestep = stop_timesteps.float().mean().item()
    return {
        "Policy": policy,
        "Threshold": "" if threshold is None else float(threshold),
        "Lambda": "" if lambda_cost is None else float(lambda_cost),
        "Accuracy": accuracy,
        "Average Timestep": average_timestep,
        "Normalized Average Timestep": average_timestep / float(tmax),
        "Error Rate": 100.0 - accuracy,
        "Number of Samples": n,
    }


def confidence_stopping(confidence: Tensor, threshold: float, min_timestep: int = 1) -> Tensor:
    return first_satisfied_timestep(confidence >= threshold, min_timestep)


def entropy_stopping(entropy: Tensor, threshold: float, min_timestep: int = 1) -> Tensor:
    return first_satisfied_timestep(entropy <= threshold, min_timestep)


def margin_stopping(margin: Tensor, threshold: float, min_timestep: int = 1) -> Tensor:
    return first_satisfied_timestep(margin >= threshold, min_timestep)


def confidence_stability_stopping(
    predictions: Tensor,
    confidence: Tensor,
    threshold: float,
    *,
    window: int = 2,
    min_timestep: int = 1,
) -> Tensor:
    if window < 2:
        raise ValueError("window must be at least 2.")
    condition = torch.zeros_like(confidence, dtype=torch.bool)
    for end in range(window - 1, predictions.shape[1]):
        recent = predictions[:, end - window + 1 : end + 1]
        stable = recent.eq(recent[:, -1:]).all(dim=1)
        condition[:, end] = stable & (confidence[:, end] >= threshold)
    return first_satisfied_timestep(condition, max(min_timestep, window))


def earliest_correct_oracle(correct: Tensor, min_timestep: int = 1) -> Tensor:
    return first_satisfied_timestep(correct.bool(), min_timestep)


def earliest_stable_correct_oracle(correct: Tensor, min_timestep: int = 1) -> Tensor:
    suffix_stable = torch.flip(
        torch.cumprod(torch.flip(correct.to(torch.int64), dims=[1]), dim=1),
        dims=[1],
    ).bool()
    return first_satisfied_timestep(suffix_stable, min_timestep)


def cost_aware_oracle(correct: Tensor, lambda_cost: float, min_timestep: int = 1) -> Tensor:
    """Minimize classification error + lambda * t/T; argmin breaks ties early."""
    n, tmax = correct.shape
    timesteps = torch.arange(1, tmax + 1, device=correct.device, dtype=torch.float32)
    costs = (~correct.bool()).float() + float(lambda_cost) * timesteps[None, :] / float(tmax)
    minimum = max(1, min(int(min_timestep), tmax))
    if minimum > 1:
        costs[:, : minimum - 1] = torch.inf
    return costs.argmin(dim=1) + 1


def trajectory_outcomes(correct: Tensor) -> dict[str, Tensor]:
    """Classify every sample-prefix pair into four continuation outcomes."""
    correct = correct.bool()
    n, tmax = correct.shape
    future_correct = torch.zeros((n, tmax), dtype=torch.bool, device=correct.device)
    future_incorrect = torch.zeros_like(future_correct)
    if tmax > 1:
        future_correct[:, :-1] = torch.flip(
            torch.cumsum(torch.flip(correct[:, 1:].to(torch.int64), dims=[1]), dim=1), dims=[1]
        ) > 0
        future_incorrect[:, :-1] = torch.flip(
            torch.cumsum(torch.flip((~correct[:, 1:]).to(torch.int64), dims=[1]), dim=1), dims=[1]
        ) > 0
    return {
        "safe_stop": correct & ~future_incorrect,
        "beneficial_continuation": ~correct & future_correct,
        "destructive_continuation": correct & future_incorrect,
        "futile_continuation": ~correct & ~future_correct,
    }


def pareto_frontier(rows: Iterable[dict[str, Any]], group_key: str = "Policy") -> list[dict[str, Any]]:
    """Remove accuracy/timestep dominated points independently per policy family."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    frontier: list[dict[str, Any]] = []
    for group_rows in groups.values():
        for candidate in group_rows:
            accuracy = float(candidate["Accuracy"])
            timestep = float(candidate["Average Timestep"])
            dominated = any(
                float(other["Accuracy"]) >= accuracy
                and float(other["Average Timestep"]) <= timestep
                and (
                    float(other["Accuracy"]) > accuracy
                    or float(other["Average Timestep"]) < timestep
                )
                for other in group_rows
                if other is not candidate
            )
            if not dominated:
                frontier.append(candidate)
    return sorted(frontier, key=lambda row: (str(row[group_key]), float(row["Average Timestep"])))
