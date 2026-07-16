"""Policy and calibration metrics for learned stopping rules."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def policy_metrics(predictions: Tensor, targets: Tensor, stop_timesteps: Tensor, policy: str, **metadata: Any) -> dict[str, Any]:
    n, tmax = predictions.shape
    if stop_timesteps.shape != (n,) or not ((stop_timesteps >= 1) & (stop_timesteps <= tmax)).all():
        raise ValueError("stop_timesteps must be one-based values within [1,T].")
    selected = predictions.gather(1, (stop_timesteps - 1).long().unsqueeze(1)).squeeze(1)
    accuracy = selected.eq(targets).float().mean().item() * 100.0
    values = stop_timesteps.float()
    row = {"Policy": policy, "Accuracy": accuracy, "Error Rate": 100.0 - accuracy,
           "Average Timestep": values.mean().item(), "Normalized Average Timestep": values.mean().item() / tmax,
           "Median Timestep": values.median().item(), "Timestep Standard Deviation": values.std(unbiased=False).item(),
           "Number of Samples": n}
    row.update({f"Fraction Stopped T{t}": stop_timesteps.eq(t).float().mean().item() for t in range(1, tmax + 1)})
    row.update(metadata)
    return row


def binary_metrics(probabilities: Tensor, targets: Tensor, bins: int = 10) -> dict[str, Any]:
    probabilities, targets = probabilities.flatten().float(), targets.flatten().float()
    prevalence = targets.mean().item()
    order = probabilities.argsort(descending=True)
    y = targets[order]
    positives, negatives = y.sum(), (1 - y).sum()
    valid = bool(positives > 0 and negatives > 0)
    if valid:
        tpr = torch.cat([y.new_zeros(1), y.cumsum(0) / positives])
        fpr = torch.cat([y.new_zeros(1), (1 - y).cumsum(0) / negatives])
        auroc = torch.trapz(tpr, fpr).item()
        precision = y.cumsum(0) / torch.arange(1, len(y) + 1, device=y.device)
        auprc = (precision * y).sum().div(positives).item()
    else:
        auroc = auprc = math.nan
    ece = 0.0
    for lower in torch.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1 / bins
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if selected.any():
            ece += selected.float().mean().item() * abs(probabilities[selected].mean().item() - targets[selected].mean().item())
    return {"auroc": auroc, "auprc": auprc, "valid": valid, "brier_score": ((probabilities - targets) ** 2).mean().item(),
            "ece": ece, "positive_prevalence": prevalence}


def multiclass_metrics(logits: Tensor, targets: Tensor, class_names: dict[int, str]) -> dict[str, Any]:
    logits, targets = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long()
    predictions = logits.argmax(dim=-1)
    result: dict[str, Any] = {"cross_entropy": F.cross_entropy(logits, targets).item()}
    f1_values = []
    probabilities = logits.softmax(dim=-1)
    for class_id, name in class_names.items():
        true = targets.eq(class_id); predicted = predictions.eq(class_id)
        tp = (true & predicted).sum().item(); fp = (~true & predicted).sum().item(); fn = (true & ~predicted).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        result[f"{name}_precision"] = precision; result[f"{name}_recall"] = recall
        if name in {"improve", "harm"}:
            binary = binary_metrics(probabilities[:, class_id], true.float())
            result[f"{name}_vs_rest_auroc"] = binary["auroc"]
            result[f"{name}_vs_rest_auroc_valid"] = binary["valid"]
    result["macro_f1"] = sum(f1_values) / len(f1_values)
    return result


def masked_probability_metrics(probabilities: Tensor, targets: Tensor, mask: Tensor) -> dict[str, Any]:
    selected_probabilities = probabilities[mask.bool()]
    selected_targets = targets.float()[mask.bool()]
    result = binary_metrics(selected_probabilities, selected_targets)
    result["error_probability_mae"] = (selected_probabilities - selected_targets).abs().mean().item()
    result["masked_bce"] = F.binary_cross_entropy(selected_probabilities.clamp(1e-7, 1 - 1e-7), selected_targets).item()
    return result
