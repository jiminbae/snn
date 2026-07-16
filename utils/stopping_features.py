"""Causal fixed-width features derived only from observed prefix logits."""

from __future__ import annotations

import torch
from torch import Tensor


def _summary(logits: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    probabilities = logits.softmax(dim=-1)
    top2 = probabilities.topk(min(2, logits.shape[-1]), dim=-1).values
    confidence = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1] if top2.shape[-1] > 1 else top2[..., 0]
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return probabilities, confidence, entropy, margin


def build_causal_features(prefix_logits: Tensor, mode: str) -> Tensor:
    if prefix_logits.ndim != 3 or not torch.isfinite(prefix_logits).all():
        raise ValueError("prefix_logits must be a finite [N,T,C] tensor.")
    n, tmax, classes = prefix_logits.shape
    probabilities, confidence, entropy, margin = _summary(prefix_logits)
    rows = []
    for timestep in range(tmax):
        current = prefix_logits[:, timestep]
        normalized_t = current.new_full((n, 1), (timestep + 1) / tmax)
        if mode == "current_logits":
            rows.append(torch.cat([current, probabilities[:, timestep], confidence[:, timestep, None],
                                   entropy[:, timestep, None], margin[:, timestep, None], normalized_t], dim=1))
            continue
        if mode != "logit_history":
            raise ValueError(f"Unknown feature mode: {mode}")
        observed = prefix_logits[:, : timestep + 1]
        padded_logits = current.new_zeros((n, tmax, classes))
        padded_logits[:, : timestep + 1] = observed
        valid = current.new_zeros((n, tmax))
        valid[:, : timestep + 1] = 1.0
        previous = prefix_logits[:, timestep - 1] if timestep >= 1 else torch.zeros_like(current)
        delta = current - previous if timestep >= 1 else torch.zeros_like(current)
        second = (current - 2 * previous + prefix_logits[:, timestep - 2]) if timestep >= 2 else torch.zeros_like(current)
        availability = current.new_tensor([float(timestep >= 1), float(timestep >= 2)]).expand(n, -1)
        mean = observed.mean(dim=1)
        std = observed.std(dim=1, unbiased=False)
        history_stats = current.new_zeros((n, 3, tmax))
        history_stats[:, 0, : timestep + 1] = confidence[:, : timestep + 1]
        history_stats[:, 1, : timestep + 1] = entropy[:, : timestep + 1]
        history_stats[:, 2, : timestep + 1] = margin[:, : timestep + 1]
        predictions = observed.argmax(dim=-1)
        switches = predictions[:, 1:].ne(predictions[:, :-1]).sum(dim=1, keepdim=True).float()
        persistence = current.new_ones((n, 1))
        if timestep >= 1:
            same = predictions.eq(predictions[:, -1:])
            for offset in range(1, timestep + 1):
                persistence += same[:, -(offset + 1)].float().unsqueeze(1) * (persistence == offset).float()
        cosine = torch.nn.functional.cosine_similarity(current, previous, dim=1).unsqueeze(1) if timestep >= 1 else current.new_zeros((n, 1))
        rows.append(torch.cat([padded_logits.flatten(1), valid, current, previous, delta, second,
                               availability, mean, std, history_stats.flatten(1), switches,
                               persistence, cosine, normalized_t], dim=1))
    return torch.stack(rows, dim=1)


def fit_feature_normalization(features: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
    flat = features.reshape(-1, features.shape[-1])
    if mask is not None:
        flat = flat[mask.reshape(-1).bool()]
    if flat.shape[0] == 0:
        raise ValueError("Cannot fit normalization on an empty feature set.")
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return {"feature_mean": mean, "feature_std": std}


def normalize_features(features: Tensor, statistics: dict[str, Tensor]) -> Tensor:
    return (features - statistics["feature_mean"]) / statistics["feature_std"]
