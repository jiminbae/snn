"""Transition-selector oracles, target-free features, gates, and metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from export_confirmatory_prefix_trajectories import MAX_PREFIX_CORRECT_COUNT_DRIFT
from utils.prefix_metrics import first_correct_timestep, stable_correct_timestep

FEATURE_SCHEMA = [
    {"name": "previous_top1_confidence", "description": "Top-1 probability of the accepted previous candidate.", "uses_target": False},
    {"name": "previous_top1_margin", "description": "Top-1 minus top-2 probability of the accepted previous candidate.", "uses_target": False},
    {"name": "previous_normalized_entropy", "description": "Entropy of the accepted previous distribution divided by log(C).", "uses_target": False},
    {"name": "new_top1_confidence", "description": "Top-1 probability of the new raw candidate.", "uses_target": False},
    {"name": "new_top1_margin", "description": "Top-1 minus top-2 probability of the new raw candidate.", "uses_target": False},
    {"name": "new_normalized_entropy", "description": "Entropy of the new raw distribution divided by log(C).", "uses_target": False},
    {"name": "same_top1_class", "description": "Whether previous and new candidates predict the same class.", "uses_target": False},
    {"name": "new_probability_of_previous_class", "description": "New distribution probability assigned to the previous top-1 class.", "uses_target": False},
    {"name": "previous_probability_of_new_class", "description": "Previous distribution probability assigned to the new top-1 class.", "uses_target": False},
    {"name": "top1_confidence_delta", "description": "New minus previous top-1 confidence.", "uses_target": False},
    {"name": "top1_margin_delta", "description": "New minus previous top-1 margin.", "uses_target": False},
    {"name": "js_divergence", "description": "Jensen-Shannon divergence between candidate distributions.", "uses_target": False},
    {"name": "normalized_timestep", "description": "One-based destination timestep divided by T.", "uses_target": False},
]
FEATURE_NAMES = [row["name"] for row in FEATURE_SCHEMA]
TRANSITION_TYPES = ("C_TO_C", "C_TO_W", "W_TO_C", "W_TO_W")


def _candidate_stats(logits: Tensor) -> dict[str, Tensor]:
    probability = logits.float().softmax(dim=-1)
    top = probability.topk(k=2, dim=-1)
    prediction = top.indices[..., 0]
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "probability": probability,
        "prediction": prediction,
        "confidence": top.values[..., 0],
        "margin": top.values[..., 0] - top.values[..., 1],
        "entropy": entropy,
    }


def transition_type(previous_correct: Tensor, new_correct: Tensor) -> Tensor:
    return previous_correct.long() * 2 + new_correct.long()


def transition_type_names(codes: Tensor) -> list[str]:
    names = ("W_TO_W", "W_TO_C", "C_TO_W", "C_TO_C")
    return [names[int(code)] for code in codes.reshape(-1)]


def validate_final_ce_trajectories(trajectories: list[dict[str, Any]]) -> None:
    if len(trajectories) != 3:
        raise ValueError("Exactly three final_ce trajectories are required")
    expected_seeds = (3, 4, 5)
    reference = trajectories[0]
    for trajectory, seed in zip(trajectories, expected_seeds):
        logits = trajectory["prefix_logits"]
        targets = trajectory["targets"]
        indices = trajectory["sample_index"]
        if trajectory.get("method") != "final_ce" or int(trajectory.get("seed", -1)) != seed:
            raise ValueError(f"Expected final_ce seed {seed}")
        if logits.ndim != 3 or tuple(logits.shape[1:]) != (8, 10):
            raise ValueError(f"seed {seed}: prefix shape must be [N,8,10]")
        n = logits.shape[0]
        if tuple(targets.shape) != (n,) or tuple(indices.shape) != (n,):
            raise ValueError(f"seed {seed}: targets/sample_index shape mismatch")
        predictions = logits.argmax(dim=-1)
        if not torch.equal(trajectory["predictions"], predictions):
            raise ValueError(f"seed {seed}: predictions mismatch")
        if not torch.equal(trajectory["correct"].bool(), predictions.eq(targets[:, None])):
            raise ValueError(f"seed {seed}: correct mask mismatch")
        if not torch.equal(indices, torch.arange(n)):
            raise ValueError(f"seed {seed}: sample_index is not arange(N)")
        validation = trajectory.get("validation")
        if validation and (
            not validation.get("final_correct_count_exact", False)
            or validation.get("max_abs_prefix_correct_count_drift", MAX_PREFIX_CORRECT_COUNT_DRIFT + 1)
            > MAX_PREFIX_CORRECT_COUNT_DRIFT
        ):
            raise ValueError(f"seed {seed}: stored validation metadata failed")
        if seed != expected_seeds[0]:
            for key in ("targets", "sample_index"):
                if not torch.equal(reference[key], trajectory[key]):
                    raise ValueError(f"seed {seed}: {key} differs across seeds")
            if tuple(reference["prefix_logits"].shape) != tuple(logits.shape):
                raise ValueError(f"seed {seed}: shape differs across seeds")


def build_derived_trajectory(
    base_trajectory: dict[str, Any], prefix_logits: Tensor, method: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tuple(prefix_logits.shape) != tuple(base_trajectory["prefix_logits"].shape):
        raise ValueError("Derived logits shape differs from base trajectory")
    targets = base_trajectory["targets"].long()
    stats = _candidate_stats(prefix_logits)
    true_probability = stats["probability"].gather(
        -1, targets[:, None, None].expand(-1, prefix_logits.shape[1], 1)
    ).squeeze(-1)
    return {
        "prefix_logits": prefix_logits.float(),
        "targets": targets.clone(),
        "predictions": stats["prediction"].long(),
        "correct": stats["prediction"].eq(targets[:, None]),
        "true_class_probability": true_probability,
        "top1_confidence": stats["confidence"],
        "top1_margin": stats["margin"],
        "sample_index": base_trajectory["sample_index"].clone(),
        "method": method,
        "seed": int(base_trajectory["seed"]),
        "config": base_trajectory.get("config", {}).copy(),
        "source_method": base_trajectory.get("method"),
        "selector_metadata": dict(metadata or {}),
    }


def transition_features(previous_logits: Tensor, new_logits: Tensor, to_t: int, tmax: int) -> tuple[Tensor, dict[str, Tensor]]:
    previous = _candidate_stats(previous_logits)
    new = _candidate_stats(new_logits)
    c = previous_logits.shape[-1]
    previous_probability = previous["probability"]
    new_probability = new["probability"]
    mixture = 0.5 * (previous_probability + new_probability)
    js = 0.5 * (
        (previous_probability * (previous_probability.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(-1)
        + (new_probability * (new_probability.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(-1)
    )
    new_of_previous = new_probability.gather(-1, previous["prediction"][:, None]).squeeze(-1)
    previous_of_new = previous_probability.gather(-1, new["prediction"][:, None]).squeeze(-1)
    values = {
        "previous_top1_confidence": previous["confidence"],
        "previous_top1_margin": previous["margin"],
        "previous_normalized_entropy": previous["entropy"] / math.log(c),
        "new_top1_confidence": new["confidence"],
        "new_top1_margin": new["margin"],
        "new_normalized_entropy": new["entropy"] / math.log(c),
        "same_top1_class": previous["prediction"].eq(new["prediction"]).float(),
        "new_probability_of_previous_class": new_of_previous,
        "previous_probability_of_new_class": previous_of_new,
        "top1_confidence_delta": new["confidence"] - previous["confidence"],
        "top1_margin_delta": new["margin"] - previous["margin"],
        "js_divergence": js,
        "normalized_timestep": previous_logits.new_full((previous_logits.shape[0],), float(to_t) / tmax),
    }
    return torch.stack([values[name] for name in FEATURE_NAMES], dim=1), {**previous, **{f"new_{k}": v for k, v in new.items()}}


def oracle_rollout(base: dict[str, Any], oracle: str) -> tuple[dict[str, Any], dict[str, Tensor]]:
    raw = base["prefix_logits"].float()
    targets = base["targets"].long()
    accepted = [raw[:, 0]]
    logs: dict[str, list[Tensor]] = {key: [] for key in (
        "sample_index", "from_t", "to_t", "previous_prediction", "new_prediction",
        "accepted_prediction", "previous_correct", "new_correct", "accepted_correct",
        "transition_type_code", "keep", "previous_top1_confidence", "previous_margin",
        "previous_entropy", "new_top1_confidence", "new_margin", "new_entropy",
        "same_top1_class", "new_probability_of_previous_class",
        "previous_probability_of_new_class", "js_divergence", "normalized_timestep",
    )}
    for t in range(1, raw.shape[1]):
        previous_logits = accepted[-1]
        new_logits = raw[:, t]
        features, stats = transition_features(previous_logits, new_logits, t + 1, raw.shape[1])
        previous_prediction = previous_logits.argmax(-1)
        new_prediction = new_logits.argmax(-1)
        previous_correct = previous_prediction.eq(targets)
        new_correct = new_prediction.eq(targets)
        if oracle == "oracle_block_destructive":
            keep = previous_correct & ~new_correct
        elif oracle == "oracle_best_candidate":
            previous_true = previous_logits.softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            new_true = new_logits.softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            keep = (previous_correct & ~new_correct) | (
                previous_correct.eq(new_correct) & (previous_true > new_true + 1e-12)
            )
        else:
            raise ValueError(f"Unknown oracle: {oracle}")
        chosen = torch.where(keep[:, None], previous_logits, new_logits)
        accepted.append(chosen)
        accepted_prediction = chosen.argmax(-1)
        code = transition_type(previous_correct, new_correct)
        row_values = {
            "sample_index": base["sample_index"],
            "from_t": torch.full_like(targets, t),
            "to_t": torch.full_like(targets, t + 1),
            "previous_prediction": previous_prediction,
            "new_prediction": new_prediction,
            "accepted_prediction": accepted_prediction,
            "previous_correct": previous_correct,
            "new_correct": new_correct,
            "accepted_correct": accepted_prediction.eq(targets),
            "transition_type_code": code,
            "keep": keep,
            "previous_top1_confidence": features[:, 0],
            "previous_margin": features[:, 1],
            "previous_entropy": features[:, 2] * math.log(raw.shape[-1]),
            "new_top1_confidence": features[:, 3],
            "new_margin": features[:, 4],
            "new_entropy": features[:, 5] * math.log(raw.shape[-1]),
            "same_top1_class": features[:, 6].bool(),
            "new_probability_of_previous_class": features[:, 7],
            "previous_probability_of_new_class": features[:, 8],
            "js_divergence": features[:, 11],
            "normalized_timestep": features[:, 12],
        }
        for key, value in row_values.items():
            logs[key].append(value.detach().cpu())
    stacked = {key: torch.stack(value, dim=1) for key, value in logs.items()}
    stacked["oracle"] = oracle
    stacked["seed"] = int(base["seed"])
    trajectory = build_derived_trajectory(
        base, torch.stack(accepted, dim=1), oracle,
        {"analysis_type": "label-informed upper-bound feasibility analysis", "oracle": oracle},
    )
    return trajectory, stacked


def oracle_training_examples(base: dict[str, Any]) -> dict[str, Tensor]:
    oracle, log = oracle_rollout(base, "oracle_block_destructive")
    raw = base["prefix_logits"].float()
    accepted = oracle["prefix_logits"]
    features = []
    for t in range(1, raw.shape[1]):
        feature, _ = transition_features(accepted[:, t - 1], raw[:, t], t + 1, raw.shape[1])
        features.append(feature)
    return {
        "features": torch.stack(features, dim=1),
        "target": log["keep"].bool(),
        "transition_type_code": log["transition_type_code"].long(),
        "sample_index": base["sample_index"].clone(),
        "seed": torch.full((raw.shape[0], raw.shape[1] - 1), int(base["seed"]), dtype=torch.long),
    }


def deterministic_fold_assignment(sample_indices: Tensor, fold_count: int = 5, seed: int = 2026) -> dict[int, int]:
    unique = torch.unique(sample_indices.cpu().long(), sorted=True)
    generator = torch.Generator().manual_seed(seed)
    permutation = unique[torch.randperm(unique.numel(), generator=generator)]
    return {int(sample): int(position % fold_count) for position, sample in enumerate(permutation.tolist())}


def standardize_train_validation(train: Tensor, validation: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = train.mean(0)
    std = train.std(0, unbiased=False)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return (train - mean) / std, (validation - mean) / std, mean, std


class LinearGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__(); self.network = nn.Linear(input_dim, 1)
    def forward(self, x: Tensor) -> Tensor: return self.network(x).squeeze(-1)


class MLP16Gate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__(); self.network = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x: Tensor) -> Tensor: return self.network(x).squeeze(-1)


def train_gate(
    model_name: str, train_features: Tensor, train_targets: Tensor, *,
    epochs: int = 300, batch_size: int = 4096, seed: int = 2026,
    learning_rate: float = 1e-2, weight_decay: float = 1e-4, device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    positives = int(train_targets.sum())
    negatives = int((~train_targets.bool()).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("insufficient_support: training fold requires both KEEP and SWITCH examples")
    torch.manual_seed(seed)
    cls = LinearGate if model_name == "linear_gate" else MLP16Gate if model_name == "mlp16_gate" else None
    if cls is None: raise ValueError(model_name)
    model = cls(train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = train_features.float(); y = train_targets.float()
    generator = torch.Generator().manual_seed(seed)
    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            index = order[start:start + batch_size]
            xb = x[index].to(device); yb = y[index].to(device)
            weights = torch.where(yb.bool(), 0.5 / positives, 0.5 / negatives)
            loss = (F.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * weights).sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            final_loss = float(loss.detach().cpu())
    return model.cpu().eval(), {
        "positive_count": positives, "negative_count": negatives,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "training_seed": seed, "final_batch_loss": final_loss,
    }


def action_metrics(probability: Tensor, target: Tensor, transition_codes: Tensor, threshold: float = 0.5) -> dict[str, float]:
    predicted = probability >= threshold
    target = target.bool(); transition_codes = transition_codes.long()
    tp = int((predicted & target).sum()); fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum()); tn = int((~predicted & ~target).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    switch_recall = tn / (tn + fp) if tn + fp else float("nan")
    def conditional(mask: Tensor, event: Tensor) -> float:
        return float(event[mask].float().mean()) if mask.any() else float("nan")
    return {
        "example_count": int(target.numel()), "keep_count": int(target.sum()),
        "keep_prevalence": float(target.float().mean()), "keep_precision": precision,
        "keep_recall": recall, "keep_f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
        "switch_recall": switch_recall, "balanced_accuracy": 0.5 * (recall + switch_recall),
        "C_TO_W_keep_recall": conditional(transition_codes == 2, predicted),
        "W_TO_C_false_keep_rate": conditional(transition_codes == 1, predicted),
        "C_TO_C_false_keep_rate": conditional(transition_codes == 3, predicted),
        "W_TO_W_false_keep_rate": conditional(transition_codes == 0, predicted),
    }


def closed_loop_rollout(
    base: dict[str, Any], probability_fn: Callable[[Tensor], Tensor], threshold: float,
    method: str, sample_mask: Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    if sample_mask is None: sample_mask = torch.ones(base["targets"].shape[0], dtype=torch.bool)
    subset = {key: value[sample_mask] if isinstance(value, Tensor) and value.shape[:1] == sample_mask.shape else value for key, value in base.items()}
    raw = subset["prefix_logits"].float(); accepted = [raw[:, 0]]; keeps=[]; probabilities=[]; codes=[]
    targets = subset["targets"]
    for t in range(1, raw.shape[1]):
        feature, _ = transition_features(accepted[-1], raw[:, t], t + 1, raw.shape[1])
        probability = probability_fn(feature).detach().cpu()
        keep = probability >= threshold
        previous_correct = accepted[-1].argmax(-1).eq(targets)
        new_correct = raw[:, t].argmax(-1).eq(targets)
        accepted.append(torch.where(keep[:, None], accepted[-1], raw[:, t]))
        keeps.append(keep); probabilities.append(probability); codes.append(transition_type(previous_correct, new_correct))
    trajectory = build_derived_trajectory(subset, torch.stack(accepted, 1), method, {"threshold": threshold})
    return trajectory, {"keep": torch.stack(keeps, 1), "probability": torch.stack(probabilities, 1), "transition_type_code": torch.stack(codes, 1)}


def trajectory_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    correct = trajectory["correct"].bool(); n, tmax = correct.shape
    c2w = correct[:, :-1] & ~correct[:, 1:]
    w2c = ~correct[:, :-1] & correct[:, 1:]
    correct_support = int(correct[:, :-1].sum()); wrong_support = int((~correct[:, :-1]).sum())
    curve = correct.float().mean(0) * 100.0
    first = first_correct_timestep(trajectory["prefix_logits"], trajectory["targets"])
    stable = stable_correct_timestep(trajectory["prefix_logits"], trajectory["targets"])
    sentinel = tmax + 1
    return {
        "prefix_accuracy_curve": curve.tolist(), "final_accuracy": float(curve[-1]),
        "mean_prefix_accuracy": float(curve.mean()), "minimum_prefix_accuracy": float(curve.min()),
        "normalized_prefix_auc": float(curve.mean()), "normalized_prefix_auc_definition": "arithmetic mean of prefix accuracy percentages",
        "ever_regressed_fraction": float(c2w.any(1).float().mean()), "ever_recovered_fraction": float(w2c.any(1).float().mean()),
        "population_destructive_rate": 100.0 * int(c2w.sum()) / (n * (tmax - 1)),
        "conditional_regression_rate": 100.0 * int(c2w.sum()) / correct_support if correct_support else float("nan"),
        "population_beneficial_rate": 100.0 * int(w2c.sum()) / (n * (tmax - 1)),
        "conditional_recovery_rate": 100.0 * int(w2c.sum()) / wrong_support if wrong_support else float("nan"),
        "correct_to_wrong_count": int(c2w.sum()), "wrong_to_correct_count": int(w2c.sum()),
        "first_correct_timestep_mean": float(first[first < sentinel].float().mean()) if (first < sentinel).any() else float("nan"),
        "stable_correct_timestep_mean": float(stable[stable < sentinel].float().mean()) if (stable < sentinel).any() else float("nan"),
    }


def paired_metrics(candidate: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    cc = candidate["correct"].bool(); rc = raw["correct"].bool()
    common_correct = cc[:, :-1] & rc[:, :-1]
    common_wrong = ~cc[:, :-1] & ~rc[:, :-1]
    candidate_reg = int((common_correct & ~cc[:, 1:]).sum()); raw_reg = int((common_correct & ~rc[:, 1:]).sum())
    candidate_rec = int((common_wrong & cc[:, 1:]).sum()); raw_rec = int((common_wrong & rc[:, 1:]).sum())
    reg_support = int(common_correct.sum()); rec_support = int(common_wrong.sum())
    def timing(kind):
        fn = first_correct_timestep if kind == "first" else stable_correct_timestep
        a=fn(candidate["prefix_logits"], candidate["targets"]); b=fn(raw["prefix_logits"], raw["targets"]); sentinel=cc.shape[1]+1
        valid=(a<sentinel)&(b<sentinel); d=(a[valid]-b[valid]).float()
        return {
            f"{kind}_delay_mean": float(d.mean()) if d.numel() else float("nan"),
            f"{kind}_delay_median": float(d.median()) if d.numel() else float("nan"),
            f"{kind}_delay_q25": float(torch.quantile(d,.25)) if d.numel() else float("nan"),
            f"{kind}_delay_q75": float(torch.quantile(d,.75)) if d.numel() else float("nan"),
            f"{kind}_fraction_delayed": float((d>0).float().mean()) if d.numel() else float("nan"),
            f"{kind}_fraction_accelerated": float((d<0).float().mean()) if d.numel() else float("nan"),
            f"{kind}_fraction_unchanged": float((d==0).float().mean()) if d.numel() else float("nan"),
        }
    base=trajectory_metrics(candidate); raw_base=trajectory_metrics(raw)
    return {
        **base,
        "final_accuracy_change": base["final_accuracy"]-raw_base["final_accuracy"],
        "mean_prefix_accuracy_change": base["mean_prefix_accuracy"]-raw_base["mean_prefix_accuracy"],
        "minimum_prefix_accuracy_change": base["minimum_prefix_accuracy"]-raw_base["minimum_prefix_accuracy"],
        "normalized_prefix_auc_change": base["normalized_prefix_auc"]-raw_base["normalized_prefix_auc"],
        "matched_common_correct_support": reg_support,
        "candidate_matched_regression_count": candidate_reg, "raw_matched_regression_count": raw_reg,
        "micro_matched_regression_difference": 100.0*(candidate_reg-raw_reg)/reg_support if reg_support else float("nan"),
        "matched_common_wrong_support": rec_support,
        "candidate_matched_recovery_count": candidate_rec, "raw_matched_recovery_count": raw_rec,
        "seed_recovery_preservation_ratio": candidate_rec/raw_rec if raw_rec else float("nan"),
        **timing("first"), **timing("stable"),
    }


def intervention_metrics(action: Tensor, codes: Tensor) -> dict[str, float]:
    keep=action.bool(); result={"overall_keep_rate":float(keep.float().mean()), "keep_count":int(keep.sum()), "action_count":int(keep.numel())}
    for code,name in enumerate(("W_TO_W","W_TO_C","C_TO_W","C_TO_C")):
        mask=codes==code
        result[f"keep_rate_{name}"]=float(keep[mask].float().mean()) if mask.any() else float("nan")
        result[f"switch_rate_{name}"]=float((~keep[mask]).float().mean()) if mask.any() else float("nan")
        result[f"support_{name}"]=int(mask.sum())
    return result


def aggregate_seed_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric=[key for key,value in rows[0].items() if isinstance(value,(int,float)) and key != "seed"]
    result={}
    for key in numeric:
        vals=[float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        if vals:
            result[f"{key}_mean"]=sum(vals)/len(vals)
            result[f"{key}_sample_std"]=(sum((x-sum(vals)/len(vals))**2 for x in vals)/(len(vals)-1))**.5 if len(vals)>1 else 0.0
    cand=sum(row.get("candidate_matched_recovery_count",0) for row in rows); raw=sum(row.get("raw_matched_recovery_count",0) for row in rows)
    result["pooled_matched_recovery_preservation"]=cand/raw if raw else float("nan")
    result["regression_improved_seed_count"]=sum(row.get("micro_matched_regression_difference",0)<0 for row in rows)
    return result


def guardrail_pass(rows: list[dict[str, Any]], required_improvement: float) -> bool:
    agg=aggregate_seed_metrics(rows)
    return (
        agg["regression_improved_seed_count"] >= 2
        and agg.get("micro_matched_regression_difference_mean", float("inf")) <= -required_improvement
        and agg.get("pooled_matched_recovery_preservation", 0) >= .90
        and all(row.get("seed_recovery_preservation_ratio",0) >= .80 for row in rows)
        and agg.get("final_accuracy_change_mean", float("-inf")) >= -.10
        and agg.get("mean_prefix_accuracy_change_mean", float("-inf")) >= -1.0
        and agg.get("minimum_prefix_accuracy_change_mean", float("-inf")) >= -3.0
        and agg.get("first_delay_mean_mean", float("inf")) <= .10
    )


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier=[]
    for row in rows:
        dominated=False
        for other in rows:
            if other is row: continue
            better=(
                other["regression_improvement"] >= row["regression_improvement"]
                and other["pooled_matched_recovery_preservation"] >= row["pooled_matched_recovery_preservation"]
                and other["mean_prefix_accuracy_change"] >= row["mean_prefix_accuracy_change"]
            )
            strict=(
                other["regression_improvement"] > row["regression_improvement"]
                or other["pooled_matched_recovery_preservation"] > row["pooled_matched_recovery_preservation"]
                or other["mean_prefix_accuracy_change"] > row["mean_prefix_accuracy_change"]
            )
            if better and strict: dominated=True; break
        if not dominated: frontier.append(row)
    return frontier
