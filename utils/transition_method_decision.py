"""Utilities for the transition-method branch decision experiment.

The experiment compares target-free output selectors, low-capacity causal
filters, and a hidden-state adaptive causal evidence decoder (RC-CED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from utils.stopping_policy_evaluation import binary_ranking_metrics
from utils.transition_selector import (
    FEATURE_SCHEMA,
    aggregate_seed_metrics,
    build_derived_trajectory,
    oracle_training_examples,
    paired_metrics,
    train_gate,
    transition_features,
    transition_type,
)


DIRECTION_MODELS = ("linear_direction_ranker", "mlp16_direction_ranker")
FALSE_VETO_BUDGETS = (0.025, 0.05, 0.10)
HIDDEN_FEATURE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FilterSpec:
    """One predeclared simple-filter candidate."""

    method: str
    parameters: tuple[tuple[str, float], ...] = ()
    complexity_rank: int = 0
    control_only: bool = False

    @property
    def parameter_dict(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def candidate_id(self) -> str:
        if not self.parameters:
            return self.method
        suffix = "_".join(f"{key}-{value:g}" for key, value in self.parameters)
        return f"{self.method}__{suffix}"


def simple_filter_specs() -> tuple[FilterSpec, ...]:
    """Return the fixed candidate grid; controls are never eligible to win."""

    rows: list[FilterSpec] = [
        FilterSpec("raw_final_ce", control_only=True),
        FilterSpec("arithmetic_probability_ema", (("alpha", 1.0),), control_only=True),
        FilterSpec("geometric_probability_ema", (("alpha", 1.0),), control_only=True),
        FilterSpec("clipped_logit_innovation", (("clip", math.inf),), control_only=True),
    ]
    rows.extend(
        FilterSpec("arithmetic_probability_ema", (("alpha", alpha),), 1)
        for alpha in (0.25, 0.50, 0.75)
    )
    rows.extend(
        FilterSpec("geometric_probability_ema", (("alpha", alpha),), 1)
        for alpha in (0.25, 0.50, 0.75)
    )
    rows.extend(
        FilterSpec("clipped_logit_innovation", (("clip", clip),), 1)
        for clip in (0.25, 0.50, 1.0, 2.0)
    )
    rows.extend(
        FilterSpec("entropy_adaptive_ema", (("alpha_min", alpha_min),), 2)
        for alpha_min in (0.10, 0.25, 0.50)
    )
    rows.extend(
        (
            FilterSpec(
                "confidence_hysteresis",
                (("previous_confidence", 0.70), ("new_confidence", 0.60)),
                2,
            ),
            FilterSpec("max_hold_one", (), 2),
        )
    )
    return tuple(rows)


@dataclass(frozen=True)
class DecisionGuardrails:
    """Prespecified safety and materiality requirements for a candidate lane."""

    required_regression_improvement_pp: float
    regression_improved_seed_count_min: int = 2
    ever_regressed_improved_seed_count_min: int = 2
    pooled_recovery_min: float = 0.95
    per_seed_recovery_min: float = 0.90
    final_accuracy_change_pp_min: float = -0.10
    mean_prefix_accuracy_change_pp_min: float = -1.0
    minimum_prefix_accuracy_change_pp_min: float = -3.0
    first_correct_delay_mean_max: float = 0.10
    minimum_pooled_raw_regression_count: int = 100
    minimum_pooled_raw_recovery_count: int = 100

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def split_payload_to_trajectory(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    """Normalize a split-export payload to the transition-selector schema."""

    logits = payload["prefix_logits"].float()
    targets = payload["targets"].long()
    probabilities = logits.softmax(-1)
    top2 = probabilities.topk(k=min(2, logits.shape[-1]), dim=-1).values
    predictions = logits.argmax(-1)
    sample_index = payload.get("sample_index", payload.get("sample_indices"))
    if sample_index is None:
        raise ValueError("split payload is missing sample_indices")
    trajectory = {
        "prefix_logits": logits,
        "targets": targets,
        "predictions": predictions,
        "correct": predictions.eq(targets[:, None]),
        "true_class_probability": probabilities.gather(
            -1, targets[:, None, None].expand(-1, logits.shape[1], 1)
        ).squeeze(-1),
        "top1_confidence": top2[..., 0],
        "top1_margin": top2[..., 0] - top2[..., 1],
        "sample_index": torch.as_tensor(sample_index).long(),
        "method": "final_ce",
        "seed": int(seed),
        "config": dict(payload.get("metadata", {})),
        "split": payload.get("split"),
    }
    if "hidden_features" in payload:
        trajectory["hidden_features"] = payload["hidden_features"].float()
        trajectory["hidden_feature_metadata"] = dict(
            payload.get("hidden_feature_metadata", {})
        )
    return trajectory


def validate_hidden_trajectory(
    trajectory: dict[str, Any], *, reference: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a target-free, causal hidden representation."""

    if "hidden_features" not in trajectory:
        raise ValueError("hidden_features are missing")
    hidden = trajectory["hidden_features"]
    logits = trajectory["prefix_logits"]
    if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(logits.shape[:2]):
        raise ValueError("hidden_features must have shape [N,T,H] aligned to logits")
    if hidden.shape[-1] < 1 or not torch.isfinite(hidden).all():
        raise ValueError("hidden_features must be finite and non-empty")
    metadata = trajectory.get("hidden_feature_metadata", {})
    if int(metadata.get("format_version", -1)) != HIDDEN_FEATURE_FORMAT_VERSION:
        raise ValueError("unsupported hidden feature format version")
    if metadata.get("uses_target") is not False or metadata.get("causal") is not True:
        raise ValueError("hidden features must be explicitly target-free and causal")
    if int(metadata.get("dimension", -1)) != hidden.shape[-1]:
        raise ValueError("hidden feature dimension metadata mismatch")
    forbidden = ("target", "label", "correct", "true_class")
    schema_text = str(metadata.get("groups", "")).lower()
    if any(token in schema_text for token in forbidden):
        raise ValueError("hidden feature schema appears target-dependent")
    if reference is not None and "hidden_features" in reference:
        if hidden.shape[-1] != reference["hidden_features"].shape[-1]:
            raise ValueError("hidden dimension differs across seeds")
        if metadata != reference.get("hidden_feature_metadata", {}):
            raise ValueError("hidden feature schema differs across seeds")
    return {
        "status": "valid",
        "samples": hidden.shape[0],
        "timesteps": hidden.shape[1],
        "dimension": hidden.shape[2],
    }


def validate_split_trajectories(
    trajectories_by_split: dict[str, list[dict[str, Any]]],
    *,
    require_hidden: bool = False,
) -> dict[str, Any]:
    """Validate split provenance, alignment, and optional hidden features."""

    required_splits = ("train", "val", "test")
    for split in required_splits:
        if split not in trajectories_by_split:
            raise ValueError(f"missing split: {split}")
        rows = trajectories_by_split[split]
        if len(rows) != 3 or sorted(int(row["seed"]) for row in rows) != [3, 4, 5]:
            raise ValueError(f"{split}: expected final_ce seeds 3, 4, and 5")
        reference = rows[0]
        for row in rows:
            logits = row["prefix_logits"]
            targets = row["targets"]
            indices = row["sample_index"]
            if row.get("split") != split:
                raise ValueError(f"seed {row['seed']}: split metadata mismatch")
            if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] < 2:
                raise ValueError("prefix_logits must have shape [N,T,C]")
            if tuple(targets.shape) != (logits.shape[0],) or tuple(indices.shape) != (
                logits.shape[0],
            ):
                raise ValueError("targets/sample indices do not align with logits")
            if not torch.isfinite(logits).all():
                raise ValueError("prefix_logits contain non-finite values")
            if not torch.equal(row["predictions"], logits.argmax(-1)):
                raise ValueError("predictions do not match prefix_logits")
            if not torch.equal(row["correct"], row["predictions"].eq(targets[:, None])):
                raise ValueError("correct mask mismatch")
            if not torch.equal(reference["sample_index"], indices) or not torch.equal(
                reference["targets"], targets
            ):
                raise ValueError(f"{split}: targets/sample order differ across seeds")
            if tuple(reference["prefix_logits"].shape) != tuple(logits.shape):
                raise ValueError(f"{split}: prefix shape differs across seeds")
            if require_hidden:
                validate_hidden_trajectory(row, reference=reference)
    train_ids = set(trajectories_by_split["train"][0]["sample_index"].tolist())
    val_ids = set(trajectories_by_split["val"][0]["sample_index"].tolist())
    if train_ids & val_ids:
        raise ValueError("train and validation sample indices overlap")
    return {
        "status": "passed",
        "splits": {
            split: {
                "samples": int(trajectories_by_split[split][0]["targets"].numel()),
                "shape": list(trajectories_by_split[split][0]["prefix_logits"].shape),
            }
            for split in required_splits
        },
        "hidden_features_present": require_hidden,
    }


def _center_logits(logits: Tensor) -> Tensor:
    return logits.float() - logits.float().mean(dim=-1, keepdim=True)


def _probability_logits(probability: Tensor) -> Tensor:
    log_probability = probability.clamp_min(1e-12).log()
    return log_probability - log_probability.mean(dim=-1, keepdim=True)


def simple_filter_rollout(
    base: dict[str, Any], spec: FilterSpec
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Apply one causal simple filter to a raw prefix trajectory."""

    raw = base["prefix_logits"].float()
    n, tmax, classes = raw.shape
    parameters = spec.parameter_dict
    states = [raw[:, 0]]
    alpha_log: list[Tensor] = []
    hold_log: list[Tensor] = []

    if spec.method == "raw_final_ce":
        states = [raw[:, t] for t in range(tmax)]
    elif spec.method == "arithmetic_probability_ema":
        alpha = float(parameters["alpha"])
        probability = raw[:, 0].softmax(-1)
        states = [_probability_logits(probability)]
        for t in range(1, tmax):
            probability = (1.0 - alpha) * probability + alpha * raw[:, t].softmax(-1)
            states.append(_probability_logits(probability))
            alpha_log.append(raw.new_full((n,), alpha))
    elif spec.method == "geometric_probability_ema":
        alpha = float(parameters["alpha"])
        log_probability = raw[:, 0].log_softmax(-1)
        states = [_center_logits(log_probability)]
        for t in range(1, tmax):
            log_probability = (1.0 - alpha) * log_probability + alpha * raw[:, t].log_softmax(-1)
            log_probability = log_probability - log_probability.logsumexp(-1, keepdim=True)
            states.append(_center_logits(log_probability))
            alpha_log.append(raw.new_full((n,), alpha))
    elif spec.method == "clipped_logit_innovation":
        clip = float(parameters["clip"])
        centered = _center_logits(raw)
        state = centered[:, 0]
        states = [state]
        for t in range(1, tmax):
            innovation = centered[:, t] - centered[:, t - 1]
            if math.isfinite(clip):
                innovation = innovation.clamp(-clip, clip)
            state = _center_logits(state + innovation)
            states.append(state)
    elif spec.method == "entropy_adaptive_ema":
        alpha_min = float(parameters["alpha_min"])
        probability = raw[:, 0].softmax(-1)
        states = [_probability_logits(probability)]
        for t in range(1, tmax):
            new_probability = raw[:, t].softmax(-1)
            entropy = -(new_probability * new_probability.clamp_min(1e-12).log()).sum(-1)
            reliability = (1.0 - entropy / math.log(classes)).clamp(0.0, 1.0)
            alpha = alpha_min + (1.0 - alpha_min) * reliability
            probability = (1.0 - alpha[:, None]) * probability + alpha[:, None] * new_probability
            states.append(_probability_logits(probability))
            alpha_log.append(alpha)
    elif spec.method == "confidence_hysteresis":
        previous_threshold = float(parameters["previous_confidence"])
        new_threshold = float(parameters["new_confidence"])
        for t in range(1, tmax):
            previous = states[-1]
            new = raw[:, t]
            previous_probability = previous.softmax(-1)
            new_probability = new.softmax(-1)
            class_changed = previous.argmax(-1).ne(new.argmax(-1))
            keep = (
                class_changed
                & previous_probability.max(-1).values.ge(previous_threshold)
                & new_probability.max(-1).values.le(new_threshold)
            )
            states.append(torch.where(keep[:, None], previous, new))
            hold_log.append(keep)
    elif spec.method == "max_hold_one":
        held_last_step = torch.zeros(n, dtype=torch.bool, device=raw.device)
        for t in range(1, tmax):
            previous = states[-1]
            new = raw[:, t]
            class_changed = previous.argmax(-1).ne(new.argmax(-1))
            keep = class_changed & ~held_last_step
            states.append(torch.where(keep[:, None], previous, new))
            held_last_step = keep
            hold_log.append(keep)
    else:
        raise ValueError(f"unknown filter: {spec.method}")

    logits = torch.stack(states, dim=1)
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{spec.candidate_id} produced non-finite logits")
    trajectory = build_derived_trajectory(
        base,
        logits,
        spec.candidate_id,
        {
            "family": "simple_filter",
            "filter": spec.method,
            "parameters": parameters,
            "causal": True,
            "target_free": True,
        },
    )
    intervention = trajectory["predictions"][:, 1:].ne(base["predictions"][:, 1:])
    logs = {"intervention": intervention}
    if alpha_log:
        logs["alpha"] = torch.stack(alpha_log, dim=1)
    if hold_log:
        logs["keep"] = torch.stack(hold_log, dim=1)
    return trajectory, logs


def build_direction_examples(base: dict[str, Any]) -> dict[str, Tensor]:
    """Build C-to-W (positive) versus W-to-C (negative) examples only."""

    examples = oracle_training_examples(base)
    codes = examples["transition_type_code"]
    directional = (codes == 1) | (codes == 2)
    sample = examples["sample_index"][:, None].expand_as(codes)
    timestep = torch.arange(2, codes.shape[1] + 2)[None, :].expand_as(codes)
    return {
        "features": examples["features"][directional],
        "target": codes[directional].eq(2),
        "transition_type_code": codes[directional],
        "sample_index": sample[directional],
        "to_t": timestep[directional],
        "seed": examples["seed"][directional],
    }


def train_direction_ranker(
    model_name: str,
    features: Tensor,
    targets: Tensor,
    **kwargs: Any,
) -> tuple[nn.Module, dict[str, Any]]:
    """Fit a fixed output-only ranker; its score is not a probability."""

    mapping = {
        "linear_direction_ranker": "linear_gate",
        "mlp16_direction_ranker": "mlp16_gate",
    }
    if model_name not in mapping:
        raise ValueError(model_name)
    model, metadata = train_gate(mapping[model_name], features, targets, **kwargs)
    metadata.update(
        {
            "model": model_name,
            "positive_definition": "C_TO_W",
            "negative_definition": "W_TO_C",
            "score_semantics": "rank score only; not a calibrated probability",
        }
    )
    return model, metadata


def threshold_at_false_veto_budget(scores: Tensor, targets: Tensor, budget: float) -> float:
    """Choose the lowest calibration threshold with W-to-C FPR in budget.

    Predictions use score >= threshold. Searching unique scores plus positive
    infinity handles ties conservatively and maximizes empirical positive
    recall among thresholds satisfying the constraint.
    """

    scores = scores.flatten().float()
    targets = targets.flatten().bool()
    if scores.numel() != targets.numel() or not 0.0 <= budget <= 1.0:
        raise ValueError("invalid scores/targets or false-veto budget")
    negatives = ~targets
    if not negatives.any():
        raise ValueError("insufficient_support: no W_TO_C calibration examples")
    negative_count = int(negatives.sum())
    allowed_false_vetoes = math.floor(budget * negative_count + 1e-12)
    candidates = [float("inf")] + sorted(set(float(v) for v in scores.tolist()))
    valid = [
        threshold
        for threshold in candidates
        if int((scores[negatives] >= threshold).sum()) <= allowed_false_vetoes
    ]
    return min(valid) if valid else float("inf")


def direction_metrics(scores: Tensor, targets: Tensor, threshold: float) -> dict[str, Any]:
    """Low-FPR direction metrics with C-to-W as the positive class."""

    scores = scores.flatten().float()
    targets = targets.flatten().bool()
    ranking = binary_ranking_metrics(scores, targets.float())
    predicted = scores >= threshold
    positives = targets
    negatives = ~targets
    return {
        **ranking,
        "support": int(targets.numel()),
        "C_TO_W_support": int(positives.sum()),
        "W_TO_C_support": int(negatives.sum()),
        "threshold": float(threshold),
        "C_TO_W_recall": float(predicted[positives].float().mean())
        if positives.any()
        else float("nan"),
        "W_TO_C_false_veto_rate": float(predicted[negatives].float().mean())
        if negatives.any()
        else float("nan"),
    }


def direction_selector_rollout(
    base: dict[str, Any],
    score_fn: Callable[[Tensor], Tensor],
    threshold: float,
    method: str,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Closed-loop rollout permitting vetoes only on top-1 class changes."""

    raw = base["prefix_logits"].float()
    targets = base["targets"].long()
    accepted = [raw[:, 0]]
    keeps: list[Tensor] = []
    scores: list[Tensor] = []
    codes: list[Tensor] = []
    for t in range(1, raw.shape[1]):
        previous = accepted[-1]
        new = raw[:, t]
        feature, _ = transition_features(previous, new, t + 1, raw.shape[1])
        score = score_fn(feature).detach().cpu().flatten()
        if score.shape != (raw.shape[0],):
            raise ValueError("direction score function returned an invalid shape")
        class_changed = previous.argmax(-1).ne(new.argmax(-1))
        keep = class_changed & score.ge(threshold)
        previous_correct = previous.argmax(-1).eq(targets)
        new_correct = new.argmax(-1).eq(targets)
        accepted.append(torch.where(keep[:, None], previous, new))
        keeps.append(keep)
        scores.append(score)
        codes.append(transition_type(previous_correct, new_correct))
    trajectory = build_derived_trajectory(
        base,
        torch.stack(accepted, dim=1),
        method,
        {
            "family": "output_only_selector",
            "threshold": threshold,
            "class_change_only": True,
            "target_free": True,
        },
    )
    return trajectory, {
        "keep": torch.stack(keeps, dim=1),
        "score": torch.stack(scores, dim=1),
        "transition_type_code": torch.stack(codes, dim=1),
    }


def first_regression_timestep(correct: Tensor) -> Tensor:
    """One-based destination timestep of the first C-to-W event, or T+1."""

    correct = correct.bool()
    events = correct[:, :-1] & ~correct[:, 1:]
    sentinel = correct.shape[1] + 1
    first = events.float().argmax(1) + 2
    return torch.where(events.any(1), first, torch.full_like(first, sentinel))


def regression_survival_rows(
    candidate: dict[str, Any],
    raw: dict[str, Any],
    *,
    family: str,
    candidate_id: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    """All-sample empirical survival from the first destructive transition."""

    candidate_first = first_regression_timestep(candidate["correct"])
    raw_first = first_regression_timestep(raw["correct"])
    rows = []
    for timestep in range(1, candidate["correct"].shape[1] + 1):
        rows.append(
            {
                "family": family,
                "candidate_id": candidate_id,
                "seed": seed,
                "split": split,
                "timestep": timestep,
                "sample_count": int(candidate_first.numel()),
                "candidate_regression_free_survival": float(
                    (candidate_first > timestep).float().mean()
                ),
                "raw_regression_free_survival": float(
                    (raw_first > timestep).float().mean()
                ),
            }
        )
    return rows


def regression_event_comparison(
    candidate: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    """Separate prevention, delay, induction, and never-event strata."""

    candidate_first = first_regression_timestep(candidate["correct"])
    raw_first = first_regression_timestep(raw["correct"])
    sentinel = candidate["correct"].shape[1] + 1
    candidate_event = candidate_first < sentinel
    raw_event = raw_first < sentinel
    both = candidate_event & raw_event
    delays = (candidate_first[both] - raw_first[both]).float()
    candidate_fraction = float(candidate_event.float().mean())
    raw_fraction = float(raw_event.float().mean())
    return {
        "candidate_ever_regressed_fraction": candidate_fraction,
        "raw_ever_regressed_fraction": raw_fraction,
        "ever_regressed_fraction_change": candidate_fraction - raw_fraction,
        "ever_regressed_reduction_pp": 100.0 * (raw_fraction - candidate_fraction),
        "regression_prevented_count": int((raw_event & ~candidate_event).sum()),
        "regression_both_event_count": int(both.sum()),
        "regression_induced_count": int((~raw_event & candidate_event).sum()),
        "regression_neither_event_count": int((~raw_event & ~candidate_event).sum()),
        "both_event_delay_mean": float(delays.mean()) if delays.numel() else float("nan"),
        "both_event_delay_median": float(delays.median()) if delays.numel() else float("nan"),
    }


def protection_followup_metrics(
    candidate: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    """Measure whether apparent protection merely postpones a wrong state."""

    candidate_correct = candidate["correct"].bool()
    raw_correct = raw["correct"].bool()
    protection = candidate_correct & ~raw_correct
    result: dict[str, Any] = {"protection_event_count": int(protection.sum())}
    tmax = candidate_correct.shape[1]
    for horizon in (1, 2, 3):
        support = 0
        future_wrong = 0
        for timestep in range(tmax):
            if timestep + horizon >= tmax:
                continue
            event = protection[:, timestep]
            support += int(event.sum())
            if event.any():
                window = candidate_correct[event, timestep + 1:timestep + horizon + 1]
                future_wrong += int((~window).any(1).sum())
        result[f"protection_followup_{horizon}_support"] = support
        result[f"wrong_within_{horizon}_after_protection"] = (
            future_wrong / support if support else float("nan")
        )

    stale_wrong = (~candidate_correct) & raw_correct
    run_lengths: list[int] = []
    for row in stale_wrong.tolist():
        length = 0
        for value in row:
            if value:
                length += 1
            elif length:
                run_lengths.append(length)
                length = 0
        if length:
            run_lengths.append(length)
    result.update(
        {
            "stale_wrong_run_count": len(run_lengths),
            "stale_wrong_run_mean": (
                sum(run_lengths) / len(run_lengths) if run_lengths else 0.0
            ),
            "stale_wrong_run_max": max(run_lengths) if run_lengths else 0,
        }
    )
    return result


def evaluate_candidate(
    candidate: dict[str, Any],
    raw: dict[str, Any],
    *,
    family: str,
    candidate_id: str,
    seed: int,
    split: str,
    **metadata: Any,
) -> dict[str, Any]:
    """Produce one seed-level decision row."""

    paired = paired_metrics(candidate, raw)
    return {
        "family": family,
        "candidate_id": candidate_id,
        "method": candidate["method"],
        "seed": seed,
        "split": split,
        **metadata,
        **paired,
        **regression_event_comparison(candidate, raw),
        **protection_followup_metrics(candidate, raw),
        "regression_reduction_pp": -paired["micro_matched_regression_difference"],
    }


def candidate_guardrail_pass(
    rows: list[dict[str, Any]],
    guardrails: DecisionGuardrails,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Apply recovery, accuracy, support, and no-delay guardrails."""

    aggregate = aggregate_seed_metrics(rows)
    failures: list[str] = []
    raw_regression_count = sum(
        int(row.get("raw_matched_regression_count", 0)) for row in rows
    )
    raw_recovery_count = sum(
        int(row.get("raw_matched_recovery_count", 0)) for row in rows
    )
    ever_improved = sum(
        row.get("ever_regressed_fraction_change", 0.0) < 0.0 for row in rows
    )
    checks = (
        (
            aggregate.get("regression_improved_seed_count", 0)
            >= guardrails.regression_improved_seed_count_min,
            "regression_improved_seed_count",
        ),
        (
            aggregate.get("micro_matched_regression_difference_mean", math.inf)
            <= -guardrails.required_regression_improvement_pp,
            "regression_improvement",
        ),
        (
            aggregate.get("pooled_matched_recovery_preservation", 0.0)
            >= guardrails.pooled_recovery_min,
            "pooled_recovery",
        ),
        (
            all(
                row.get("seed_recovery_preservation_ratio", 0.0)
                >= guardrails.per_seed_recovery_min
                for row in rows
            ),
            "per_seed_recovery",
        ),
        (
            aggregate.get("final_accuracy_change_mean", -math.inf)
            >= guardrails.final_accuracy_change_pp_min,
            "final_accuracy",
        ),
        (
            aggregate.get("mean_prefix_accuracy_change_mean", -math.inf)
            >= guardrails.mean_prefix_accuracy_change_pp_min,
            "mean_prefix_accuracy",
        ),
        (
            aggregate.get("minimum_prefix_accuracy_change_mean", -math.inf)
            >= guardrails.minimum_prefix_accuracy_change_pp_min,
            "minimum_prefix_accuracy",
        ),
        (
            aggregate.get("first_delay_mean_mean", math.inf)
            <= guardrails.first_correct_delay_mean_max,
            "first_correct_delay",
        ),
        (
            ever_improved >= guardrails.ever_regressed_improved_seed_count_min
            and aggregate.get("ever_regressed_fraction_change_mean", math.inf) < 0.0,
            "ever_regressed_not_merely_delayed",
        ),
        (
            raw_regression_count >= guardrails.minimum_pooled_raw_regression_count,
            "regression_support",
        ),
        (
            raw_recovery_count >= guardrails.minimum_pooled_raw_recovery_count,
            "recovery_support",
        ),
    )
    failures.extend(name for passed, name in checks if not passed)
    aggregate.update(
        {
            "regression_reduction_pp": -aggregate.get(
                "micro_matched_regression_difference_mean", float("nan")
            ),
            "ever_regressed_improved_seed_count": ever_improved,
            "pooled_raw_regression_count": raw_regression_count,
            "pooled_raw_recovery_count": raw_recovery_count,
        }
    )
    return not failures, failures, aggregate


def aggregate_candidate_rows(
    rows: list[dict[str, Any]],
    guardrails: DecisionGuardrails,
    **identity: Any,
) -> dict[str, Any]:
    passed, failures, aggregate = candidate_guardrail_pass(rows, guardrails)
    return {
        **identity,
        "row_scope": "aggregate",
        "guardrail_pass": passed,
        "guardrail_failures": ";".join(failures),
        **aggregate,
    }


def material_dominates(
    challenger: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    regression_margin_pp: float = 0.02,
) -> bool:
    """Require a material regression gain with no safety or accuracy loss."""

    return (
        challenger.get("regression_reduction_pp", -math.inf)
        >= incumbent.get("regression_reduction_pp", -math.inf) + regression_margin_pp
        and challenger.get("pooled_matched_recovery_preservation", -math.inf)
        >= incumbent.get("pooled_matched_recovery_preservation", -math.inf)
        and challenger.get("ever_regressed_fraction_change_mean", math.inf)
        <= incumbent.get("ever_regressed_fraction_change_mean", math.inf)
        and challenger.get("mean_prefix_accuracy_change_mean", -math.inf)
        >= incumbent.get("mean_prefix_accuracy_change_mean", -math.inf)
        and challenger.get("final_accuracy_change_mean", -math.inf)
        >= incumbent.get("final_accuracy_change_mean", -math.inf)
    )


def recommend_branch(
    simple_filter: dict[str, Any] | None,
    output_selector: dict[str, Any] | None,
    hidden_rc_ced: dict[str, Any] | None,
    *,
    sufficient_support: bool = True,
) -> dict[str, Any]:
    """Return the predeclared three-lane branch decision."""

    if not sufficient_support:
        return {
            "recommended_branch": "inconclusive_insufficient_support",
            "status": "inconclusive",
            "rationale": "Decision supports did not meet the predeclared minimum.",
        }
    simple_pass = bool(simple_filter and simple_filter.get("guardrail_pass"))
    output_pass = bool(output_selector and output_selector.get("guardrail_pass"))
    hidden_pass = bool(hidden_rc_ced and hidden_rc_ced.get("guardrail_pass"))
    if simple_pass:
        if output_pass and material_dominates(output_selector, simple_filter):
            branch = "continue_output_only_selector"
            rationale = "The selector materially dominates the passing simple filter."
        else:
            branch = "finish_with_simple_filter"
            rationale = "A simple filter passes and parsimony wins without selector dominance."
    elif output_pass:
        branch = "continue_output_only_selector"
        rationale = "The selector passes while the simple-filter lane does not."
    elif hidden_pass:
        branch = "move_to_hidden_state_rc_ced"
        rationale = "Only the hidden-state causal decoder passes the guardrails."
    elif hidden_rc_ced is None:
        branch = "escalate_to_hidden_state_rc_ced_not_yet_evaluated"
        rationale = "Output-only and simple-filter lanes failed; hidden evidence is required."
    else:
        branch = "no_current_lane_passes"
        rationale = "The implemented hidden-state decoder also fails its guardrails."
    return {
        "recommended_branch": branch,
        "status": "post_hoc_branch_recommendation",
        "rationale": rationale,
    }


class HiddenRCCED(nn.Module):
    """Hidden-state-controlled, class-symmetric causal evidence decoder."""

    def __init__(
        self,
        hidden_mean: Tensor,
        hidden_std: Tensor,
        *,
        alpha_min: float,
        hidden_width: int = 32,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_min < 1.0:
            raise ValueError("alpha_min must be in (0,1)")
        self.alpha_min = float(alpha_min)
        self.register_buffer("hidden_mean", hidden_mean.float())
        self.register_buffer("hidden_std", hidden_std.float().clamp_min(1e-6))
        self.reliability = nn.Sequential(
            nn.Linear(hidden_mean.numel() + 4, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(
        self, raw_logits: Tensor, hidden_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        if hidden_features.shape[:2] != raw_logits.shape[:2]:
            raise ValueError("hidden features and raw logits must align")
        raw = _center_logits(raw_logits)
        probability = raw.softmax(-1)
        top2 = probability.topk(2, dim=-1).values
        confidence = top2[..., 0]
        margin = top2[..., 0] - top2[..., 1]
        entropy = -(
            probability * probability.clamp_min(1e-12).log()
        ).sum(-1) / math.log(raw.shape[-1])
        innovation = torch.zeros_like(confidence)
        innovation[:, 1:] = (
            raw[:, 1:] - raw[:, :-1]
        ).pow(2).mean(-1).sqrt()
        normalized_hidden = (
            hidden_features.float() - self.hidden_mean
        ) / self.hidden_std

        states = [raw[:, 0]]
        alphas = [raw.new_ones(raw.shape[0])]
        for timestep in range(1, raw.shape[1]):
            features = torch.cat(
                [
                    normalized_hidden[:, timestep],
                    confidence[:, timestep, None],
                    margin[:, timestep, None],
                    entropy[:, timestep, None],
                    innovation[:, timestep, None],
                ],
                dim=1,
            )
            alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(
                self.reliability(features).squeeze(-1)
            )
            state = (
                (1.0 - alpha[:, None]) * states[-1]
                + alpha[:, None] * raw[:, timestep]
            )
            states.append(_center_logits(state))
            alphas.append(alpha)
        return torch.stack(states, dim=1), torch.stack(alphas, dim=1)


def hidden_rc_ced_loss(
    decoded_logits: Tensor,
    raw_logits: Tensor,
    targets: Tensor,
    *,
    destructive_weight: float,
    intervention_weight: float,
    tau: float = 0.10,
    gamma: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Asymmetric loss with stop-gradient previous-margin weighting."""

    n, tmax, classes = decoded_logits.shape
    repeated_targets = targets[:, None].expand(-1, tmax)
    cross_entropy = F.cross_entropy(
        decoded_logits.reshape(-1, classes), repeated_targets.reshape(-1)
    )
    probability = decoded_logits.softmax(-1)
    true_probability = probability.gather(
        -1, repeated_targets[..., None]
    ).squeeze(-1)
    other = probability.clone()
    other.scatter_(-1, repeated_targets[..., None], -1.0)
    margin = true_probability - other.max(-1).values
    previous_correct_weight = torch.sigmoid(margin[:, :-1] / tau).detach()
    destructive = (
        previous_correct_weight
        * F.softplus((gamma - margin[:, 1:]) / tau)
    ).mean()
    raw_probability = raw_logits.softmax(-1).detach()
    intervention = (
        raw_probability
        * (
            raw_probability.clamp_min(1e-12).log()
            - decoded_logits.log_softmax(-1)
        )
    ).sum(-1).mean()
    loss = (
        cross_entropy
        + destructive_weight * destructive
        + intervention_weight * intervention
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "destructive_loss": float(destructive.detach().cpu()),
        "minimal_intervention_kl": float(intervention.detach().cpu()),
    }


def train_hidden_rc_ced(
    trajectories: list[dict[str, Any]],
    *,
    alpha_min: float,
    destructive_weight: float,
    intervention_weight: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[HiddenRCCED, dict[str, Any]]:
    """Train one fixed RC-CED configuration on official train trajectories."""

    raw = torch.cat([row["prefix_logits"] for row in trajectories])
    hidden = torch.cat([row["hidden_features"] for row in trajectories])
    targets = torch.cat([row["targets"] for row in trajectories])
    hidden_mean = hidden.mean(dim=(0, 1))
    hidden_std = hidden.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    torch.manual_seed(seed)
    model = HiddenRCCED(
        hidden_mean, hidden_std, alpha_min=alpha_min
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    final: dict[str, float] = {}
    model.train()
    for _ in range(epochs):
        order = torch.randperm(raw.shape[0], generator=generator)
        for start in range(0, raw.shape[0], batch_size):
            index = order[start:start + batch_size]
            raw_batch = raw[index].to(device)
            hidden_batch = hidden[index].to(device)
            target_batch = targets[index].to(device)
            decoded, _ = model(raw_batch, hidden_batch)
            loss, final = hidden_rc_ced_loss(
                decoded,
                raw_batch,
                target_batch,
                destructive_weight=destructive_weight,
                intervention_weight=intervention_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.cpu().eval()
    return model, {
        **final,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "training_seed": seed,
        "alpha_min": alpha_min,
        "destructive_weight": destructive_weight,
        "intervention_weight": intervention_weight,
        "sample_trajectory_count": int(raw.shape[0]),
        "hidden_dimension": int(hidden.shape[-1]),
    }


@torch.no_grad()
def decode_hidden_trajectory(
    base: dict[str, Any],
    model: HiddenRCCED,
    *,
    device: str,
    batch_size: int,
    method: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], Tensor]:
    """Decode one split/seed without labels in the forward path."""

    raw = base["prefix_logits"]
    hidden = base["hidden_features"]
    decoded_parts: list[Tensor] = []
    alpha_parts: list[Tensor] = []
    model = model.to(device).eval()
    for start in range(0, raw.shape[0], batch_size):
        decoded, alpha = model(
            raw[start:start + batch_size].to(device),
            hidden[start:start + batch_size].to(device),
        )
        decoded_parts.append(decoded.cpu())
        alpha_parts.append(alpha.cpu())
    model.cpu()
    trajectory = build_derived_trajectory(
        base,
        torch.cat(decoded_parts),
        method,
        {"family": "hidden_state_rc_ced", **metadata},
    )
    return trajectory, torch.cat(alpha_parts)
