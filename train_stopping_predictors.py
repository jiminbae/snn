#!/usr/bin/env python3
"""Train and evaluate the logit-only Phase A stopping predictors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from utils.stopping_analysis import (confidence_stability_stopping, confidence_stopping,
    cost_aware_oracle, earliest_correct_oracle, earliest_stable_correct_oracle,
    entropy_stopping, global_pareto_frontier, margin_stopping)
from utils.stopping_features import build_causal_features, fit_feature_normalization, normalize_features
from utils.stopping_policy_evaluation import binary_metrics, masked_probability_metrics, multiclass_metrics, policy_metrics
from utils.stopping_predictors import (StoppingMLP, masked_bce_with_logits, multi_horizon_stops,
    one_step_stops, predictor_output_dim, recoverability_stops)
from utils.stopping_targets import action_margin_weights, build_stopping_targets, oracle_future_choice
from utils.trajectory_export import load_torch_compat

PREDICTORS = ["recoverability_final", "final_horizon_gain", "one_step", "multi_horizon"]
THRESHOLDS = [round(i * 0.05, 3) for i in range(1, 20)] + [0.975, 0.99]
TOLERANCES = [0.0, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictors", nargs="+", choices=PREDICTORS, default=PREDICTORS)
    parser.add_argument("--feature-modes", nargs="+", choices=["current_logits", "logit_history"], default=["current_logits", "logit_history"])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.5, 1, 1.5, 2, 3, 4, 6, 8])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selective-weighting", choices=["none", "oracle_margin"], default="none")
    parser.add_argument("--margin-temperature", type=float, default=0.25)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def predictor_loss(name: str, output: torch.Tensor, targets: dict[str, Any], weights: torch.Tensor | None = None) -> torch.Tensor:
    mask = targets["continue_mask"]
    if name == "recoverability_final":
        loss = F.binary_cross_entropy_with_logits(output.squeeze(-1), targets["recoverable_final"], reduction="none")
        effective = mask.float() * (weights if weights is not None else 1.0)
        return (loss * effective).sum() / effective.sum().clamp_min(1.0)
    if name == "final_horizon_gain":
        loss = F.cross_entropy(output.transpose(1, 2), targets["final_horizon_outcome"], reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1)
    if name == "one_step":
        current = F.binary_cross_entropy_with_logits(output[..., 0], targets["current_error"], reduction="none")
        following = F.binary_cross_entropy_with_logits(output[..., 1], targets["next_error"], reduction="none")
        effective = mask.float() * (weights if weights is not None else 1.0)
        return ((current + following) * 0.5 * effective).sum() / effective.sum().clamp_min(1.0)
    return masked_bce_with_logits(output, targets["future_error_target"], targets["future_horizon_mask"],
                                  weights[:, :, None] if weights is not None else None)


def train_predictor(name: str, train_x: torch.Tensor, val_x: torch.Tensor, train_targets: dict[str, Any],
                    val_targets: dict[str, Any], args: argparse.Namespace, output_dir: Path,
                    device: torch.device) -> tuple[StoppingMLP, list[dict[str, Any]]]:
    model = StoppingMLP(train_x.shape[-1], predictor_output_dim(name, train_x.shape[1]), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x, val_x = train_x.to(device), val_x.to(device)
    train_targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in train_targets.items()}
    val_targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_targets.items()}
    weights = None
    if args.selective_weighting == "oracle_margin":
        weights = action_margin_weights(train_targets["error"], args.lambdas[0], args.margin_temperature)
    best, stale, rows = math.inf, 0, []
    checkpoint = output_dir / "best_predictor.pt"
    for epoch in range(1, args.epochs + 1):
        model.train(); permutation = torch.randperm(train_x.shape[0], device=device); total = count = 0.0
        for start in range(0, len(permutation), args.batch_size):
            selected = permutation[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_targets = {k: v[selected] if isinstance(v, torch.Tensor) and v.shape[0] == train_x.shape[0] else v for k, v in train_targets.items()}
            loss = predictor_loss(name, model(train_x[selected]), batch_targets, weights[selected] if weights is not None else None)
            loss.backward(); optimizer.step(); total += loss.item() * len(selected); count += len(selected)
        model.eval()
        with torch.no_grad(): val_loss = predictor_loss(name, model(val_x), val_targets).item()
        rows.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if math.isfinite(val_loss) and val_loss < best:
            best, stale = val_loss, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best,
                        "predictor": name, "config": vars(args)}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience: break
    state = load_torch_compat(checkpoint, device); model.load_state_dict(state["model_state_dict"]); model.eval()
    write_csv(output_dir / "training_metrics.csv", rows)
    return model, rows


def final_gain_stops(probabilities: torch.Tensor, lambda_cost: float) -> torch.Tensor:
    n, tmax, _ = probabilities.shape; stops = torch.full((n,), tmax, dtype=torch.long)
    active = torch.ones(n, dtype=torch.bool)
    gain = probabilities[..., 1] - probabilities[..., 2]
    for t in range(tmax - 1):
        stop = active & (gain[:, t] <= lambda_cost * (tmax - t - 1) / tmax)
        stops[stop] = t + 1; active &= ~stop
    return stops


def predictor_stops(name: str, output: torch.Tensor, value: float) -> torch.Tensor:
    output = output.cpu()
    if name == "recoverability_final": return recoverability_stops(output.sigmoid().squeeze(-1), value)
    if name == "final_horizon_gain": return final_gain_stops(output.softmax(-1), value)
    if name == "one_step": return one_step_stops(output.sigmoid(), value)
    return multi_horizon_stops(output.sigmoid(), value)


def select_operating_points(rows: list[dict[str, Any]], final_accuracy: float, value_key: str) -> list[dict[str, Any]]:
    selected = []
    for tolerance in TOLERANCES:
        eligible = [r for r in rows if r["Accuracy"] >= final_accuracy - tolerance]
        if eligible:
            best = min(eligible, key=lambda r: (r["Average Timestep"], -r["Accuracy"], -float(r[value_key])))
            selected.append({**best, "Accuracy Tolerance PP": tolerance})
    return selected


def baseline_rows(payload: dict[str, Any], lambdas: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, targets = payload["predictions"], payload["targets"]
    n, tmax = predictions.shape; deployable, oracle = [], []
    for t in range(1, tmax + 1): deployable.append(policy_metrics(predictions, targets, torch.full((n,), t), f"Fixed T{t}"))
    for threshold in THRESHOLDS:
        deployable.append(policy_metrics(predictions, targets, confidence_stopping(payload["confidence"], threshold), "Confidence", Threshold=threshold))
        deployable.append(policy_metrics(predictions, targets, confidence_stability_stopping(predictions, payload["confidence"], threshold), "Confidence Stability", Threshold=threshold))
    for threshold in torch.linspace(payload["entropy"].min(), payload["entropy"].max(), 20).tolist():
        deployable.append(policy_metrics(predictions, targets, entropy_stopping(payload["entropy"], threshold), "Entropy", Threshold=threshold))
    for threshold in torch.linspace(payload["margin"].min(), payload["margin"].max(), 20).tolist():
        deployable.append(policy_metrics(predictions, targets, margin_stopping(payload["margin"], threshold), "Margin", Threshold=threshold))
    oracle.append(policy_metrics(predictions, targets, earliest_correct_oracle(payload["correct"]), "Earliest Correct Oracle"))
    oracle.append(policy_metrics(predictions, targets, earliest_stable_correct_oracle(payload["correct"]), "Earliest Stable Correct Oracle"))
    for value in lambdas: oracle.append(policy_metrics(predictions, targets, cost_aware_oracle(payload["correct"], value), "Cost-aware Oracle", Lambda=value))
    return deployable, oracle


def main() -> None:
    args = parse_args(); random.seed(args.policy_seed); np.random.seed(args.policy_seed); torch.manual_seed(args.policy_seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    trajectory_dir, output_dir = Path(args.trajectory_dir), Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {split: load_torch_compat(trajectory_dir / f"{split}_trajectories.pt") for split in ("train", "val", "test")}
    targets = {split: build_stopping_targets(p["prefix_logits"], p["targets"]) for split, p in payloads.items()}
    features, normalization = {}, {}
    for mode in ("current_logits", "logit_history"):
        raw = {split: build_causal_features(p["prefix_logits"], mode) for split, p in payloads.items()}
        normalization[mode] = fit_feature_normalization(raw["train"])
        features[mode] = {split: normalize_features(value, normalization[mode]) for split, value in raw.items()}
    torch.save(normalization, output_dir / "feature_normalization.pt")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    all_predictor_rows, selected_test_rows, predictor_summaries = [], [], {}
    for name in args.predictors:
        mode = "current_logits" if name in {"recoverability_final", "final_horizon_gain"} else "logit_history"
        predictor_dir = output_dir / name; predictor_dir.mkdir(exist_ok=True)
        model, _ = train_predictor(name, features[mode]["train"], features[mode]["val"], targets["train"], targets["val"], args, predictor_dir, device)
        with torch.no_grad(): outputs = {split: model(features[mode][split].to(device)).cpu() for split in ("val", "test")}
        if name == "recoverability_final":
            valid = targets["test"]["continue_mask"]
            metrics = binary_metrics(outputs["test"].sigmoid().squeeze(-1)[valid], targets["test"]["recoverable_final"][valid])
            values, value_key = THRESHOLDS, "Threshold"
        elif name == "final_horizon_gain":
            valid = targets["test"]["continue_mask"]
            metrics = multiclass_metrics(outputs["test"][valid], targets["test"]["final_horizon_outcome"][valid],
                                         targets["test"]["outcome_class_names"])
            values, value_key = args.lambdas, "Lambda"
        else:
            probabilities = outputs["test"].sigmoid()
            if name == "one_step":
                mask = targets["test"]["next_target_mask"]
                metric_probabilities = probabilities[..., 1]
                metric_targets = targets["test"]["next_error"]
            else:
                mask = targets["test"]["future_horizon_mask"]
                metric_probabilities = probabilities
                metric_targets = targets["test"]["future_error_target"]
            metrics = masked_probability_metrics(metric_probabilities, metric_targets, mask)
            metrics["oracle_action_by_lambda"] = {}
            error = targets["test"]["error"]
            for lambda_cost in args.lambdas:
                _, oracle_action = oracle_future_choice(error, lambda_cost)
                if name == "one_step":
                    score = probabilities[..., 0] - probabilities[..., 1]
                else:
                    tmax = probabilities.shape[1]
                    score = torch.zeros_like(error)
                    time_cost = lambda_cost * torch.arange(1, tmax + 1) / tmax
                    for timestep in range(tmax - 1):
                        current = probabilities[:, timestep, timestep] + time_cost[timestep]
                        future = (probabilities[:, timestep, timestep + 1:] + time_cost[timestep + 1:]).min(dim=1).values
                        score[:, timestep] = current - future
                valid = targets["test"]["continue_mask"]
                metrics["oracle_action_by_lambda"][str(lambda_cost)] = binary_metrics(score[valid].sigmoid(), oracle_action[valid].float())
            values, value_key = args.lambdas, "Lambda"
        (predictor_dir / "predictor_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        curves = {}
        for split in ("val", "test"):
            curves[split] = [policy_metrics(payloads[split]["predictions"], payloads[split]["targets"],
                              predictor_stops(name, outputs[split], value), name,
                              **({value_key: value}), Result_Type="post_hoc_diagnostic") for value in values]
        write_csv(predictor_dir / "diagnostic_policy_curve.csv", curves["test"])
        final_val = payloads["val"]["correct"][:, -1].float().mean().item() * 100
        selected_val = select_operating_points(curves["val"], final_val, value_key)
        selected_test = []
        for selected in selected_val:
            value = float(selected[value_key]); match = next(r for r in curves["test"] if float(r[value_key]) == value)
            selected_test.append({**match, "Accuracy Tolerance PP": selected["Accuracy Tolerance PP"], "Result_Type": "validation_selected"})
        write_csv(predictor_dir / "validation_selected_operating_points.csv", selected_val)
        write_csv(predictor_dir / "test_selected_operating_points.csv", selected_test)
        all_predictor_rows += curves["test"]; selected_test_rows += selected_test
        predictor_summaries[name] = min(selected_test, key=lambda r: r["Average Timestep"]) if selected_test else {}
    val_baseline, _ = baseline_rows(payloads["val"], args.lambdas)
    baseline, oracle = baseline_rows(payloads["test"], args.lambdas)
    final_val_accuracy = payloads["val"]["correct"][:, -1].float().mean().item() * 100.0
    baseline_selected_test = []
    for policy in ("Confidence", "Confidence Stability"):
        selected_val = select_operating_points(
            [row for row in val_baseline if row["Policy"] == policy], final_val_accuracy, "Threshold"
        )
        for selected in selected_val:
            match = next(
                row for row in baseline
                if row["Policy"] == policy and float(row["Threshold"]) == float(selected["Threshold"])
            )
            baseline_selected_test.append(
                {**match, "Accuracy Tolerance PP": selected["Accuracy Tolerance PP"],
                 "Result_Type": "validation_selected"}
            )
    fixed = [row for row in baseline if row["Policy"].startswith("Fixed T")]
    write_csv(output_dir / "fixed_timestep_results.csv", fixed); write_csv(output_dir / "baseline_policy_results.csv", baseline)
    write_csv(output_dir / "oracle_results.csv", oracle); all_rows = baseline + all_predictor_rows + oracle
    write_csv(output_dir / "all_policy_results.csv", all_rows)
    write_csv(output_dir / "deployable_pareto_frontier.csv", global_pareto_frontier(baseline + all_predictor_rows))
    write_csv(output_dir / "oracle_pareto_frontier.csv", global_pareto_frontier(oracle))
    confidence_selected = [row for row in baseline_selected_test if row["Policy"] == "Confidence"]
    stability_selected = [row for row in baseline_selected_test if row["Policy"] == "Confidence Stability"]
    best_conf = min(confidence_selected, key=lambda r: r["Average Timestep"]) if confidence_selected else {}
    matched_rows = []
    for method in selected_test_rows:
        tolerance = method["Accuracy Tolerance PP"]
        confidence = next((r for r in confidence_selected if r["Accuracy Tolerance PP"] == tolerance), None)
        stability = next((r for r in stability_selected if r["Accuracy Tolerance PP"] == tolerance), None)
        eligible_oracles = [r for r in oracle if r["Accuracy"] >= method["Accuracy"]]
        oracle_match = min(eligible_oracles, key=lambda r: r["Average Timestep"]) if eligible_oracles else None
        row = {"Method": method["Policy"], "Accuracy Tolerance PP": tolerance,
               "Accuracy": method["Accuracy"], "Average Timestep": method["Average Timestep"]}
        if confidence:
            row["method_to_confidence_timestep_gain"] = confidence["Average Timestep"] - method["Average Timestep"]
        if stability:
            row["method_to_stability_timestep_gain"] = stability["Average Timestep"] - method["Average Timestep"]
        if oracle_match:
            row["method_to_oracle_timestep_gap"] = method["Average Timestep"] - oracle_match["Average Timestep"]
            if confidence:
                denominator = confidence["Average Timestep"] - oracle_match["Average Timestep"]
                row["oracle_gap_closed"] = ((confidence["Average Timestep"] - method["Average Timestep"]) / denominator
                                            if denominator > 0 else "")
        matched_rows.append(row)
    write_csv(output_dir / "matched_accuracy_comparisons.csv", matched_rows)
    multi = predictor_summaries.get("multi_horizon", {}); final = predictor_summaries.get("final_horizon_gain", {})
    beats_final = bool(multi and final and multi["Average Timestep"] < final["Average Timestep"])
    beats_conf = bool(multi and best_conf and multi["Average Timestep"] < best_conf["Average Timestep"])
    recommendation = "go" if beats_final and beats_conf else "weak_go" if beats_final or beats_conf else "no_go"
    metadata = payloads["test"].get("metadata", {})
    summary = {"dataset": metadata.get("dataset"), "backbone_seed": metadata.get("backbone_seed"),
               "checkpoint_epoch": metadata.get("checkpoint_epoch"), "train_samples": len(payloads["train"]["targets"]),
               "validation_samples": len(payloads["val"]["targets"]), "test_samples": len(payloads["test"]["targets"]),
               "fixed_t8_test_accuracy": payloads["test"]["correct"][:, -1].float().mean().item() * 100,
               "best_confidence_result": best_conf,
               "best_confidence_stability_result": min(stability_selected, key=lambda r: r["Average Timestep"]) if stability_selected else {},
               "best_final_horizon_result": final,
               "best_one_step_result": predictor_summaries.get("one_step", {}), "best_multi_horizon_result": multi,
               "multi_horizon_beats_final_horizon": beats_final, "multi_horizon_beats_confidence": beats_conf,
               "recommendation": recommendation, "recommendation_reasons": ["Automatic Phase A diagnostic; inspect seed aggregate before drawing conclusions."]}
    (output_dir / "kill_test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    frontier = global_pareto_frontier(all_rows); plt.figure(figsize=(7, 5))
    for policy in sorted({r["Policy"] for r in frontier}):
        rows = [r for r in frontier if r["Policy"] == policy]; plt.plot([r["Average Timestep"] for r in rows], [r["Accuracy"] for r in rows], "o-", label=policy)
    plt.xlabel("Average Timestep"); plt.ylabel("Accuracy (%)"); plt.grid(alpha=.3); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_average_timestep.png", dpi=160); plt.close()


if __name__ == "__main__": main()
