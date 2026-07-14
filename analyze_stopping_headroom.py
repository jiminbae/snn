#!/usr/bin/env python3
"""Analyze deployable stopping rules and ground-truth oracle headroom."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.stopping_analysis import (
    confidence_stability_stopping,
    confidence_stopping,
    cost_aware_oracle,
    earliest_correct_oracle,
    earliest_stable_correct_oracle,
    entropy_stopping,
    evaluate_stopping_policy,
    margin_stopping,
    pareto_frontier,
    trajectory_outcomes,
)


POLICY_FIELDS = [
    "Policy", "Threshold", "Lambda", "Accuracy", "Average Timestep",
    "Normalized Average Timestep", "Error Rate", "Number of Samples",
]
OUTCOME_NAMES = [
    "safe_stop", "beneficial_continuation", "destructive_continuation", "futile_continuation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal stopping headroom from saved prefix trajectories.")
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-thresholds", type=float, nargs="+")
    parser.add_argument("--entropy-thresholds", type=float, nargs="+")
    parser.add_argument("--margin-thresholds", type=float, nargs="+")
    parser.add_argument("--cost-lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--min-timestep", type=int, default=1)
    parser.add_argument("--stability-window", type=int, default=2)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def policy_row(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stops: torch.Tensor,
    policy: str,
    threshold: float | None = None,
    lambda_cost: float | None = None,
) -> dict[str, Any]:
    return evaluate_stopping_policy(
        predictions, targets, stops, policy=policy, threshold=threshold, lambda_cost=lambda_cost
    )


def outcome_rows_by_timestep(outcomes: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    tmax = next(iter(outcomes.values())).shape[1]
    return [
        {"Timestep": timestep, **{
            name: float(mask[:, timestep - 1].float().mean().item() * 100.0)
            for name, mask in outcomes.items()
        }}
        for timestep in range(1, tmax + 1)
    ]


def outcome_rows_by_confidence(
    confidence: torch.Tensor,
    outcomes: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= lower) & (confidence <= upper if index == len(edges) - 2 else confidence < upper)
        count = int(selected.sum().item())
        row: dict[str, Any] = {
            "Confidence Bin": f"[{lower:.2f}, {upper:.2f}{']' if index == len(edges) - 2 else ')'}",
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Number of Prefix Predictions": count,
        }
        for name, mask in outcomes.items():
            row[name] = float(mask[selected].float().mean().item() * 100.0) if count else 0.0
        rows.append(row)
    return rows


def plot_tradeoff(rows: list[dict[str, Any]], output: Path) -> None:
    plt.figure(figsize=(7, 5))
    for policy in sorted({str(row["Policy"]) for row in rows}):
        selected = [row for row in rows if row["Policy"] == policy]
        plt.plot(
            [float(row["Average Timestep"]) for row in selected],
            [float(row["Accuracy"]) for row in selected],
            marker="o",
            linewidth=1.5,
            label=policy,
        )
    plt.xlabel("Average Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title("Stopping Accuracy vs. Average Timestep")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_outcomes_by_timestep(rows: list[dict[str, Any]], output: Path) -> None:
    plt.figure(figsize=(7, 5))
    timesteps = [row["Timestep"] for row in rows]
    for name in OUTCOME_NAMES:
        plt.plot(timesteps, [row[name] for row in rows], marker="o", label=name)
    plt.xlabel("Timestep")
    plt.ylabel("Samples (%)")
    plt.title("Trajectory Outcomes by Timestep")
    plt.xticks(timesteps)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_confidence_bins(rows: list[dict[str, Any]], output: Path) -> None:
    x = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    plt.figure(figsize=(9, 5))
    for name in OUTCOME_NAMES:
        values = [row[name] for row in rows]
        plt.bar(x, values, bottom=bottom, label=name)
        bottom = [left + value for left, value in zip(bottom, values)]
    plt.xlabel("Confidence Bin")
    plt.ylabel("Prefix Predictions (%)")
    plt.title("Continuation Outcomes by Confidence")
    plt.xticks(x, [row["Confidence Bin"] for row in rows], rotation=35, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectory_file).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    required = {"prefix_logits", "targets", "predictions", "confidence", "entropy", "margin", "correct"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"Trajectory file is missing fields: {sorted(missing)}")

    logits = payload["prefix_logits"].float()
    targets = payload["targets"].long()
    predictions = payload["predictions"].long()
    confidence = payload["confidence"].float()
    entropy = payload["entropy"].float()
    margin = payload["margin"].float()
    correct = payload["correct"].bool()
    if logits.ndim != 3:
        raise ValueError("prefix_logits must have shape [N,T,C].")
    n, tmax, classes = logits.shape
    minimum = max(1, min(args.min_timestep, tmax))

    confidence_thresholds = args.confidence_thresholds or torch.linspace(0.50, 0.99, 20).tolist()
    entropy_thresholds = args.entropy_thresholds or torch.linspace(0.0, math.log(classes), 20).tolist()
    margin_thresholds = args.margin_thresholds or torch.linspace(0.0, 0.9, 19).tolist()

    fixed_rows = []
    for timestep in range(1, tmax + 1):
        stops = torch.full((n,), timestep, dtype=torch.long)
        fixed_rows.append(policy_row(predictions, targets, stops, "Fixed T", threshold=float(timestep)))

    confidence_rows = [
        policy_row(predictions, targets, confidence_stopping(confidence, value, minimum), "Confidence", value)
        for value in confidence_thresholds
    ]
    entropy_rows = [
        policy_row(predictions, targets, entropy_stopping(entropy, value, minimum), "Entropy", value)
        for value in entropy_thresholds
    ]
    margin_rows = [
        policy_row(predictions, targets, margin_stopping(margin, value, minimum), "Margin", value)
        for value in margin_thresholds
    ]
    stability_rows = [
        policy_row(
            predictions,
            targets,
            confidence_stability_stopping(
                predictions, confidence, value, window=args.stability_window, min_timestep=minimum
            ),
            "Confidence Stability",
            value,
        )
        for value in confidence_thresholds
    ]

    earliest_correct_stops = earliest_correct_oracle(correct, minimum)
    earliest_stable_stops = earliest_stable_correct_oracle(correct, minimum)
    earliest_correct_row = policy_row(predictions, targets, earliest_correct_stops, "Earliest Correct Oracle")
    earliest_stable_row = policy_row(predictions, targets, earliest_stable_stops, "Earliest Stable Correct Oracle")
    cost_rows = [
        policy_row(
            predictions,
            targets,
            cost_aware_oracle(correct, value, minimum),
            "Cost-Aware Oracle",
            lambda_cost=value,
        )
        for value in args.cost_lambdas
    ]
    oracle_rows = [earliest_correct_row, earliest_stable_row, *cost_rows]

    write_csv(output_dir / "fixed_timestep_results.csv", fixed_rows, POLICY_FIELDS)
    write_csv(output_dir / "confidence_stopping.csv", confidence_rows, POLICY_FIELDS)
    write_csv(output_dir / "entropy_stopping.csv", entropy_rows, POLICY_FIELDS)
    write_csv(output_dir / "margin_stopping.csv", margin_rows, POLICY_FIELDS)
    write_csv(output_dir / "confidence_stability_stopping.csv", stability_rows, POLICY_FIELDS)
    write_csv(output_dir / "oracle_stopping.csv", oracle_rows, POLICY_FIELDS)

    outcomes = trajectory_outcomes(correct)
    timestep_outcomes = outcome_rows_by_timestep(outcomes)
    confidence_outcomes = outcome_rows_by_confidence(confidence, outcomes)
    write_csv(
        output_dir / "trajectory_outcomes_by_timestep.csv",
        timestep_outcomes,
        ["Timestep", *OUTCOME_NAMES],
    )
    write_csv(
        output_dir / "trajectory_outcomes_by_confidence_bin.csv",
        confidence_outcomes,
        ["Confidence Bin", "Lower Bound", "Upper Bound", "Number of Prefix Predictions", *OUTCOME_NAMES],
    )

    all_rows = [*fixed_rows, *confidence_rows, *entropy_rows, *margin_rows, *stability_rows, *oracle_rows]
    frontier = pareto_frontier(all_rows)
    write_csv(
        output_dir / "pareto_frontier.csv",
        frontier,
        ["Policy", "Threshold", "Lambda", "Accuracy", "Average Timestep"],
    )

    final_accuracy = float(fixed_rows[-1]["Accuracy"])
    selected_levels: dict[str, Any] = {}
    for fraction in (0.90, 0.95, 0.99):
        target_accuracy = final_accuracy * fraction
        eligible = [row for row in confidence_rows if float(row["Accuracy"]) >= target_accuracy]
        selected_levels[f"{int(fraction * 100)}_percent_of_final"] = (
            min(eligible, key=lambda row: float(row["Average Timestep"])) if eligible else None
        )

    confidence_frontier = [row for row in frontier if row["Policy"] == "Confidence"]
    oracle_candidates = cost_rows + [earliest_correct_row, earliest_stable_row]
    pareto_gap = []
    for row in confidence_frontier:
        eligible = [candidate for candidate in oracle_candidates if candidate["Accuracy"] >= row["Accuracy"]]
        oracle_best = min(eligible, key=lambda candidate: candidate["Average Timestep"]) if eligible else None
        pareto_gap.append({
            "confidence_threshold": row["Threshold"],
            "confidence_accuracy": row["Accuracy"],
            "confidence_average_timestep": row["Average Timestep"],
            "oracle_average_timestep_at_least_accuracy": None if oracle_best is None else oracle_best["Average Timestep"],
            "average_timestep_gap": None if oracle_best is None else row["Average Timestep"] - oracle_best["Average Timestep"],
        })

    summary = {
        "trajectory_file": str(trajectory_path),
        "N": n,
        "T": tmax,
        "C": classes,
        "min_timestep": minimum,
        "fixed_t_accuracy": {f"T{index}": row["Accuracy"] for index, row in enumerate(fixed_rows, start=1)},
        "best_confidence_policy_at_selected_accuracy_levels": selected_levels,
        "earliest_correct_oracle": earliest_correct_row,
        "earliest_stable_correct_oracle": earliest_stable_row,
        "cost_aware_oracle_results": cost_rows,
        "confidence_to_oracle_pareto_gap": pareto_gap,
        "metadata": {
            "accuracy_and_outcome_unit": "percentage_points_0_to_100",
            "average_timestep": "one_based",
            "normalized_cost": "timestep_divided_by_T",
            "oracle_warning": "Earliest-correct, earliest-stable-correct, and cost-aware policies use ground-truth labels and are not deployable stopping policies.",
        },
    }
    with (output_dir / "stopping_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plot_tradeoff(frontier, output_dir / "accuracy_vs_average_timestep.png")
    plot_outcomes_by_timestep(timestep_outcomes, output_dir / "trajectory_outcomes_by_timestep.png")
    plot_confidence_bins(confidence_outcomes, output_dir / "confidence_bin_outcomes.png")
    print(f"Saved stopping headroom analysis to: {output_dir}")


if __name__ == "__main__":
    main()
