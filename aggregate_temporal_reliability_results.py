#!/usr/bin/env python3
"""Aggregate seed-level temporal reliability kill-test results."""

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

METRICS = ("final_accuracy", "mean_prefix_accuracy", "minimum_prefix_accuracy",
           "ever_regressed_fraction", "mean_population_regression", "mean_conditional_regression",
           "correct_to_wrong_transition_count", "destructive_transition_fraction",
           "ever_recovered_fraction", "wrong_to_correct_transition_count",
           "beneficial_transition_fraction", "stable_correct_fraction")


def summarize(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": mean, "sample_standard_deviation": std,
            "individual_seed_values": values, "valid_seed_count": len(values)}


def method_name(summary: dict[str, Any]) -> str:
    mode = summary["temporal_training_mode"]
    if mode == "selective_regression":
        return f"selective_regression_thr{float(summary.get('temporal_confidence_threshold', 0.8)):g}"
    return mode


def recommendation_from_records(records: list[dict[str, Any]]) -> tuple[str, list[str]]:
    by_method_seed = {(record["method"], int(record["seed"])): record for record in records}
    all_seeds = sorted({int(record["seed"]) for record in records})
    matched_seeds = [seed for seed in all_seeds if all((method, seed) in by_method_seed
                     for method in ("final_ce", "all_prefix_ce", "symmetric_kl", "selective_regression_thr0.8"))]
    per_seed = []
    for seed in matched_seeds:
        baseline = by_method_seed.get(("final_ce", seed))
        selective = by_method_seed.get(("selective_regression_thr0.8", seed))
        symmetric = by_method_seed.get(("symmetric_kl", seed))
        all_prefix = by_method_seed.get(("all_prefix_ce", seed))
        preservation = selective["beneficial_transition_fraction"] / max(1e-8, baseline["beneficial_transition_fraction"])
        regression_reduced = selective["ever_regressed_fraction"] < baseline["ever_regressed_fraction"]
        symmetric_ok = not symmetric or selective["destructive_transition_fraction"] <= symmetric["destructive_transition_fraction"]
        accuracy_ok = selective["final_accuracy"] - baseline["final_accuracy"] >= -0.5
        prefix_ok = selective["mean_prefix_accuracy"] >= baseline["mean_prefix_accuracy"]
        all_prefix_dominates = (all_prefix["destructive_transition_fraction"] <= selective["destructive_transition_fraction"]
                                and all_prefix["beneficial_transition_fraction"] >= selective["beneficial_transition_fraction"]
                                and all_prefix["final_accuracy"] >= selective["final_accuracy"]
                                and all_prefix["mean_prefix_accuracy"] >= selective["mean_prefix_accuracy"])
        per_seed.append(regression_reduced and symmetric_ok and preservation >= 0.9 and accuracy_ok and prefix_ok
                        and not all_prefix_dominates)
    baseline_rows = [by_method_seed[("final_ce", seed)] for seed in matched_seeds]
    selective_rows = [by_method_seed[("selective_regression_thr0.8", seed)] for seed in matched_seeds]
    symmetric_rows = [by_method_seed[("symmetric_kl", seed)] for seed in matched_seeds]
    all_prefix_rows = [by_method_seed[("all_prefix_ce", seed)] for seed in matched_seeds]
    strict_aggregate_direction = False
    weak_direction = False
    weak_seed_support = 0
    aggregate_all_prefix_dominates = False
    if baseline_rows and selective_rows and symmetric_rows and all_prefix_rows:
        baseline_mean = {metric: summarize([row[metric] for row in baseline_rows])["mean"] for metric in METRICS}
        selective_mean = {metric: summarize([row[metric] for row in selective_rows])["mean"] for metric in METRICS}
        all_prefix_mean = {metric: summarize([row[metric] for row in all_prefix_rows])["mean"] for metric in METRICS}
        preservation = selective_mean["beneficial_transition_fraction"] / max(1e-8, baseline_mean["beneficial_transition_fraction"])
        symmetric_direction = (not symmetric_rows or selective_mean["destructive_transition_fraction"] <=
                               summarize([row["destructive_transition_fraction"] for row in symmetric_rows])["mean"])
        regression_reduction = selective_mean["ever_regressed_fraction"] < baseline_mean["ever_regressed_fraction"]
        accuracy_change = selective_mean["final_accuracy"] - baseline_mean["final_accuracy"]
        strict_aggregate_direction = (regression_reduction and symmetric_direction and preservation >= 0.9
                                      and accuracy_change >= -0.5
                                      and selective_mean["mean_prefix_accuracy"] >= baseline_mean["mean_prefix_accuracy"])
        weak_direction = regression_reduction and preservation >= 0.8 and accuracy_change >= -1.0
        weak_seed_support = sum(
            row["ever_regressed_fraction"] < by_method_seed[("final_ce", seed)]["ever_regressed_fraction"]
            for seed, row in zip(matched_seeds, selective_rows)
        )
        aggregate_all_prefix_dominates = (
            all_prefix_mean["destructive_transition_fraction"] <= selective_mean["destructive_transition_fraction"]
            and all_prefix_mean["beneficial_transition_fraction"] >= selective_mean["beneficial_transition_fraction"]
            and all_prefix_mean["final_accuracy"] >= selective_mean["final_accuracy"]
            and all_prefix_mean["mean_prefix_accuracy"] >= selective_mean["mean_prefix_accuracy"]
        )
    successes = sum(per_seed)
    if aggregate_all_prefix_dominates:
        result = "no_go"
    elif len(matched_seeds) >= 3 and successes >= 2 and strict_aggregate_direction:
        result = "go"
    elif weak_direction and weak_seed_support >= 2:
        result = "weak_go"
    else:
        result = "no_go"
    return result, [f"Strict primary criteria passed in {successes} of {len(per_seed)} matched seeds.",
                    f"Matched seed count: {len(matched_seeds)} (go requires final_ce, all_prefix_ce, symmetric_kl, and selective results for at least 3).",
                    f"Strict aggregate direction: {strict_aggregate_direction}; weak aggregate direction: {weak_direction} with {weak_seed_support} supporting seeds; all-prefix dominance: {aggregate_all_prefix_dominates}.",
                    "No statistical significance is claimed."]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields: writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    records = []
    for run_dir in map(Path, args.run_dirs):
        summary = json.loads((run_dir / "temporal_reliability_summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary["temporal_confidence_threshold"] = config.get("temporal_confidence_threshold", 0.8)
        records.append({"method": method_name(summary), "seed": int(config["seed"]),
                        **{metric: float(summary[metric]) for metric in METRICS},
                        "prefix_accuracy_curve": summary["prefix_accuracy_curve"]})
    metric_rows = []
    for method in sorted({record["method"] for record in records}):
        selected = [record for record in records if record["method"] == method]
        for metric in METRICS:
            metric_rows.append({"method": method, "metric": metric, **summarize([row[metric] for row in selected])})
    write_csv(output / "aggregate_temporal_metrics.csv", metric_rows)
    baseline = {row["seed"]: row for row in records if row["method"] == "final_ce"}
    comparisons = []
    for row in records:
        if row["method"] == "final_ce" or row["seed"] not in baseline: continue
        reference = baseline[row["seed"]]
        comparisons.append({"method": row["method"], "seed": row["seed"],
            "regression_reduction_vs_final_ce": reference["ever_regressed_fraction"] - row["ever_regressed_fraction"],
            "final_accuracy_change_vs_final_ce": row["final_accuracy"] - reference["final_accuracy"],
            "mean_prefix_accuracy_change_vs_final_ce": row["mean_prefix_accuracy"] - reference["mean_prefix_accuracy"],
            "beneficial_correction_change_vs_final_ce": row["beneficial_transition_fraction"] - reference["beneficial_transition_fraction"],
            "beneficial_transition_preservation_ratio": row["beneficial_transition_fraction"] / max(1e-8, reference["beneficial_transition_fraction"]),
            "ever_recovered_preservation_ratio": row["ever_recovered_fraction"] / max(1e-8, reference["ever_recovered_fraction"])})
    write_csv(output / "seed_method_comparisons.csv", comparisons)
    aggregate_comparisons = []
    comparison_metrics = (
        "regression_reduction_vs_final_ce", "final_accuracy_change_vs_final_ce",
        "mean_prefix_accuracy_change_vs_final_ce", "beneficial_correction_change_vs_final_ce",
        "beneficial_transition_preservation_ratio", "ever_recovered_preservation_ratio",
    )
    for method in sorted({row["method"] for row in comparisons}):
        method_rows = [row for row in comparisons if row["method"] == method]
        for metric in comparison_metrics:
            aggregate_comparisons.append({"method": method, "metric": metric,
                                          **summarize([float(row[metric]) for row in method_rows])})
    write_csv(output / "aggregate_method_comparisons.csv", aggregate_comparisons)
    recommendation, reasons = recommendation_from_records(records)
    (output / "aggregate_summary.json").write_text(json.dumps(
        {"recommendation": recommendation, "recommendation_reasons": reasons,
         "methods": sorted({record["method"] for record in records}), "seed_count": len({record["seed"] for record in records})},
        indent=2, sort_keys=True), encoding="utf-8")
    plt.figure(figsize=(7, 5))
    for method in sorted({record["method"] for record in records}):
        curves = [record["prefix_accuracy_curve"] for record in records if record["method"] == method]
        mean_curve = [sum(values) / len(values) for values in zip(*curves)]
        plt.plot(range(1, len(mean_curve) + 1), mean_curve, "o-", label=method)
    plt.xlabel("Timestep"); plt.ylabel("Accuracy (%)"); plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(output / "prefix_accuracy_comparison.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 5))
    for record in records:
        plt.scatter(record["destructive_transition_fraction"], record["beneficial_transition_fraction"], label=record["method"])
    plt.xlabel("Destructive transitions (%)"); plt.ylabel("Beneficial transitions (%)"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(output / "regression_recovery_tradeoff.png", dpi=160); plt.close()


if __name__ == "__main__": main()
