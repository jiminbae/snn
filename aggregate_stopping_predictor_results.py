#!/usr/bin/env python3
"""Aggregate Phase A outcomes at the backbone-seed level."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": mean, "sample_standard_deviation": std, "individual_seed_values": values, "valid_seed_count": len(values)}


def aggregate_method_success(rows: list[dict[str, Any]]) -> bool:
    """Require per-seed directional agreement and positive aggregate gains for one method."""
    valid_rows = [row for row in rows if row.get("Meets Test Tolerance", "True") == "True"]
    if not valid_rows:
        return False
    seed_success = 0
    for row in valid_rows:
        final_gain = float(row["Timestep Gain vs Final-Horizon Same Feature"])
        confidence_gain = float(row["Timestep Gain vs Confidence"])
        stability_gain = float(row["Timestep Gain vs Confidence Stability"])
        seed_success += int(final_gain > 0 and (confidence_gain > 0 or stability_gain > 0))
    mean_final = sum(float(row["Timestep Gain vs Final-Horizon Same Feature"]) for row in valid_rows) / len(valid_rows)
    mean_confidence = sum(float(row["Timestep Gain vs Confidence"]) for row in valid_rows) / len(valid_rows)
    mean_stability = sum(float(row["Timestep Gain vs Confidence Stability"]) for row in valid_rows) / len(valid_rows)
    return seed_success >= 2 and mean_final > 0 and (mean_confidence > 0 or mean_stability > 0)


def legacy_main() -> None:
    args = parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    summaries, selected_rows, metric_rows = [], [], []
    for run in map(Path, args.run_dirs):
        root = run / "logit_kill_test" if (run / "logit_kill_test").exists() else run
        summary = json.loads((root / "kill_test_summary.json").read_text(encoding="utf-8")); summaries.append(summary)
        seed = summary["backbone_seed"]
        for predictor in ("recoverability_final", "final_horizon_gain", "one_step", "multi_horizon"):
            selected = root / predictor / "test_selected_operating_points.csv"
            if selected.exists():
                selected_rows += [{**row, "Backbone Seed": seed, "Predictor": predictor} for row in read_csv(selected)]
            metrics = root / predictor / "predictor_metrics.json"
            if metrics.exists():
                payload = json.loads(metrics.read_text(encoding="utf-8"))
                metric_rows += [{"Backbone Seed": seed, "Predictor": predictor, "Metric": key, "Value": value}
                                for key, value in payload.items() if isinstance(value, (int, float))]
    aggregate_selected = []
    selected_groups = sorted({(r["Predictor"], r.get("Accuracy Tolerance PP", "")) for r in selected_rows})
    for predictor, tolerance in selected_groups:
        group = [r for r in selected_rows if r["Predictor"] == predictor and r.get("Accuracy Tolerance PP", "") == tolerance]
        for metric in ("Accuracy", "Average Timestep"):
            values = [float(r[metric]) for r in group if r.get(metric, "") != ""]
            if values:
                aggregate_selected.append({"Predictor": predictor, "Accuracy Tolerance PP": tolerance,
                                           "Metric": metric, **summarize(values)})
    aggregate_metrics = []
    for predictor, metric in sorted({(r["Predictor"], r["Metric"]) for r in metric_rows}):
        values = [float(r["Value"]) for r in metric_rows if r["Predictor"] == predictor and r["Metric"] == metric
                  and math.isfinite(float(r["Value"]))]
        if values:
            aggregate_metrics.append({"Predictor": predictor, "Metric": metric, **summarize(values)})
    write_csv(output / "aggregate_selected_operating_points.csv", aggregate_selected)
    write_csv(output / "aggregate_predictor_metrics.csv", aggregate_metrics)
    aggregate = {"seed_count": len(summaries), "recommendations": [s["recommendation"] for s in summaries]}
    for key in ("fixed_t8_test_accuracy",): aggregate[key] = summarize([float(s[key]) for s in summaries])
    for method in ("best_confidence_result", "best_final_horizon_result", "best_one_step_result", "best_multi_horizon_result"):
        for metric in ("Accuracy", "Average Timestep"):
            values = [float(s[method][metric]) for s in summaries if s.get(method) and metric in s[method]]
            if values: aggregate[f"{method}.{metric}"] = summarize(values)
    (output / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    plt.figure(figsize=(7, 5))
    for predictor in sorted({row["Predictor"] for row in selected_rows}):
        rows = [r for r in selected_rows if r["Predictor"] == predictor]
        plt.scatter([float(r["Average Timestep"]) for r in rows], [float(r["Accuracy"]) for r in rows], label=predictor)
    plt.xlabel("Average Timestep"); plt.ylabel("Accuracy (%)"); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(output / "aggregate_accuracy_vs_timestep.png", dpi=160); plt.close()


def main() -> None:
    args = parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    summaries, selected_rows, metric_rows, comparison_rows = [], [], [], []
    for run in map(Path, args.run_dirs):
        root = run / "logit_kill_test" if (run / "logit_kill_test").exists() else run
        summary = json.loads((root / "kill_test_summary.json").read_text(encoding="utf-8")); summaries.append(summary)
        seed = summary["backbone_seed"]
        selected_path = root / "validation_selected_test_results.csv"
        if selected_path.exists():
            selected_rows += [{**row, "Backbone Seed": seed} for row in read_csv(selected_path)]
        comparisons = root / "tolerance_matched_comparisons.csv"
        if comparisons.exists():
            comparison_rows += [{**row, "Backbone Seed": seed} for row in read_csv(comparisons)]
        for method_dir in sorted(path for path in root.iterdir() if path.is_dir() and "__" in path.name):
            metrics_path = method_dir / "predictor_metrics.json"
            if not metrics_path.exists(): continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    metric_rows.append({"Backbone Seed": seed, "Predictor": payload["Predictor"],
                                        "Feature Mode": payload["Feature Mode"], "Method": payload["Method"],
                                        "Metric": key, "Value": value})
    aggregate_selected = []
    groups = sorted({(r.get("Predictor", ""), r.get("Feature Mode", ""), r.get("Method", ""),
                      r.get("Accuracy Tolerance PP", "")) for r in selected_rows})
    for predictor, mode, method, tolerance in groups:
        group = [r for r in selected_rows if (r.get("Predictor", ""), r.get("Feature Mode", ""),
                 r.get("Method", ""), r.get("Accuracy Tolerance PP", "")) == (predictor, mode, method, tolerance)]
        for metric in ("Accuracy", "Average Timestep"):
            values = [float(row[metric]) for row in group if row.get(metric, "") != ""]
            if values:
                aggregate_selected.append({"Predictor": predictor, "Feature Mode": mode, "Method": method,
                    "Accuracy Tolerance PP": tolerance, "Metric": metric, **summarize(values)})
    aggregate_metrics = []
    for predictor, mode, method, metric in sorted({(r["Predictor"], r["Feature Mode"], r["Method"], r["Metric"]) for r in metric_rows}):
        values = [float(r["Value"]) for r in metric_rows if (r["Predictor"], r["Feature Mode"], r["Method"], r["Metric"])
                  == (predictor, mode, method, metric) and math.isfinite(float(r["Value"]))]
        aggregate_metrics.append({"Predictor": predictor, "Feature Mode": mode, "Method": method,
                                  "Metric": metric, **summarize(values)})
    aggregate_comparisons = []
    comparison_metrics = ["Timestep Gain vs Confidence", "Timestep Gain vs Confidence Stability",
                          "Timestep Gain vs Final-Horizon Same Feature", "Timestep Gain vs Same-Predictor Current Logits"]
    comparison_groups = sorted({(r["Method"], r["Accuracy Tolerance PP"]) for r in comparison_rows})
    for method, tolerance in comparison_groups:
        group = [r for r in comparison_rows if r["Method"] == method and r["Accuracy Tolerance PP"] == tolerance]
        for metric in comparison_metrics:
            values = [float(r[metric]) for r in group if r.get(metric, "") != ""]
            if values:
                aggregate_comparisons.append({"Method": method, "Accuracy Tolerance PP": tolerance,
                                              "Metric": metric, **summarize(values)})
    write_csv(output / "aggregate_selected_operating_points.csv", aggregate_selected)
    write_csv(output / "aggregate_predictor_metrics.csv", aggregate_metrics)
    write_csv(output / "aggregate_tolerance_comparisons.csv", aggregate_comparisons)
    successful_tolerances = 0
    for tolerance in ("0.0", "0.5", "1.0", "2.0"):
        rows = [r for r in comparison_rows if r["Accuracy Tolerance PP"] == tolerance
                and r["Predictor"] == "multi_horizon"]
        method_groups = {method: [row for row in rows if row["Method"] == method]
                         for method in {row["Method"] for row in rows}}
        successful_tolerances += int(any(aggregate_method_success(group) for group in method_groups.values()))
    recommendation = "go" if len(summaries) >= 3 and successful_tolerances >= 2 else "weak_go" if successful_tolerances >= 1 else "no_go"
    def comparison_summary(method: str, metric: str) -> dict[str, Any]:
        return {str(tolerance): next((row for row in aggregate_comparisons
                if row["Method"] == method and row["Metric"] == metric
                and str(row["Accuracy Tolerance PP"]) == str(tolerance)), {})
                for tolerance in (0.0, 0.5, 1.0, 2.0)}

    aggregate = {"seed_count": len(summaries), "successful_tolerance_count": successful_tolerances,
                 "recommendation": recommendation, "recommendation_reasons": [
                     f"Criteria reproduced in at least two seeds at {successful_tolerances} tolerances.",
                     "No statistical significance is claimed."],
                 "core_comparisons": {
                     "multi_horizon_vs_final_horizon__current_logits": comparison_summary(
                         "multi_horizon__current_logits", "Timestep Gain vs Final-Horizon Same Feature"),
                     "multi_horizon_vs_final_horizon__logit_history": comparison_summary(
                         "multi_horizon__logit_history", "Timestep Gain vs Final-Horizon Same Feature"),
                     "multi_horizon_logit_history_vs_current_logits": comparison_summary(
                         "multi_horizon__logit_history", "Timestep Gain vs Same-Predictor Current Logits"),
                     "multi_horizon_vs_confidence": comparison_summary(
                         "multi_horizon__logit_history", "Timestep Gain vs Confidence"),
                     "multi_horizon_vs_confidence_stability": comparison_summary(
                         "multi_horizon__logit_history", "Timestep Gain vs Confidence Stability")}}
    (output / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    plt.figure(figsize=(7, 5))
    for method in sorted({row.get("Method", "") for row in selected_rows}):
        rows = [row for row in selected_rows if row.get("Method", "") == method]
        plt.scatter([float(row["Average Timestep"]) for row in rows], [float(row["Accuracy"]) for row in rows], label=method)
    plt.xlabel("Average Timestep"); plt.ylabel("Accuracy (%)"); plt.grid(alpha=.3); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(output / "aggregate_accuracy_vs_timestep.png", dpi=160); plt.close()


if __name__ == "__main__": main()
