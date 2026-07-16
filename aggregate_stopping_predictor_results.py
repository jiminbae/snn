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


def main() -> None:
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


if __name__ == "__main__": main()
