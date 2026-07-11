#!/usr/bin/env python3
"""Run and aggregate prefix diagnostics across independent seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ANCHOR_TIMESTEPS = (1, 2, 4, 6, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed prefix diagnostics.")
    parser.add_argument("--dataset", choices=["nmnist", "dvs_gesture"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--event-frame-mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--event-downsample-size", type=int, default=None)
    parser.add_argument("--results-dir", default="results/prefix_diagnostics_multiseed")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_data: list[tuple[int, dict, dict, list[dict[str, str]]]] = []

    for seed in args.seeds:
        seed_dir = results_dir / f"seed_{seed}"
        command = [
            sys.executable, str(root / "run_prefix_diagnostics.py"),
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--tmax", str(args.tmax),
            "--device", args.device,
            "--seed", str(seed),
            "--event-frame-mode", args.event_frame_mode,
            "--data-dir", args.data_dir,
            "--results-dir", str(seed_dir),
        ]
        if args.event_downsample_size is not None:
            command.extend(["--event-downsample-size", str(args.event_downsample_size)])
        if args.amp:
            command.append("--amp")
        if args.overwrite:
            command.append("--overwrite")
        for name in ("num_workers", "limit_train_batches", "limit_test_batches"):
            value = getattr(args, name)
            if value is not None:
                command.extend([f"--{name.replace('_', '-')}", str(value)])
        subprocess.run(command, cwd=root, check=True)

        shared = _read_json(seed_dir / f"shared_fixed_lif_T{args.tmax}" / "prefix_metrics.json")
        summary = _read_json(seed_dir / "diagnostic_summary.json")
        with (seed_dir / "prefix_regret.csv").open("r", newline="", encoding="utf-8") as f:
            regrets = list(csv.DictReader(f))
        seed_data.append((seed, shared, summary, regrets))

    metric_names = [
        "negative_temporal_gain", "ever_regressed_rate", "worst_prefix_accuracy",
        "prefix_accuracy_auc", "mean_prefix_regret",
    ]
    metric_rows = []
    for metric in metric_names:
        values = [float(shared[metric]) if metric in shared else float(summary[metric]) for _, shared, summary, _ in seed_data]
        mean, std = _mean_std(values)
        metric_rows.append({"Metric": metric, "Mean": mean, "Std": std})
    for timestep in ANCHOR_TIMESTEPS:
        values = [float(summary[f"specialist_accuracy_t{timestep}"]) for _, _, summary, _ in seed_data]
        mean, std = _mean_std(values)
        metric_rows.append({"Metric": f"specialist_accuracy_t{timestep}", "Mean": mean, "Std": std})
    _write_rows(results_dir / "aggregate_prefix_metrics.csv", metric_rows)

    curve_rows = []
    for timestep in range(1, args.tmax + 1):
        values = [float(shared["prefix_accuracy_curve"][timestep - 1]) for _, shared, _, _ in seed_data]
        mean, std = _mean_std(values)
        curve_rows.append({"Timestep": timestep, "Mean Accuracy": mean, "Std Accuracy": std})
    _write_rows(results_dir / "aggregate_prefix_accuracy_curve.csv", curve_rows)

    regret_rows = []
    for index, timestep in enumerate(ANCHOR_TIMESTEPS):
        shared_values = [float(regrets[index]["Shared Accuracy"]) for _, _, _, regrets in seed_data]
        specialist_values = [float(regrets[index]["Specialist Accuracy"]) for _, _, _, regrets in seed_data]
        regret_values = [float(regrets[index]["Prefix Regret"]) for _, _, _, regrets in seed_data]
        shared_mean, shared_std = _mean_std(shared_values)
        specialist_mean, specialist_std = _mean_std(specialist_values)
        regret_mean, regret_std = _mean_std(regret_values)
        regret_rows.append({
            "Timestep": timestep,
            "Shared Mean": shared_mean, "Shared Std": shared_std,
            "Specialist Mean": specialist_mean, "Specialist Std": specialist_std,
            "Regret Mean": regret_mean, "Regret Std": regret_std,
        })
    _write_rows(results_dir / "aggregate_prefix_regret.csv", regret_rows)
    print(f"Saved multi-seed aggregates to: {results_dir}")


if __name__ == "__main__":
    main()
